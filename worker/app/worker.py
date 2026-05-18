import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

import redis


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE        = "/logs/worker.log"
LOG_MAX_BYTES   = 5 * 1024 * 1024
LOG_BACKUP_COUNT = 3

os.makedirs("/logs", exist_ok=True)


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[36m",
        logging.INFO:     "\033[32m",
        logging.WARNING:  "\033[33m",
        logging.ERROR:    "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record):
        color  = self.COLORS.get(record.levelno, self.RESET)
        thread = f"{record.threadName:<14}"
        level  = f"{record.levelname:<8}"
        msg    = super().format(record)
        msg    = msg.replace(record.levelname, f"{color}{self.BOLD}{level}{self.RESET}", 1)
        msg    = msg.replace(record.threadName, f"\033[34m{thread}{self.RESET}", 1)
        return msg


def setup_logging() -> logging.Logger:
    logger  = logging.getLogger("worker")
    logger.setLevel(LOG_LEVEL)
    fmt     = "%(asctime)s | %(levelname)-8s | [%(threadName)-14s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


log = setup_logging()


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST      = os.getenv("REDIS_HOST", "redis")
REDIS_PORT      = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT       = int(os.getenv("GPU_COUNT", "1"))
JOB_TIMEOUT     = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
JOB_TTL         = 60 * 60 * 24
JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ── Redis helpers ─────────────────────────────────────────────────────────────
def update_job(job_id: str, updates: dict):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return
    job = json.loads(raw)
    job.update(updates)
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(job))


def get_job(job_id: str) -> dict | None:
    raw = r.get(f"job:{job_id}")
    return json.loads(raw) if raw else None


# ── GPU management ────────────────────────────────────────────────────────────
def get_free_gpu() -> int | None:
    for gpu_id in range(GPU_COUNT):
        key = f"gpu:{gpu_id}"
        try:
            with r.pipeline() as pipe:
                while True:
                    try:
                        pipe.watch(key)
                        raw = pipe.get(key)
                        if not raw:
                            pipe.unwatch()
                            break
                        gpu = json.loads(raw)
                        if gpu["status"] != "idle":
                            pipe.unwatch()
                            break
                        pipe.multi()
                        gpu["status"]     = "busy"
                        gpu["updated_at"] = datetime.now(timezone.utc).isoformat()
                        pipe.set(key, json.dumps(gpu))
                        pipe.execute()
                        return gpu_id
                    except redis.WatchError:
                        continue
        except Exception as e:
            log.error(f"Error checking gpu:{gpu_id}: {e}")
    return None


def reserve_gpu_for_job(gpu_id: int, job_id: str):
    key = f"gpu:{gpu_id}"
    raw = r.get(key)
    if not raw:
        return
    gpu = json.loads(raw)
    gpu["status"]     = "busy"
    gpu["job_id"]     = job_id
    gpu["updated_at"] = datetime.now(timezone.utc).isoformat()
    r.set(key, json.dumps(gpu))


def release_gpu(gpu_id: int):
    key = f"gpu:{gpu_id}"
    raw = r.get(key)
    if not raw:
        return
    gpu = json.loads(raw)
    gpu["status"]     = "idle"
    gpu["job_id"]     = None
    gpu["updated_at"] = datetime.now(timezone.utc).isoformat()
    r.set(key, json.dumps(gpu))
    log.debug(f"GPU {gpu_id} released → idle")


# ── Startup checks ────────────────────────────────────────────────────────────
def check_real_gpu_count():
    log.info("Detecting available GPUs...")
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--gpus", "all",
             "nvidia/cuda:12.3.2-base-ubuntu22.04",
             "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        lines      = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        real_count = len(lines)
        log.info(f"Detected GPUs: {real_count}")
        for i, name in enumerate(lines):
            log.info(f"  [{i}] {name}")
        if real_count == 0:
            log.warning("No GPUs detected! Check NVIDIA drivers and Container Toolkit.")
        elif real_count < GPU_COUNT:
            log.warning(f"GPU_COUNT={GPU_COUNT} but only {real_count} GPU(s) found.")
        elif real_count > GPU_COUNT:
            log.info(f"{real_count} GPUs available, using {GPU_COUNT} per config.")
        else:
            log.info(f"GPU_COUNT={GPU_COUNT} matches real GPU count ✓")
        return real_count
    except Exception as e:
        log.error(f"Could not detect GPU count: {e}")
        return None


def recover_stale_gpus():
    log.info("Checking for stale GPU reservations...")
    recovered = 0
    for gpu_id in range(GPU_COUNT):
        key = f"gpu:{gpu_id}"
        raw = r.get(key)
        if not raw:
            continue
        gpu = json.loads(raw)
        if gpu["status"] != "busy":
            continue

        job_id         = gpu.get("job_id")
        container_name = f"job-{job_id}" if job_id else None
        container_running = False

        if container_name:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                capture_output=True, text=True,
            )
            container_running = result.stdout.strip() == "true"

        if not container_running:
            log.warning(f"GPU {gpu_id} stuck as busy (job={job_id}) — releasing")
            release_gpu(gpu_id)
            recovered += 1
            if job_id:
                job = get_job(job_id)
                if job and job.get("status") == "running":
                    update_job(job_id, {
                        "status":       "failed",
                        "stderr":       "Worker restarted — job was lost",
                        "finished_at":  datetime.now(timezone.utc).isoformat(),
                        "container_name": None,
                    })
                    log.warning(f"Job {job_id[:8]} marked as failed (worker restart)")

    if recovered == 0:
        log.info("No stale reservations found ✓")
    else:
        log.info(f"Recovered {recovered} stale GPU reservation(s)")


# ── Log streaming helpers ─────────────────────────────────────────────────────
def _push_log(log_key: str, stream_type: str, line: str):
    entry = json.dumps({"type": stream_type, "line": line})
    r.rpush(log_key, entry)
    r.expire(log_key, JOB_TTL)


def _push_end(log_key: str, status: str):
    entry = json.dumps({"type": "end", "status": status})
    r.rpush(log_key, entry)
    r.expire(log_key, JOB_TTL)


def _stream_reader(stream, stream_type: str, lines_list: list, log_key: str):
    """Thread target: reads a subprocess stream line by line and pushes to Redis."""
    for raw_line in stream:
        line = raw_line.rstrip("\n")
        lines_list.append(line)
        _push_log(log_key, stream_type, line)


# ── Repo cloning ──────────────────────────────────────────────────────────────
def _clone_repo(short_id: str, repo_url: str, dest: str):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    log.info(f"[{short_id}] Cloning {repo_url}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, dest],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed:\n{result.stderr[:500]}")
    log.info(f"[{short_id}] Repo cloned ✓")


# ── Artifact collection ───────────────────────────────────────────────────────
def _collect_artifacts(output_dir: str) -> list[dict]:
    if not os.path.exists(output_dir):
        return []
    artifacts = []
    for root, _, files in os.walk(output_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel   = os.path.relpath(fpath, output_dir)
            artifacts.append({"name": rel, "size": os.path.getsize(fpath)})
    return artifacts


# ── Job execution ─────────────────────────────────────────────────────────────
def execute_job(job_id: str, gpu_id: int):
    job = get_job(job_id)
    if not job:
        log.error(f"Job {job_id[:8]} not found in Redis — aborting")
        release_gpu(gpu_id)
        return

    short_id        = job_id[:8]
    image           = job["image"]
    cpus            = job.get("cpus", 1.0)
    memory          = job.get("memory", "2g")
    code            = job.get("code", "")
    repo_url        = job.get("repo_url")
    has_dataset     = job.get("has_dataset", False)
    entrypoint      = job.get("entrypoint", "main.py")
    requirements    = job.get("requirements", "")

    log.info(f"[{short_id}] Starting on GPU {gpu_id} | image={image.split('/')[-1]} "
             f"cpus={cpus} mem={memory}")

    update_job(job_id, {
        "status":         "running",
        "gpu_id":         gpu_id,
        "container_name": f"job-{job_id}",
        "started_at":     datetime.now(timezone.utc).isoformat(),
    })
    reserve_gpu_for_job(gpu_id, job_id)

    container_name  = f"job-{job_id}"
    code_file       = f"/jobs/{job_id}.py"
    repo_dir        = f"/jobs/{job_id}_repo"
    output_dir      = f"/jobs/{job_id}_output"
    requirements_fp = f"/jobs/{job_id}_requirements.txt"
    log_key         = f"job:logs:{job_id}"

    try:
        os.makedirs("/jobs", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # ── Prepare code / repo ───────────────────────────────────────────────
        if repo_url:
            _clone_repo(short_id, repo_url, repo_dir)
            python_target = f"/workspace/{job_id}_repo/{entrypoint}"
        else:
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code)
            python_target = f"/workspace/{job_id}.py"

        # ── Prepare requirements (if any) ─────────────────────────────────────
        if requirements:
            with open(requirements_fp, "w", encoding="utf-8") as f:
                f.write(requirements)
            log.info(f"[{short_id}] Requirements provided — pip install will run "
                     f"(network: bridge for this job)")

        # ── Decide network mode ───────────────────────────────────────────────
        # Default: --network none (max isolation).
        # If requirements are provided, we need internet for pip install.
        network_mode = "bridge" if requirements else "none"

        # ── Build python entrypoint (wrapped in sh -c if requirements) ────────
        if requirements:
            req_path = f"/workspace/{job_id}_requirements.txt"
            inner = (
                f"echo '== installing requirements ==' && "
                f"pip install --quiet --disable-pip-version-check "
                f"--no-cache-dir -r {shlex.quote(req_path)} && "
                f"echo '== running user code ==' && "
                f"python -u {shlex.quote(python_target)}"
            )
            workspace_cmd = ["sh", "-c", inner]
        else:
            workspace_cmd = ["python", "-u", python_target]

        # ── Build docker command ──────────────────────────────────────────────
        docker_cmd = [
            "docker", "run", "--rm",
            "--name",         container_name,
            "--cpus",         str(cpus),
            "--memory",       memory,
            "--gpus",         f"device={gpu_id}",
            "-v",             f"{JOB_VOLUME_NAME}:/workspace",
            "--network",      network_mode,
            # Security hardening
            "--cap-drop",     "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit",   "200",
            # Per-job paths as env vars (student uses os.environ)
            "-e", f"OUTPUT_DIR=/workspace/{job_id}_output",
        ]
        if has_dataset:
            docker_cmd += ["-e", f"DATA_DIR=/workspace/{job_id}_data"]

        docker_cmd += [image] + workspace_cmd

        process = subprocess.Popen(
            docker_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # ── Stream stdout / stderr via threads ────────────────────────────────
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []

        t_out = threading.Thread(
            target=_stream_reader,
            args=(process.stdout, "stdout", stdout_lines, log_key),
            daemon=True,
        )
        t_err = threading.Thread(
            target=_stream_reader,
            args=(process.stderr, "stderr", stderr_lines, log_key),
            daemon=True,
        )
        t_out.start()
        t_err.start()

        # ── Monitor loop (cancel / timeout) ──────────────────────────────────
        start_time = time.time()
        cancelled  = False

        while process.poll() is None:
            current = get_job(job_id)
            if current and current.get("cancel_requested"):
                log.info(f"[{short_id}] Cancel requested — killing container")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                cancelled = True
                break

            elapsed = int(time.time() - start_time)
            if elapsed > JOB_TIMEOUT:
                log.warning(f"[{short_id}] Timeout after {JOB_TIMEOUT}s — killing")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                t_out.join(timeout=5)
                t_err.join(timeout=5)
                _push_end(log_key, "failed")
                update_job(job_id, {
                    "stdout":       "\n".join(stdout_lines),
                    "stderr":       "\n".join(stderr_lines) + f"\nJob timed out after {JOB_TIMEOUT}s",
                    "return_code":  -1,
                    "status":       "failed",
                    "finished_at":  datetime.now(timezone.utc).isoformat(),
                    "container_name": None,
                })
                return

            if elapsed > 0 and elapsed % 30 == 0:
                log.debug(f"[{short_id}] Still running... ({elapsed}s / {JOB_TIMEOUT}s)")

            time.sleep(0.5)

        t_out.join(timeout=5)
        t_err.join(timeout=5)
        elapsed = round(time.time() - start_time, 2)

        # ── Cancelled ─────────────────────────────────────────────────────────
        if cancelled:
            log.info(f"[{short_id}] Cancelled after {elapsed}s")
            _push_end(log_key, "cancelled")
            update_job(job_id, {
                "stdout":       "\n".join(stdout_lines),
                "stderr":       "\n".join(stderr_lines) + "\nJob was cancelled by user",
                "return_code":  -1,
                "status":       "cancelled",
                "finished_at":  datetime.now(timezone.utc).isoformat(),
                "container_name": None,
            })
            return

        # ── Completed / Failed ────────────────────────────────────────────────
        status = "completed" if process.returncode == 0 else "failed"

        if status == "completed":
            log.info(f"[{short_id}] ✓ Completed in {elapsed}s (GPU {gpu_id})")
        else:
            log.error(f"[{short_id}] ✗ Failed in {elapsed}s (exit {process.returncode})")
            if stderr_lines:
                log.error(f"[{short_id}] stderr preview:\n" + "\n".join(stderr_lines[:3]))

        artifacts = _collect_artifacts(output_dir)
        if artifacts:
            log.info(f"[{short_id}] {len(artifacts)} artifact(s) saved")

        _push_end(log_key, status)
        update_job(job_id, {
            "stdout":       "\n".join(stdout_lines),
            "stderr":       "\n".join(stderr_lines),
            "return_code":  process.returncode,
            "status":       status,
            "finished_at":  datetime.now(timezone.utc).isoformat(),
            "container_name": None,
            "artifacts":    artifacts,
        })

    except Exception as e:
        log.exception(f"[{short_id}] Unexpected error: {e}")
        _push_end(log_key, "failed")
        update_job(job_id, {
            "stdout":       "",
            "stderr":       str(e),
            "return_code":  -1,
            "status":       "failed",
            "finished_at":  datetime.now(timezone.utc).isoformat(),
            "container_name": None,
        })

    finally:
        release_gpu(gpu_id)
        for path in (code_file, repo_dir, requirements_fp):
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        log.debug(f"[{short_id}] Cleanup done")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("GPU Job Scheduler Worker")
    log.info(f"GPU_COUNT={GPU_COUNT} | JOB_TIMEOUT={JOB_TIMEOUT}s | LOG_LEVEL={LOG_LEVEL}")
    log.info("=" * 60)

    check_real_gpu_count()
    recover_stale_gpus()
    log.info("Waiting for jobs...")

    while True:
        try:
            item = r.blpop("job_queue", timeout=5)
            if not item:
                continue

            _, job_id = item
            raw = r.get(f"job:{job_id}")
            if not raw:
                log.warning(f"Job {job_id[:8]} not found in Redis — skipping")
                continue

            job = json.loads(raw)
            if job.get("status") == "cancelled" or job.get("cancel_requested"):
                log.info(f"Job {job_id[:8]} already cancelled — skipping")
                continue

            log.info(f"Job {job_id[:8]} dequeued | runtime={job.get('runtime')} "
                     f"cpus={job.get('cpus')} mem={job.get('memory')}")

            # Wait for a free GPU
            gpu_id     = None
            wait_start = time.time()
            while gpu_id is None:
                current = get_job(job_id)
                if not current or current.get("cancel_requested"):
                    log.info(f"Job {job_id[:8]} cancelled while waiting for GPU")
                    break
                gpu_id = get_free_gpu()
                if gpu_id is None:
                    waited = int(time.time() - wait_start)
                    if waited % 10 == 0 and waited > 0:
                        log.info(f"Job {job_id[:8]} waiting for free GPU... ({waited}s)")
                    time.sleep(1)

            if gpu_id is None:
                continue

            t = threading.Thread(
                target=execute_job,
                args=(job_id, gpu_id),
                daemon=True,
                name=f"job-{job_id[:8]}",
            )
            t.start()
            log.info(f"Job {job_id[:8]} dispatched → GPU {gpu_id} "
                     f"(active threads: {threading.active_count() - 1})")

        except Exception as e:
            log.exception(f"Main loop error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()
