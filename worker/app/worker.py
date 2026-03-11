import subprocess
import os
import time
import json
import threading
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import redis


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = "/logs/worker.log"

os.makedirs("/logs", exist_ok=True)

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        thread = f"{record.threadName:<14}"
        level  = f"{record.levelname:<8}"
        msg    = super().format(record)
        msg = msg.replace(record.levelname, f"{color}{self.BOLD}{level}{self.RESET}", 1)
        msg = msg.replace(record.threadName, f"\033[34m{thread}{self.RESET}", 1)
        return msg

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("worker")
    logger.setLevel(LOG_LEVEL)

    fmt = "%(asctime)s | %(levelname)-8s | [%(threadName)-14s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(fmt, datefmt=datefmt))

    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger

log = setup_logging()



REDIS_HOST     = os.getenv("REDIS_HOST", "redis")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT      = int(os.getenv("GPU_COUNT", "1"))
JOB_TIMEOUT    = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)



def update_job(job_id: str, updates: dict):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return
    job = json.loads(raw)
    job.update(updates)
    r.setex(f"job:{job_id}", 60 * 60 * 24, json.dumps(job))


def get_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return None
    return json.loads(raw)


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
                        gpu["status"] = "busy"
                        gpu["job_id"] = None
                        gpu["updated_at"] = datetime.utcnow().isoformat()
                        pipe.set(key, json.dumps(gpu))
                        pipe.execute()
                        return gpu_id

                    except redis.WatchError:
                        continue
        except Exception as e:
            log.error(f"Error checking gpu:{gpu_id}: {e}")
            continue

    return None


def reserve_gpu_for_job(gpu_id: int, job_id: str):
    key = f"gpu:{gpu_id}"
    raw = r.get(key)
    if not raw:
        return
    gpu = json.loads(raw)
    gpu["status"] = "busy"
    gpu["job_id"] = job_id
    gpu["updated_at"] = datetime.utcnow().isoformat()
    r.set(key, json.dumps(gpu))


def release_gpu(gpu_id: int):
    key = f"gpu:{gpu_id}"
    raw = r.get(key)
    if not raw:
        return
    gpu = json.loads(raw)
    gpu["status"] = "idle"
    gpu["job_id"] = None
    gpu["updated_at"] = datetime.utcnow().isoformat()
    r.set(key, json.dumps(gpu))
    log.debug(f"GPU {gpu_id} released → idle")


def check_real_gpu_count():

    log.info("Detecting available GPUs...")
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--gpus", "all",
                "nvidia/cuda:12.3.2-base-ubuntu22.04",
                "nvidia-smi", "--query-gpu=name", "--format=csv,noheader",
            ],
            capture_output=True, text=True, timeout=30
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        real_count = len(lines)

        log.info(f"Detected GPUs: {real_count}")
        for i, name in enumerate(lines):
            log.info(f"  [{i}] {name}")

        if real_count == 0:
            log.warning("No GPUs detected! Check NVIDIA drivers and Container Toolkit.")
        elif real_count < GPU_COUNT:
            log.warning(
                f"GPU_COUNT={GPU_COUNT} but only {real_count} GPU(s) found. "
                f"Update GPU_COUNT in .env to avoid errors."
            )
        elif real_count > GPU_COUNT:
            log.info(
                f"{real_count} GPUs available but GPU_COUNT={GPU_COUNT}. "
                f"Only {GPU_COUNT} GPU(s) will be used by the scheduler."
            )
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

        job_id = gpu.get("job_id")
        container_name = f"job-{job_id}" if job_id else None
        container_running = False

        if container_name:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
                capture_output=True, text=True
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
                        "status": "failed",
                        "stderr": "Worker restarted — job was lost",
                        "finished_at": datetime.utcnow().isoformat(),
                        "container_name": None,
                    })
                    log.warning(f"Job {job_id[:8]} marked as failed (worker restart)")

    if recovered == 0:
        log.info("No stale reservations found ✓")
    else:
        log.info(f"Recovered {recovered} stale GPU reservation(s)")


def execute_job(job_id: str, code: str, gpu_id: int, image: str, cpus: float, memory: str):
    short_id = job_id[:8]
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"
    container_name = f"job-{job_id}"

    log.info(f"[{short_id}] Starting on GPU {gpu_id} | image={image.split('/')[-1]} "
             f"cpus={cpus} mem={memory}")

    update_job(job_id, {
        "status": "running",
        "gpu_id": gpu_id,
        "container_name": container_name,
        "started_at": datetime.utcnow().isoformat(),
    })
    reserve_gpu_for_job(gpu_id, job_id)

    try:
        os.makedirs("/jobs", exist_ok=True)
        with open(worker_job_file, "w", encoding="utf-8") as f:
            f.write(code)

        process = subprocess.Popen(
            [
                "docker", "run", "--rm",
                "--name", container_name,
                "--cpus", str(cpus),
                "--memory", memory,
                "--gpus", f"device={gpu_id}",
                "-v", f"{JOB_VOLUME_NAME}:/workspace",
                "--network", "none",
                image,
                "python", f"/workspace/{filename}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        start_time = time.time()
        cancelled = False

        while True:
            if process.poll() is not None:
                break

            job = get_job(job_id)
            if job and job.get("cancel_requested"):
                log.info(f"[{short_id}] Cancel requested — killing container")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                cancelled = True
                break

            elapsed = int(time.time() - start_time)
            if elapsed > JOB_TIMEOUT:
                log.warning(f"[{short_id}] Timeout after {JOB_TIMEOUT}s — killing container")
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                stdout, stderr = process.communicate(timeout=10)
                update_job(job_id, {
                    "stdout": stdout or "",
                    "stderr": (stderr or "") + f"\nJob timed out after {JOB_TIMEOUT}s",
                    "return_code": -1,
                    "status": "failed",
                    "finished_at": datetime.utcnow().isoformat(),
                    "container_name": None,
                })
                return
            if elapsed > 0 and elapsed % 30 == 0:
                log.debug(f"[{short_id}] Still running... ({elapsed}s / {JOB_TIMEOUT}s)")

            time.sleep(1)

        stdout, stderr = process.communicate(timeout=10)
        elapsed = round(time.time() - start_time, 2)

        if cancelled:
            log.info(f"[{short_id}] Cancelled after {elapsed}s")
            update_job(job_id, {
                "stdout": stdout or "",
                "stderr": (stderr or "") + "\nJob was cancelled by user",
                "return_code": -1,
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
                "container_name": None,
            })
            return

        status = "completed" if process.returncode == 0 else "failed"

        if status == "completed":
            log.info(f"[{short_id}] ✓ Completed in {elapsed}s (GPU {gpu_id})")
        else:
            log.error(f"[{short_id}] ✗ Failed in {elapsed}s (exit code {process.returncode})")
            if stderr:
                first_lines = "\n".join(stderr.strip().splitlines()[:3])
                log.error(f"[{short_id}] stderr preview:\n{first_lines}")

        update_job(job_id, {
            "stdout": stdout or "",
            "stderr": stderr or "",
            "return_code": process.returncode,
            "status": status,
            "finished_at": datetime.utcnow().isoformat(),
            "container_name": None,
        })

    except Exception as e:
        log.exception(f"[{short_id}] Unexpected error: {e}")
        update_job(job_id, {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
            "status": "failed",
            "finished_at": datetime.utcnow().isoformat(),
            "container_name": None,
        })

    finally:
        release_gpu(gpu_id)
        if os.path.exists(worker_job_file):
            os.remove(worker_job_file)
            log.debug(f"[{short_id}] Temp file removed")


def main():
    log.info("=" * 60)
    log.info(f"GPU Job Scheduler Worker")
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

            gpu_id = None
            wait_start = time.time()
            while gpu_id is None:
                current_job = get_job(job_id)
                if not current_job or current_job.get("cancel_requested"):
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
                args=(
                    job_id,
                    job["code"],
                    gpu_id,
                    job["image"],
                    job.get("cpus", 1.0),
                    job.get("memory", "2g"),
                ),
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