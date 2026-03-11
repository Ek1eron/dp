import subprocess
import os
import time
import json
import threading
from datetime import datetime
import redis
import sys
sys.stdout.reconfigure(line_buffering=True)
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
JOB_TIMEOUT = int(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
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
            print(f"[GPU] Error checking gpu:{gpu_id}: {e}")
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


def check_real_gpu_count():
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
        print(f"[GPU] Detected GPUs: {real_count}")
        for i, name in enumerate(lines):
            print(f"[GPU]   [{i}] {name}")

        if real_count == 0:
            print("[GPU] WARNING: No GPUs detected! Check NVIDIA drivers and Container Toolkit.")
        elif real_count < GPU_COUNT:
            print(f"[GPU] WARNING: GPU_COUNT={GPU_COUNT} but only {real_count} GPU(s) found. "
                  f"Update GPU_COUNT in .env to avoid errors.")
        elif real_count > GPU_COUNT:
            print(f"[GPU] INFO: {real_count} GPUs available but GPU_COUNT={GPU_COUNT}. "
                  f"Only {GPU_COUNT} GPU(s) will be used by the scheduler.")
        else:
            print(f"[GPU] OK: GPU_COUNT={GPU_COUNT} matches real GPU count.")

        return real_count
    except Exception as e:
        print(f"[GPU] Could not detect GPU count: {e}")
        return None


def recover_stale_gpus():
    print("[Recovery] Checking for stale GPU reservations...")
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
            print(f"[Recovery] GPU {gpu_id} stuck as busy (job={job_id}). Releasing.")
            release_gpu(gpu_id)
            if job_id:
                job = get_job(job_id)
                if job and job.get("status") == "running":
                    update_job(job_id, {
                        "status": "failed",
                        "stderr": "Worker restarted — job was lost",
                        "finished_at": datetime.utcnow().isoformat(),
                        "container_name": None,
                    })


def execute_job(job_id: str, code: str, gpu_id: int, image: str, cpus: float, memory: str):
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"
    container_name = f"job-{job_id}"

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
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                cancelled = True
                break

            if time.time() - start_time > JOB_TIMEOUT:
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

            time.sleep(1)

        stdout, stderr = process.communicate(timeout=10)

        if cancelled:
            update_job(job_id, {
                "stdout": stdout or "",
                "stderr": (stderr or "") + "\nJob was cancelled by user",
                "return_code": -1,
                "status": "cancelled",
                "finished_at": datetime.utcnow().isoformat(),
                "container_name": None,
            })
            return

        update_job(job_id, {
            "stdout": stdout or "",
            "stderr": stderr or "",
            "return_code": process.returncode,
            "status": "completed" if process.returncode == 0 else "failed",
            "finished_at": datetime.utcnow().isoformat(),
            "container_name": None,
        })

    except Exception as e:
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


def main():
    print(f"Worker started. GPU_COUNT={GPU_COUNT}, JOB_TIMEOUT={JOB_TIMEOUT}s")

    check_real_gpu_count()
    recover_stale_gpus()

    while True:
        try:
            item = r.blpop("job_queue", timeout=5)
            if not item:
                continue

            _, job_id = item
            raw = r.get(f"job:{job_id}")
            if not raw:
                continue

            job = json.loads(raw)

            if job.get("status") == "cancelled" or job.get("cancel_requested"):
                continue
        
            gpu_id = None
            while gpu_id is None:
                current_job = get_job(job_id)
                if not current_job or current_job.get("cancel_requested"):
                    break
                gpu_id = get_free_gpu()
                if gpu_id is None:
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
            print(f"[Worker] Started job {job_id[:8]} on GPU {gpu_id} "
                  f"(active threads: {threading.active_count()})")

        except Exception as e:
            print(f"[Worker] Loop error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()