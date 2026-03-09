import subprocess
import os
import time
import json
from datetime import datetime
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "4"))
JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def update_job(job_id: str, updates: dict):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return

    job = json.loads(raw)
    job.update(updates)
    r.set(f"job:{job_id}", json.dumps(job))


def get_free_gpu():
    for gpu_id in range(GPU_COUNT):
        key = f"gpu:{gpu_id}"
        raw = r.get(key)
        if not raw:
            continue

        gpu = json.loads(raw)
        if gpu["status"] == "idle":
            gpu["status"] = "busy"
            gpu["job_id"] = None
            gpu["updated_at"] = datetime.utcnow().isoformat()
            r.set(key, json.dumps(gpu))
            return gpu_id

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


def execute_job(job_id: str, code: str, gpu_id: int):
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"

    update_job(
        job_id,
        {
            "status": "running",
            "gpu_id": gpu_id,
            "started_at": datetime.utcnow().isoformat(),
        },
    )

    reserve_gpu_for_job(gpu_id, job_id)

    try:
        os.makedirs("/jobs", exist_ok=True)

        with open(worker_job_file, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                f"device={gpu_id}",
                "-v",
                f"{JOB_VOLUME_NAME}:/workspace",
                "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
                "python",
                f"/workspace/{filename}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        update_job(
            job_id,
            {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "status": "completed" if result.returncode == 0 else "failed",
                "finished_at": datetime.utcnow().isoformat(),
            },
        )

    except subprocess.TimeoutExpired:
        update_job(
            job_id,
            {
                "stdout": "",
                "stderr": "Job execution timed out",
                "return_code": -1,
                "status": "failed",
                "finished_at": datetime.utcnow().isoformat(),
            },
        )

    except Exception as e:
        update_job(
            job_id,
            {
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
                "status": "failed",
                "finished_at": datetime.utcnow().isoformat(),
            },
        )

    finally:
        release_gpu(gpu_id)
        if os.path.exists(worker_job_file):
            os.remove(worker_job_file)


def main():
    print("Worker started. Waiting for jobs...")

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

            gpu_id = None
            while gpu_id is None:
                gpu_id = get_free_gpu()
                if gpu_id is None:
                    time.sleep(1)

            execute_job(job_id, job["code"], gpu_id)

        except Exception as e:
            print(f"Worker loop error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()