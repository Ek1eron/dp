import subprocess
import os
import uuid
import time
import json
from datetime import datetime
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def update_job(job_id: str, updates: dict):
    raw = r.get(f"job:{job_id}")
    if not raw:
        return

    job = json.loads(raw)
    job.update(updates)
    r.set(f"job:{job_id}", json.dumps(job))


def execute_job(job_id: str, code: str):
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"

    update_job(
        job_id,
        {
            "status": "running",
            "started_at": datetime.utcnow().isoformat(),
        },
    )
    set_gpu_status("busy", job_id)

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
                "all",
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
        set_gpu_status("idle", None)
        if os.path.exists(worker_job_file):
            os.remove(worker_job_file)

def main():
    initialize_gpu_status()
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
            execute_job(job_id, job["code"])

        except Exception as e:
            print(f"Worker loop error: {e}")
            time.sleep(2)

def set_gpu_status(status: str, job_id: str | None = None):
    gpu_info = {
        "status": status,
        "job_id": job_id,
        "updated_at": datetime.utcnow().isoformat(),
    }
    r.set("gpu:0", json.dumps(gpu_info))


def initialize_gpu_status():
    if not r.get("gpu:0"):
        set_gpu_status("idle", None)

if __name__ == "__main__":
    main()