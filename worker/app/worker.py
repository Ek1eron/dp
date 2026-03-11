import subprocess
import os
import time
import json
from datetime import datetime
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
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
            print(f"[Recovery] GPU {gpu_id} was stuck as busy (job={job_id}). Releasing.")
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

        timeout_seconds = 120
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

            if time.time() - start_time > timeout_seconds:
                subprocess.run(["docker", "kill", container_name], capture_output=True)
                stdout, stderr = process.communicate(timeout=10)
                update_job(job_id, {
                    "stdout": stdout or "",
                    "stderr": (stderr or "") + "\nJob execution timed out",
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
    print(f"Worker started. GPU_COUNT={GPU_COUNT}. Waiting for jobs...")

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

            print(f"[Worker] Running job {job_id} on GPU {gpu_id}")
            execute_job(
                job_id=job_id,
                code=job["code"],
                gpu_id=gpu_id,
                image=job["image"],
                cpus=job.get("cpus", 1.0),
                memory=job.get("memory", "2g"),
            )

        except Exception as e:
            print(f"[Worker] Loop error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    main()