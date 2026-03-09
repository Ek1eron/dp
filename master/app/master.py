from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import redis
import uuid
import json
import os

app = FastAPI()

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RUNTIME_PROFILES = {
    "pytorch-cu121": "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
}


class JobRequest(BaseModel):
    code: str
    runtime: str = "pytorch-cu121"
    cpus: float = 1.0
    memory: str = "2g"


def initialize_gpus():
    for gpu_id in range(GPU_COUNT):
        key = f"gpu:{gpu_id}"
        if not r.get(key):
            gpu_data = {
                "gpu_id": gpu_id,
                "status": "idle",
                "job_id": None,
                "updated_at": datetime.utcnow().isoformat(),
            }
            r.set(key, json.dumps(gpu_data))


@app.on_event("startup")
def startup_event():
    initialize_gpus()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runtimes")
def list_runtimes():
    return {"runtimes": list(RUNTIME_PROFILES.keys())}


@app.post("/submit-job")
def submit_job(job: JobRequest):
    if job.runtime not in RUNTIME_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported runtime '{job.runtime}'. Allowed: {list(RUNTIME_PROFILES.keys())}",
        )

    job_id = str(uuid.uuid4())

    job_data = {
        "job_id": job_id,
        "code": job.code,
        "runtime": job.runtime,
        "image": RUNTIME_PROFILES[job.runtime],
        "cpus": job.cpus,
        "memory": job.memory,
        "status": "queued",
        "gpu_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "return_code": None,
    }

    r.setex(
    f"job:{job_id}",
    60 * 60 * 24,  
    json.dumps(job_data)
)
    r.rpush("job_queue", job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "runtime": job.runtime,
    }


@app.get("/jobs")
def list_jobs():
    keys = r.keys("job:*")
    jobs = []

    for key in keys:
        raw = r.get(key)
        if raw:
            jobs.append(json.loads(raw))

    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": jobs}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(raw)


@app.get("/gpus")
def list_gpus():
    gpus = []
    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if raw:
            gpus.append(json.loads(raw))
    return {"gpus": gpus}


@app.get("/cluster-status")
def cluster_status():
    gpu_total = 0
    gpu_idle = 0
    gpu_busy = 0

    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if not raw:
            continue

        gpu_total += 1
        gpu = json.loads(raw)

        if gpu["status"] == "idle":
            gpu_idle += 1
        elif gpu["status"] == "busy":
            gpu_busy += 1

    job_keys = r.keys("job:*")

    queued = 0
    running = 0
    completed = 0
    failed = 0

    for key in job_keys:
        raw = r.get(key)
        if not raw:
            continue

        job = json.loads(raw)
        status = job.get("status")

        if status == "queued":
            queued += 1
        elif status == "running":
            running += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1

    queue_length = r.llen("job_queue")

    return {
        "gpu_summary": {
            "total": gpu_total,
            "idle": gpu_idle,
            "busy": gpu_busy,
        },
        "job_summary": {
            "queued": queued,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": queued + running + completed + failed,
        },
        "queue_length": queue_length,
    }

@app.delete("/jobs/cleanup")
def cleanup_jobs():
    keys = r.keys("job:*")
    removed = 0

    for key in keys:
        raw = r.get(key)
        if not raw:
            continue

        job = json.loads(raw)

        if job["status"] in ["completed", "failed"]:
            r.delete(key)
            removed += 1

    return {"removed_jobs": removed}