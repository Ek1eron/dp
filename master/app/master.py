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
GPU_COUNT = int(os.getenv("GPU_COUNT", "4"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class JobRequest(BaseModel):
    code: str


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


@app.post("/submit-job")
def submit_job(job: JobRequest):
    job_id = str(uuid.uuid4())

    job_data = {
        "job_id": job_id,
        "code": job.code,
        "status": "queued",
        "gpu_id": None,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "return_code": None,
    }

    r.set(f"job:{job_id}", json.dumps(job_data))
    r.rpush("job_queue", job_id)

    return {
        "job_id": job_id,
        "status": "queued",
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