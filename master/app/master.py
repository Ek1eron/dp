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

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


class JobRequest(BaseModel):
    code: str


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

@app.get("/gpu-status")
def gpu_status():
    raw = r.get("gpu:0")
    if not raw:
        return {
            "gpu_id": 0,
            "status": "unknown",
            "job_id": None,
        }

    data = json.loads(raw)
    return {
        "gpu_id": 0,
        **data,
    }