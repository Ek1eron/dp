from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from datetime import datetime
import redis
import uuid
import json
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

templates = Jinja2Templates(directory="app/templates")
app = FastAPI(title="GPU Job Scheduler", version="1.0.0")

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RUNTIME_PROFILES = {
    # PyTorch + CUDA 12.1
    "pytorch-cu121": "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
    # PyTorch + CUDA 11.8 — для старіших моделей/бібліотек
    "pytorch-cu118": "pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime",
    # Чистий CUDA без фреймворків — для власного C++/CUDA коду
    "cuda-base": "nvidia/cuda:12.3.2-base-ubuntu22.04",
    # TensorFlow з GPU підтримкою
    "tensorflow": "tensorflow/tensorflow:2.15.0-gpu",
}

VALID_MEMORY_SUFFIXES = ("m", "g", "mb", "gb")


class JobRequest(BaseModel):
    code: str
    runtime: str = "pytorch-cu121"
    cpus: float = 1.0
    memory: str = "2g"

    @field_validator("code")
    @classmethod
    def code_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("code must not be empty")
        return v

    @field_validator("cpus")
    @classmethod
    def cpus_range(cls, v):
        if not (0.1 <= v <= 8.0):
            raise ValueError("cpus must be between 0.1 and 8.0")
        return v

    @field_validator("memory")
    @classmethod
    def memory_format(cls, v):
        if not any(v.lower().endswith(s) for s in VALID_MEMORY_SUFFIXES):
            raise ValueError("memory must end with m, g, mb or gb (e.g. '2g', '512m')")
        return v.lower()


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
            print(f"[Master] Initialized GPU {gpu_id}")



@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.on_event("startup")
def startup_event():
    initialize_gpus()


@app.get("/health")
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "ok" if redis_ok else "degraded",
        "redis": "ok" if redis_ok else "unavailable",
        "gpu_count": GPU_COUNT,
    }


@app.get("/runtimes")
def list_runtimes():
    """Повертає список доступних runtime-профілів з їх Docker-образами."""
    return {
        "runtimes": {name: image for name, image in RUNTIME_PROFILES.items()}
    }


@app.post("/submit-job", status_code=201)
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
        "container_name": None,
        "cancel_requested": False,
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "return_code": None,
    }

    r.setex(f"job:{job_id}", 60 * 60 * 24, json.dumps(job_data))
    r.rpush("job_queue", job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "runtime": job.runtime,
        "image": RUNTIME_PROFILES[job.runtime],
    }


@app.get("/jobs")
def list_jobs(status: str = None):

    keys = r.keys("job:*")
    jobs = []
    for key in keys:
        raw = r.get(key)
        if raw:
            job = json.loads(raw)
            if status is None or job.get("status") == status:
                jobs.append(job)

    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(raw)


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")

    job = json.loads(raw)

    if job["status"] in ["completed", "failed", "cancelled"]:
        return {
            "job_id": job_id,
            "status": job["status"],
            "message": "Job is already finished",
        }

    job["cancel_requested"] = True
    if job["status"] == "queued":
        job["status"] = "cancelled"
        job["finished_at"] = datetime.utcnow().isoformat()

    r.setex(f"job:{job_id}", 60 * 60 * 24, json.dumps(job))

    return {
        "job_id": job_id,
        "status": job["status"],
        "cancel_requested": True,
    }


@app.get("/gpus")
def list_gpus():
    gpus = []
    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if raw:
            gpus.append(json.loads(raw))
    return {"gpus": gpus, "total": GPU_COUNT}


@app.get("/cluster-status")
def cluster_status():
    gpu_total = gpu_idle = gpu_busy = 0
    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if not raw:
            continue
        gpu = json.loads(raw)
        gpu_total += 1
        if gpu["status"] == "idle":
            gpu_idle += 1
        elif gpu["status"] == "busy":
            gpu_busy += 1

    queued = running = completed = failed = cancelled = 0
    for key in r.keys("job:*"):
        raw = r.get(key)
        if not raw:
            continue
        status = json.loads(raw).get("status")
        if status == "queued":
            queued += 1
        elif status == "running":
            running += 1
        elif status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        elif status == "cancelled":
            cancelled += 1

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
            "cancelled": cancelled,
            "total": queued + running + completed + failed + cancelled,
        },
        "queue_length": r.llen("job_queue"),
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
        if job["status"] in ["completed", "failed", "cancelled"]:
            r.delete(key)
            removed += 1
    return {"removed_jobs": removed}