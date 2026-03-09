from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
import uuid
import threading
from datetime import datetime

app = FastAPI()

JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"
jobs_store = {}


class JobRequest(BaseModel):
    code: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/gpu-info")
def gpu_info():
    info = {
        "nvidia_smi": None,
        "torch_cuda_available": False,
        "torch_cuda_device_count": 0,
        "torch_cuda_current_device": None,
        "torch_cuda_device_name": None,
        "torch_cuda_version": None,
        "errors": [],
    }

    try:
        result = subprocess.check_output(["nvidia-smi"], text=True)
        info["nvidia_smi"] = result
    except Exception as e:
        info["errors"].append(f"nvidia-smi error: {e}")

    try:
        import torch

        info["torch_cuda_available"] = torch.cuda.is_available()
        info["torch_cuda_device_count"] = torch.cuda.device_count()
        info["torch_cuda_version"] = torch.version.cuda

        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            info["torch_cuda_current_device"] = current_device
            info["torch_cuda_device_name"] = torch.cuda.get_device_name(current_device)

    except Exception as e:
        info["errors"].append(f"torch error: {e}")

    return info


def execute_job(job_id: str, code: str):
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"

    jobs_store[job_id]["status"] = "running"
    jobs_store[job_id]["started_at"] = datetime.utcnow().isoformat()

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

        jobs_store[job_id]["stdout"] = result.stdout
        jobs_store[job_id]["stderr"] = result.stderr
        jobs_store[job_id]["return_code"] = result.returncode
        jobs_store[job_id]["status"] = "completed" if result.returncode == 0 else "failed"

    except subprocess.TimeoutExpired:
        jobs_store[job_id]["stdout"] = ""
        jobs_store[job_id]["stderr"] = "Job execution timed out"
        jobs_store[job_id]["return_code"] = -1
        jobs_store[job_id]["status"] = "failed"

    except Exception as e:
        jobs_store[job_id]["stdout"] = ""
        jobs_store[job_id]["stderr"] = str(e)
        jobs_store[job_id]["return_code"] = -1
        jobs_store[job_id]["status"] = "failed"

    finally:
        jobs_store[job_id]["finished_at"] = datetime.utcnow().isoformat()
        if os.path.exists(worker_job_file):
            os.remove(worker_job_file)


@app.post("/run-job")
def run_job(job: JobRequest):
    job_id = str(uuid.uuid4())

    jobs_store[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "started_at": None,
        "finished_at": None,
        "stdout": "",
        "stderr": "",
        "return_code": None,
    }

    thread = threading.Thread(target=execute_job, args=(job_id, job.code), daemon=True)
    thread.start()

    return {
        "job_id": job_id,
        "status": "queued",
    }


@app.get("/jobs")
def list_jobs():
    return {"jobs": list(jobs_store.values())}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job