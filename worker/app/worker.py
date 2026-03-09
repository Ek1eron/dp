from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os
import uuid

app = FastAPI()

JOB_VOLUME_NAME = "gpu-job-scheduler-jobs"


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


@app.post("/run-job")
def run_job(job: JobRequest):
    os.makedirs("/jobs", exist_ok=True)

    job_id = str(uuid.uuid4())
    filename = f"{job_id}.py"
    worker_job_file = f"/jobs/{filename}"

    try:
        with open(worker_job_file, "w", encoding="utf-8") as f:
            f.write(job.code)

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

        return {
            "job_id": job_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "job_id": job_id,
            "stdout": "",
            "stderr": "Job execution timed out",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }
    finally:
        if os.path.exists(worker_job_file):
            os.remove(worker_job_file)