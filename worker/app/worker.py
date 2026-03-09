from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import tempfile
import os

app = FastAPI()


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
    temp_file_path = None

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write(job.code)
            temp_file_path = temp_file.name

        result = subprocess.run(
            ["python", temp_file_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Job execution timed out",
            "return_code": -1,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "return_code": -1,
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)