from fastapi import FastAPI
import subprocess

app = FastAPI()


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