from fastapi import FastAPI
import subprocess

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/gpu-info")
def gpu_info():
    try:
        result = subprocess.check_output(["nvidia-smi"], text=True)
        return {"gpu": result}
    except Exception as e:
        return {"error": str(e)}