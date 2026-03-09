FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn

COPY app ./app

CMD ["uvicorn", "app.worker:app", "--host", "0.0.0.0", "--port", "8000"]