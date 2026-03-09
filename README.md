# GPU Job Scheduler

A simple GPU job scheduling platform that allows users to submit computational tasks which are executed in isolated Docker containers using NVIDIA GPUs.

This project was  focused on building a simplified **GPU compute cluster scheduler** similar in concept to systems like **Slurm** or **Kubernetes GPU scheduling**, but designed for a single server with multiple GPUs.

---

# Architecture

The system follows a distributed architecture with the following components:

```
Client
   │
   ▼
Master API Server (FastAPI)
   │
   ▼
Redis Job Queue
   │
   ▼
Worker
   │
   ▼
Docker Container
   │
   ▼
CUDA / PyTorch
   │
   ▼
GPU
```

---

# Features

* GPU job scheduling
* Docker-based isolated job execution
* Redis queue for task distribution
* Runtime profiles (CUDA / PyTorch environments)
* Resource limits (CPU / Memory)
* GPU allocation
* Job lifecycle management
* Job cancellation
* Cluster monitoring API
* Automatic job cleanup

---

# Tech Stack

| Component         | Technology               |
| ----------------- | ------------------------ |
| API               | FastAPI                  |
| Queue             | Redis                    |
| Container Runtime | Docker                   |
| GPU Support       | NVIDIA Container Toolkit |
| Compute Framework | PyTorch / CUDA           |
| Environment       | WSL2 / Linux             |
| Orchestration     | docker-compose           |

---

# Project Structure

```
gpu-job-scheduler
│
├── master
│   ├── Dockerfile
│   └── app
│       └── master.py
│
├── worker
│   ├── Dockerfile
│   └── app
│       └── worker.py
│
├── docker-compose.yml
├── README.md
```

---

# Running the System

### Requirements

* Docker
* NVIDIA Container Toolkit
* NVIDIA GPU drivers
* docker-compose

---

### Start the system

```
docker compose up --build
```

Services started:

```
Redis      -> port 6379
Master API -> port 8001
Worker     -> background GPU executor
```

---

# API Endpoints

## Submit Job

```
POST /submit-job
```

Example:

```
{
  "code": "import torch\nprint(torch.cuda.is_available())",
  "runtime": "pytorch-cu121",
  "cpus": 1,
  "memory": "2g"
}
```

Response:

```
{
  "job_id": "...",
  "status": "queued"
}
```

---

## Job Status

```
GET /jobs/{job_id}
```

Returns:

* status
* stdout
* stderr
* runtime
* gpu_id
* timestamps

---

## List Jobs

```
GET /jobs
```

---

## Cancel Job

```
POST /jobs/{job_id}/cancel
```

Stops running job container.

---

## GPU Status

```
GET /gpus
```

Example:

```
{
  "gpus": [
    {
      "gpu_id": 0,
      "status": "idle"
    }
  ]
}
```

---

## Cluster Status

```
GET /cluster-status
```

Returns summary of:

* GPUs
* running jobs
* queued jobs
* completed jobs

---

# Runtime Profiles

To avoid CUDA and framework conflicts, the system uses predefined runtime environments.

Example:

```
pytorch-cu121 -> pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
```

Each job runs in its own container using the selected runtime.

---

# Job Lifecycle

```
queued
  ↓
running
  ↓
completed / failed / cancelled
```

Jobs are automatically removed after a retention period.

---

# Example GPU Job

```
{
  "code": "import torch\nprint(torch.cuda.is_available())\nprint(torch.cuda.get_device_name(0))",
  "runtime": "pytorch-cu121"
}
```

---

# Future Improvements

* Multi-GPU scheduling
* Multi-node cluster support
* Web dashboard
* Job file uploads
* Authentication
* Persistent job history

---
