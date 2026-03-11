

import time
import pytest
import requests

BASE_URL = "http://localhost:8001"
TIMEOUT = 60  


def submit_job(code: str, runtime: str = "pytorch-cu121", cpus: float = 1.0, memory: str = "2g"):
    res = requests.post(f"{BASE_URL}/submit-job", json={
        "code": code,
        "runtime": runtime,
        "cpus": cpus,
        "memory": memory,
    })
    return res


def wait_for_job(job_id: str, timeout: int = TIMEOUT) -> dict:
    """Чекає поки задача завершиться (completed/failed/cancelled)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{BASE_URL}/jobs/{job_id}")
        job = res.json()
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        time.sleep(2)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")



class TestHealth:

    def test_health_returns_ok(self):
        """API сервер відповідає і Redis доступний."""
        res = requests.get(f"{BASE_URL}/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["redis"] == "ok"
        assert data["gpu_count"] >= 1

    def test_runtimes_list(self):
        """API повертає список доступних runtime профілів."""
        res = requests.get(f"{BASE_URL}/runtimes")
        assert res.status_code == 200
        runtimes = res.json()["runtimes"]
        assert "pytorch-cu121" in runtimes
        assert "pytorch-cu118" in runtimes
        assert "tensorflow" in runtimes

    def test_gpus_list(self):
        """API повертає список GPU."""
        res = requests.get(f"{BASE_URL}/gpus")
        assert res.status_code == 200
        data = res.json()
        assert "gpus" in data
        assert data["total"] >= 1

    def test_cluster_status(self):
        """API повертає статус кластера з правильними полями."""
        res = requests.get(f"{BASE_URL}/cluster-status")
        assert res.status_code == 200
        data = res.json()
        assert "gpu_summary" in data
        assert "job_summary" in data
        assert "queue_length" in data

class TestJobSubmit:

    def test_submit_valid_job(self):
        """Задача приймається і отримує статус queued."""
        res = submit_job('print("hello")', runtime="pytorch-cu121")
        assert res.status_code == 201
        data = res.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["runtime"] == "pytorch-cu121"

    def test_submit_invalid_runtime(self):
        """Невідомий runtime повертає 400."""
        res = submit_job('print("hello")', runtime="unknown-runtime")
        assert res.status_code == 400

    def test_submit_empty_code(self):
        """Порожній код повертає 422."""
        res = submit_job("   ")
        assert res.status_code == 422

    def test_submit_invalid_cpus(self):
        """Некоректне значення cpus повертає 422."""
        res = submit_job('print("hello")', cpus=100.0)
        assert res.status_code == 422

    def test_submit_invalid_memory(self):
        """Некоректний формат memory повертає 422."""
        res = submit_job('print("hello")', memory="abc")
        assert res.status_code == 422


class TestJobExecution:

    def test_job_completes_successfully(self):
        """Проста задача виконується і повертає stdout."""
        res = submit_job('print("test output 123")')
        job_id = res.json()["job_id"]

        job = wait_for_job(job_id)

        assert job["status"] == "completed"
        assert job["return_code"] == 0
        assert "test output 123" in job["stdout"]
        assert job["gpu_id"] is not None
        assert job["started_at"] is not None
        assert job["finished_at"] is not None

    def test_job_with_error_fails(self):
        """Задача з помилкою отримує статус failed."""
        res = submit_job('raise ValueError("intentional error")')
        job_id = res.json()["job_id"]

        job = wait_for_job(job_id)

        assert job["status"] == "failed"
        assert job["return_code"] != 0

    def test_gpu_is_available_in_container(self):
        """CUDA доступна в контейнері задачі."""
        code = (
            "import torch\n"
            "assert torch.cuda.is_available(), 'CUDA not available'\n"
            "print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
        )
        res = submit_job(code, runtime="pytorch-cu121")
        job_id = res.json()["job_id"]

        job = wait_for_job(job_id, timeout=120)

        assert job["status"] == "completed", f"stderr: {job['stderr']}"
        assert "GPU:" in job["stdout"]

    def test_job_status_transitions(self):
        """Задача проходить через статуси: queued → running → completed."""
        res = submit_job('import time\ntime.sleep(3)\nprint("done")')
        job_id = res.json()["job_id"]

        job = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
        assert job["status"] in ("queued", "running")
        job = wait_for_job(job_id)
        assert job["status"] == "completed"



class TestJobCancellation:

    def test_cancel_queued_job(self):
        """Задача скасовується — або з черги або вже під час виконання."""
        res = submit_job('import time\ntime.sleep(30)\nprint("should not finish")')
        job_id = res.json()["job_id"]

        cancel_res = requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200
        job = wait_for_job(job_id, timeout=40)
        assert job["status"] == "cancelled"

    def test_cancel_finished_job(self):
        """Скасування вже завершеної задачі повертає повідомлення."""
        res = submit_job('print("done")')
        job_id = res.json()["job_id"]

        wait_for_job(job_id)

        cancel_res = requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")
        assert cancel_res.status_code == 200
        assert "already finished" in cancel_res.json()["message"]



class TestMonitoring:

    def test_job_appears_in_list(self):
        """Відправлена задача з'являється у списку /jobs."""
        res = submit_job('print("listed job")')
        job_id = res.json()["job_id"]

        jobs_res = requests.get(f"{BASE_URL}/jobs")
        job_ids = [j["job_id"] for j in jobs_res.json()["jobs"]]
        assert job_id in job_ids

    def test_jobs_filter_by_status(self):
        """Фільтрація /jobs?status= повертає тільки задачі з відповідним статусом."""
        res = requests.get(f"{BASE_URL}/jobs?status=completed")
        assert res.status_code == 200
        jobs = res.json()["jobs"]
        for job in jobs:
            assert job["status"] == "completed"

    def test_get_job_not_found(self):
        """Запит неіснуючої задачі повертає 404."""
        res = requests.get(f"{BASE_URL}/jobs/non-existent-id")
        assert res.status_code == 404