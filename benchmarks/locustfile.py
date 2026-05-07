"""
locustfile.py — Load testing for GPU Job Scheduler API with Locust.

Симулює N студентів що одночасно взаємодіють з системою:
  - Submit jobs (JSON API)
  - Poll job status
  - List jobs / cluster status
  - Stream logs (SSE)
  - Submit with file upload (multipart form)

Сценарії (User classes):
  StudentUser     — типовий студент: submit → poll → get result
  ReadOnlyUser    — моніторинг: /jobs, /cluster-status, /gpus
  BurstUser       — спалах подачі задач (стрес-тест)
  AdminUser       — адмін-операції (потребує ADMIN_API_KEY)

Встановлення:
    pip install locust

Запуск (веб-інтерфейс):
    locust -f benchmarks/locustfile.py --host http://localhost:8001

Запуск (headless, для CI):
    locust -f benchmarks/locustfile.py --host http://localhost:8001 \\
           --headless -u 10 -r 2 --run-time 60s \\
           --csv benchmarks/results/locust

Веб-інтерфейс: http://localhost:8089
"""

import json
import os
import random
import time

from locust import HttpUser, TaskSet, between, events, tag, task
from locust.exception import RescheduleTask

ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")

# ── Payload helpers ───────────────────────────────────────────────────────────

CODES = [
    # Швидка задача ~2s
    'import torch\nA=torch.randn(1000,1000,device="cuda")\nprint((A@A.T).sum().item())',
    # Задача з виводом
    'for i in range(5):\n    print(f"step {i}")\nimport time\ntime.sleep(1)',
    # Невалідний код (має повернути 400 — не 5xx)
    'eval("1+1")',
    # Порожній код (422)
    '   ',
]

VALID_CODES = CODES[:2]

RUNTIMES = ["pytorch-cu121", "pytorch-cu118"]


def _rand_student() -> str:
    return f"student_{random.randint(1, 20):02d}"


def _rand_valid_code() -> str:
    return random.choice(VALID_CODES)


# ── TaskSets ──────────────────────────────────────────────────────────────────

class SubmitAndPollTaskSet(TaskSet):
    """Студент подає задачу і опитує статус до завершення."""

    submitted_jobs: list[str]

    def on_start(self):
        self.submitted_jobs = []

    @tag("submit")
    @task(5)
    def submit_valid_job(self):
        payload = {
            "code":       _rand_valid_code(),
            "runtime":    random.choice(RUNTIMES),
            "cpus":       1.0,
            "memory":     "1g",
            "student_id": _rand_student(),
            "name":       f"load-test-{random.randint(1000, 9999)}",
        }
        with self.client.post(
            "/submit-job", json=payload,
            catch_response=True, name="/submit-job [valid]"
        ) as resp:
            if resp.status_code == 201:
                job_id = resp.json().get("job_id")
                if job_id:
                    self.submitted_jobs.append(job_id)
                resp.success()
            elif resp.status_code == 429:
                resp.success()  # Queue limit — очікувана відповідь
            else:
                resp.failure(f"Unexpected {resp.status_code}: {resp.text[:100]}")

    @tag("submit")
    @task(2)
    def submit_rejected_code(self):
        """AST блокування — очікуємо 400, не 500."""
        with self.client.post(
            "/submit-job",
            json={"code": 'eval("1+1")', "runtime": "pytorch-cu121"},
            catch_response=True, name="/submit-job [rejected]"
        ) as resp:
            if resp.status_code == 400:
                resp.success()
            else:
                resp.failure(f"Expected 400, got {resp.status_code}")

    @tag("poll")
    @task(8)
    def poll_job_status(self):
        if not self.submitted_jobs:
            raise RescheduleTask()
        job_id = random.choice(self.submitted_jobs)
        with self.client.get(
            f"/jobs/{job_id}",
            catch_response=True, name="/jobs/{id}"
        ) as resp:
            if resp.status_code == 200:
                status = resp.json().get("status")
                if status in ("completed", "failed", "cancelled"):
                    self.submitted_jobs.remove(job_id)
                resp.success()
            elif resp.status_code == 404:
                self.submitted_jobs.remove(job_id)
                resp.success()
            else:
                resp.failure(f"Unexpected {resp.status_code}")

    @tag("list")
    @task(3)
    def list_my_jobs(self):
        sid = _rand_student()
        self.client.get(f"/jobs?student_id={sid}", name="/jobs [list]")

    @tag("cancel")
    @task(1)
    def cancel_random_job(self):
        if not self.submitted_jobs:
            raise RescheduleTask()
        job_id = self.submitted_jobs.pop()
        with self.client.post(
            f"/jobs/{job_id}/cancel",
            catch_response=True, name="/jobs/{id}/cancel"
        ) as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Unexpected {resp.status_code}")


class MonitoringTaskSet(TaskSet):
    """Read-only користувач — тільки моніторинг."""

    @tag("health")
    @task(5)
    def health_check(self):
        self.client.get("/health", name="/health")

    @tag("cluster")
    @task(4)
    def cluster_status(self):
        self.client.get("/cluster-status", name="/cluster-status")

    @tag("gpus")
    @task(3)
    def list_gpus(self):
        self.client.get("/gpus", name="/gpus")

    @tag("jobs")
    @task(4)
    def list_all_jobs(self):
        self.client.get("/jobs", name="/jobs [all]")

    @tag("jobs")
    @task(2)
    def list_running_jobs(self):
        self.client.get("/jobs?status=running", name="/jobs [running]")

    @tag("runtimes")
    @task(1)
    def list_runtimes(self):
        self.client.get("/runtimes", name="/runtimes")


class FormSubmitTaskSet(TaskSet):
    """Студент що завантажує .py файл через форму."""

    @tag("form")
    @task(3)
    def submit_via_form(self):
        code = _rand_valid_code().encode()
        with self.client.post(
            "/submit-job-form",
            data={"runtime": "pytorch-cu121", "cpus": "1", "memory": "1g",
                  "student_id": _rand_student()},
            files={"code_file": ("script.py", code, "text/x-python")},
            catch_response=True, name="/submit-job-form [file]"
        ) as resp:
            if resp.status_code in (201, 429):
                resp.success()
                if resp.status_code == 201:
                    jid = resp.json().get("job_id")
                    if jid:
                        self.client.post(f"/jobs/{jid}/cancel",
                                         name="/jobs/{id}/cancel [cleanup]")
            else:
                resp.failure(f"Unexpected {resp.status_code}")

    @tag("form")
    @task(1)
    def submit_invalid_file_type(self):
        with self.client.post(
            "/submit-job-form",
            data={"runtime": "pytorch-cu121"},
            files={"code_file": ("script.txt", b"print('x')", "text/plain")},
            catch_response=True, name="/submit-job-form [invalid ext]"
        ) as resp:
            if resp.status_code == 400:
                resp.success()
            else:
                resp.failure(f"Expected 400, got {resp.status_code}")


# ── User classes ──────────────────────────────────────────────────────────────

class StudentUser(HttpUser):
    """Типовий студент — submit + poll."""
    tasks         = [SubmitAndPollTaskSet]
    wait_time     = between(2, 8)
    weight        = 60


class ReadOnlyUser(HttpUser):
    """Моніторинг без подачі задач."""
    tasks     = [MonitoringTaskSet]
    wait_time = between(1, 4)
    weight    = 25


class FormUser(HttpUser):
    """Студент що завантажує файли."""
    tasks     = [FormSubmitTaskSet]
    wait_time = between(3, 10)
    weight    = 15


class AdminUser(HttpUser):
    """Адмін-операції (пропускається якщо ADMIN_API_KEY не встановлено)."""
    tasks     = []
    wait_time = between(5, 15)
    weight    = 5

    def on_start(self):
        if not ADMIN_KEY:
            self.environment.runner.quit()  # skip if no key

    @tag("admin")
    @task(3)
    def admin_stats(self):
        self.client.get(
            "/admin/stats",
            headers={"X-Admin-Key": ADMIN_KEY},
            name="/admin/stats",
        )

    @tag("admin")
    @task(2)
    def admin_list_keys(self):
        self.client.get(
            "/admin/keys",
            headers={"X-Admin-Key": ADMIN_KEY},
            name="/admin/keys",
        )

    @tag("admin")
    @task(1)
    def admin_check(self):
        self.client.get(
            "/admin/check",
            headers={"X-Admin-Key": ADMIN_KEY},
            name="/admin/check",
        )


# ── Event hooks (print summary) ───────────────────────────────────────────────

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    print("\n" + "=" * 60)
    print("  Locust Load Test — підсумок")
    print("=" * 60)
    total = stats.total
    print(f"  Запитів:     {total.num_requests}")
    print(f"  Помилок:     {total.num_failures}  "
          f"({100*total.fail_ratio:.1f}%)")
    print(f"  RPS:         {total.current_rps:.1f}")
    print(f"  Latency avg: {total.avg_response_time:.0f} ms")
    print(f"  Latency p95: {total.get_response_time_percentile(0.95):.0f} ms")
    print(f"  Latency p99: {total.get_response_time_percentile(0.99):.0f} ms")
    print("=" * 60)
