import io
import os
import time
import zipfile

import pytest
import requests

BASE_URL  = os.getenv("BASE_URL", "http://localhost:8001")
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "")
TIMEOUT   = 60


# ── Helpers ───────────────────────────────────────────────────────────────────
def submit_job(code, runtime="pytorch-cu121", cpus=1.0, memory="2g", **kwargs):
    payload = {"code": code, "runtime": runtime, "cpus": cpus, "memory": memory}
    payload.update(kwargs)
    return requests.post(f"{BASE_URL}/submit-job", json=payload)


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


def make_zip(files: dict) -> bytes:
    """Build an in-memory .zip from {filename: bytes}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def admin_headers():
    return {"X-Admin-Key": ADMIN_KEY}


admin_required = pytest.mark.skipif(
    not ADMIN_KEY, reason="ADMIN_API_KEY env var not set — admin tests skipped"
)


# ─────────────────────────────────────────────────────────────────────────────
# Health & basic
# ─────────────────────────────────────────────────────────────────────────────
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

    def test_cluster_status_has_new_fields(self):
        """cluster-status має avg_execution_time_seconds (v2.1)."""
        res = requests.get(f"{BASE_URL}/cluster-status")
        assert res.status_code == 200
        data = res.json()
        assert "gpu_summary" in data
        assert "job_summary" in data
        assert "queue_length" in data
        assert "avg_execution_time_seconds" in data


# ─────────────────────────────────────────────────────────────────────────────
# Job submission (JSON API)
# ─────────────────────────────────────────────────────────────────────────────
class TestJobSubmit:

    def test_submit_valid_job(self):
        """Задача приймається і отримує статус queued."""
        res = submit_job('print("hello")', runtime="pytorch-cu121")
        assert res.status_code == 201
        data = res.json()
        assert "job_id" in data
        assert data["status"] == "queued"
        assert data["runtime"] == "pytorch-cu121"

    def test_submit_with_name_and_description(self):
        """Поля name/description зберігаються в задачі."""
        res = submit_job(
            'print("named")',
            name="My Test Job",
            description="testing job naming",
        )
        assert res.status_code == 201
        job_id = res.json()["job_id"]
        try:
            job = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
            assert job["name"] == "My Test Job"
            assert job["description"] == "testing job naming"
        finally:
            requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")

    def test_submit_with_student_id(self):
        """student_id зберігається у задачі."""
        res = submit_job('print("ok")', student_id="ivan_test")
        assert res.status_code == 201
        job_id = res.json()["job_id"]
        try:
            job = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
            assert job["student_id"] == "ivan_test"
        finally:
            requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")

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


# ─────────────────────────────────────────────────────────────────────────────
# Code safety (AST checks)
# ─────────────────────────────────────────────────────────────────────────────
class TestCodeSafety:

    def test_eval_blocked(self):
        res = submit_job('eval("1+1")')
        assert res.status_code == 400
        assert "eval" in res.json()["detail"].lower()

    def test_exec_blocked(self):
        res = submit_job('exec("print(1)")')
        assert res.status_code == 400
        assert "exec" in res.json()["detail"].lower()

    def test_compile_blocked(self):
        res = submit_job('compile("1", "f", "exec")')
        assert res.status_code == 400

    def test_dunder_import_blocked(self):
        res = submit_job('__import__("os")')
        assert res.status_code == 400

    def test_syntax_error_rejected(self):
        res = submit_job("def foo(\n")
        assert res.status_code == 400
        assert "syntax" in res.json()["detail"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt validation
# ─────────────────────────────────────────────────────────────────────────────
class TestRequirementsSafety:

    def test_simple_requirements_accepted(self):
        res = submit_job('print("ok")', requirements="numpy\npandas")
        assert res.status_code == 201
        requests.post(f"{BASE_URL}/jobs/{res.json()['job_id']}/cancel")

    def test_editable_install_blocked(self):
        res = submit_job('print("ok")', requirements="-e ./local")
        assert res.status_code == 400

    def test_index_url_blocked(self):
        res = submit_job('print("ok")', requirements="-i https://evil.example.com\nrequests")
        assert res.status_code == 400

    def test_git_install_blocked(self):
        res = submit_job('print("ok")', requirements="git+https://github.com/foo/bar")
        assert res.status_code == 400

    def test_find_links_blocked(self):
        res = submit_job('print("ok")', requirements="--find-links http://evil.com\nrequests")
        assert res.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Multipart form submission
# ─────────────────────────────────────────────────────────────────────────────
class TestSubmitJobForm:

    def test_form_with_pasted_code(self):
        data = {"code": 'print("from form")', "runtime": "pytorch-cu121",
                "cpus": "1", "memory": "1g"}
        res  = requests.post(f"{BASE_URL}/submit-job-form", data=data)
        assert res.status_code == 201
        requests.post(f"{BASE_URL}/jobs/{res.json()['job_id']}/cancel")

    def test_form_with_py_file(self):
        files = {"code_file": ("script.py", b'print("from file")', "text/x-python")}
        data  = {"runtime": "pytorch-cu121", "cpus": "1", "memory": "1g"}
        res   = requests.post(f"{BASE_URL}/submit-job-form", data=data, files=files)
        assert res.status_code == 201
        requests.post(f"{BASE_URL}/jobs/{res.json()['job_id']}/cancel")

    def test_form_non_py_file_rejected(self):
        files = {"code_file": ("script.txt", b'print("x")', "text/plain")}
        data  = {"runtime": "pytorch-cu121"}
        res   = requests.post(f"{BASE_URL}/submit-job-form", data=data, files=files)
        assert res.status_code == 400

    def test_form_no_code_no_file_no_repo(self):
        data = {"runtime": "pytorch-cu121"}
        res  = requests.post(f"{BASE_URL}/submit-job-form", data=data)
        assert res.status_code == 400

    def test_form_invalid_repo_url(self):
        data = {"repo_url": "https://evil.example.com/user/repo", "runtime": "pytorch-cu121"}
        res  = requests.post(f"{BASE_URL}/submit-job-form", data=data)
        assert res.status_code == 400

    def test_form_github_url_accepted(self):
        # Format validation passes — actual clone happens in worker
        data = {"repo_url": "https://github.com/octocat/Hello-World", "runtime": "pytorch-cu121"}
        res  = requests.post(f"{BASE_URL}/submit-job-form", data=data)
        assert res.status_code == 201
        requests.post(f"{BASE_URL}/jobs/{res.json()['job_id']}/cancel")

    def test_form_with_zip_dataset(self):
        zip_bytes = make_zip({"data.csv": b"a,b\n1,2\n"})
        files = {"dataset": ("data.zip", zip_bytes, "application/zip")}
        data  = {"code": 'print("with dataset")', "runtime": "pytorch-cu121"}
        res   = requests.post(f"{BASE_URL}/submit-job-form", data=data, files=files)
        assert res.status_code == 201
        requests.post(f"{BASE_URL}/jobs/{res.json()['job_id']}/cancel")

    def test_form_dataset_not_zip_rejected(self):
        files = {"dataset": ("data.txt", b"just text, not a zip", "text/plain")}
        data  = {"code": 'print("x")', "runtime": "pytorch-cu121"}
        res   = requests.post(f"{BASE_URL}/submit-job-form", data=data, files=files)
        assert res.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Queue limit per student
# ─────────────────────────────────────────────────────────────────────────────
class TestQueueLimit:

    def test_queue_limit_returns_429(self):
        """Перевищення MAX_QUEUE_PER_STUDENT повертає HTTP 429."""
        sid = f"test_limit_{int(time.time())}"
        ids = []
        try:
            # Default limit is 3
            for _ in range(3):
                res = submit_job("import time\ntime.sleep(120)", student_id=sid)
                if res.status_code != 201:
                    pytest.skip(f"Setup failed: {res.status_code}")
                ids.append(res.json()["job_id"])

            res = submit_job('print("over limit")', student_id=sid)
            assert res.status_code == 429
            assert "limit" in res.json()["detail"].lower()
        finally:
            for jid in ids:
                requests.post(f"{BASE_URL}/jobs/{jid}/cancel")


# ─────────────────────────────────────────────────────────────────────────────
# Job execution
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts
# ─────────────────────────────────────────────────────────────────────────────
class TestArtifacts:

    def test_artifact_saved_and_listed(self):
        code = (
            "import os\n"
            "out = os.environ.get('OUTPUT_DIR', '/workspace/output')\n"
            "os.makedirs(out, exist_ok=True)\n"
            "with open(os.path.join(out, 'result.txt'), 'w') as f:\n"
            "    f.write('hello artifact')\n"
            "print('saved')\n"
        )
        res = submit_job(code)
        job_id = res.json()["job_id"]
        job = wait_for_job(job_id)
        assert job["status"] == "completed"
        assert len(job["artifacts"]) == 1
        assert job["artifacts"][0]["name"] == "result.txt"
        assert job["artifacts"][0]["size"] == len("hello artifact")

    def test_artifact_download(self):
        code = (
            "import os\n"
            "out = os.environ.get('OUTPUT_DIR', '/workspace/output')\n"
            "os.makedirs(out, exist_ok=True)\n"
            "with open(os.path.join(out, 'metric.txt'), 'w') as f:\n"
            "    f.write('accuracy=0.95')\n"
        )
        res = submit_job(code)
        job_id = res.json()["job_id"]
        wait_for_job(job_id)
        dl = requests.get(f"{BASE_URL}/jobs/{job_id}/artifacts/metric.txt")
        assert dl.status_code == 200
        assert dl.text == "accuracy=0.95"

    def test_artifact_not_found(self):
        res = submit_job('print("no artifacts")')
        job_id = res.json()["job_id"]
        wait_for_job(job_id)
        dl = requests.get(f"{BASE_URL}/jobs/{job_id}/artifacts/nonexistent.txt")
        assert dl.status_code == 404

    def test_artifacts_endpoint_returns_empty_list(self):
        res = submit_job('print("no artifacts created")')
        job_id = res.json()["job_id"]
        wait_for_job(job_id)
        ar = requests.get(f"{BASE_URL}/jobs/{job_id}/artifacts")
        assert ar.status_code == 200
        assert ar.json()["artifacts"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Dataset support (DATA_DIR env var)
# ─────────────────────────────────────────────────────────────────────────────
class TestDataset:

    def test_data_dir_env_var_set_when_dataset_uploaded(self):
        zip_bytes = make_zip({"hello.txt": b"hello from dataset"})
        files = {"dataset": ("data.zip", zip_bytes, "application/zip")}
        code = (
            "import os\n"
            "data_dir = os.environ.get('DATA_DIR')\n"
            "assert data_dir, 'DATA_DIR not set'\n"
            "with open(os.path.join(data_dir, 'hello.txt')) as f:\n"
            "    print(f.read())\n"
        )
        data = {"code": code, "runtime": "pytorch-cu121"}
        res = requests.post(f"{BASE_URL}/submit-job-form", data=data, files=files)
        assert res.status_code == 201
        job_id = res.json()["job_id"]
        job = wait_for_job(job_id, timeout=90)
        assert job["status"] == "completed", f"stderr: {job['stderr']}"
        assert "hello from dataset" in job["stdout"]


# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt — actual pip install
# ─────────────────────────────────────────────────────────────────────────────
class TestRequirementsInstallation:

    def test_pip_install_runs_before_code(self):
        """`pip install -r requirements.txt` виконується перед `python script.py`."""
        # 'six' — крихітний пакет, ставиться за секунди
        code = "import six\nprint(f'six version: {six.__version__}')"
        res = submit_job(code, requirements="six")
        assert res.status_code == 201
        job_id = res.json()["job_id"]
        # Більший timeout — пакет завантажується з PyPI
        job = wait_for_job(job_id, timeout=180)
        assert job["status"] == "completed", f"stderr: {job['stderr']}"
        assert "six version" in job["stdout"]


# ─────────────────────────────────────────────────────────────────────────────
# SSE log stream
# ─────────────────────────────────────────────────────────────────────────────
class TestLogStream:

    def test_logs_endpoint_for_completed_job(self):
        """SSE передає буферизовані логи + signal end для завершеної задачі."""
        res = submit_job('print("streaming hello")')
        job_id = res.json()["job_id"]
        wait_for_job(job_id)

        with requests.get(
            f"{BASE_URL}/jobs/{job_id}/logs/stream", stream=True, timeout=10
        ) as r:
            assert r.status_code == 200
            assert "text/event-stream" in r.headers.get("content-type", "")
            saw_end = False
            for raw in r.iter_lines(decode_unicode=True):
                if raw and raw.startswith("data:"):
                    payload = raw[5:].strip()
                    if '"end"' in payload:
                        saw_end = True
                        break
            assert saw_end, "Did not receive end-of-stream event"

    def test_logs_endpoint_unknown_job(self):
        res = requests.get(f"{BASE_URL}/jobs/non-existent-id/logs/stream")
        assert res.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Cancellation
# ─────────────────────────────────────────────────────────────────────────────
class TestJobCancellation:

    def test_cancel_long_running_job(self):
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


# ─────────────────────────────────────────────────────────────────────────────
# Monitoring
# ─────────────────────────────────────────────────────────────────────────────
class TestMonitoring:

    def test_job_appears_in_list(self):
        res = submit_job('print("listed job")')
        job_id = res.json()["job_id"]
        try:
            jobs_res = requests.get(f"{BASE_URL}/jobs")
            job_ids  = [j["job_id"] for j in jobs_res.json()["jobs"]]
            assert job_id in job_ids
        finally:
            requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")

    def test_jobs_filter_by_status(self):
        res = requests.get(f"{BASE_URL}/jobs?status=completed")
        assert res.status_code == 200
        for job in res.json()["jobs"]:
            assert job["status"] == "completed"

    def test_get_job_not_found(self):
        res = requests.get(f"{BASE_URL}/jobs/non-existent-id")
        assert res.status_code == 404

    def test_queue_position_field_for_queued(self):
        """Якщо задача queued — повертається queue_position."""
        blocker_id = None
        target_id  = None
        try:
            sid = f"q_block_{int(time.time())}"
            blocker = submit_job("import time\ntime.sleep(30)", student_id=sid)
            if blocker.status_code != 201:
                pytest.skip("Setup failed")
            blocker_id = blocker.json()["job_id"]

            # Wait until blocker starts running so other jobs queue up
            for _ in range(20):
                if requests.get(f"{BASE_URL}/jobs/{blocker_id}").json()["status"] == "running":
                    break
                time.sleep(0.5)
            else:
                pytest.skip("Blocker did not start running in time")

            # Use a different student to dodge per-student queue limit
            target = submit_job(
                'print("queued")', student_id=f"q_other_{int(time.time())}"
            )
            if target.status_code != 201:
                pytest.skip("Could not submit target job")
            target_id = target.json()["job_id"]

            time.sleep(0.5)
            job = requests.get(f"{BASE_URL}/jobs/{target_id}").json()
            if job["status"] == "queued":
                assert "queue_position" in job
                if job["queue_position"] is not None:
                    assert job["queue_position"] >= 1
        finally:
            for jid in (blocker_id, target_id):
                if jid:
                    requests.post(f"{BASE_URL}/jobs/{jid}/cancel")


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup endpoint
# ─────────────────────────────────────────────────────────────────────────────
class TestCleanup:

    def test_cleanup_removes_finished_jobs(self):
        """DELETE /jobs/cleanup видаляє завершені задачі."""
        res = submit_job('print("to be cleaned")')
        job_id = res.json()["job_id"]
        wait_for_job(job_id)

        del_res = requests.delete(f"{BASE_URL}/jobs/cleanup")
        assert del_res.status_code == 200
        assert "removed_jobs" in del_res.json()
        assert del_res.json()["removed_jobs"] >= 1

        check = requests.get(f"{BASE_URL}/jobs/{job_id}")
        assert check.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Admin API (skipped if ADMIN_API_KEY env var is not set)
# ─────────────────────────────────────────────────────────────────────────────
@admin_required
class TestAdminAuth:

    def test_admin_check_with_valid_key(self):
        res = requests.get(f"{BASE_URL}/admin/check", headers=admin_headers())
        assert res.status_code == 200
        assert res.json() == {"ok": True}

    def test_admin_check_with_invalid_key(self):
        res = requests.get(f"{BASE_URL}/admin/check", headers={"X-Admin-Key": "wrong-key"})
        assert res.status_code == 401

    def test_admin_check_no_key(self):
        res = requests.get(f"{BASE_URL}/admin/check")
        assert res.status_code == 401

    def test_admin_stats_structure(self):
        res = requests.get(f"{BASE_URL}/admin/stats", headers=admin_headers())
        assert res.status_code == 200
        data = res.json()
        assert "students" in data
        assert "runtime_usage" in data
        assert "daily_jobs" in data
        assert "overview" in data
        assert "total_students" in data["overview"]
        assert "total_jobs" in data["overview"]


@admin_required
class TestApiKeys:

    def test_create_list_and_revoke_key(self):
        sid = f"keytest_{int(time.time())}"
        res = requests.post(
            f"{BASE_URL}/admin/keys",
            headers=admin_headers(),
            json={"student_id": sid, "name": "Test User"},
        )
        assert res.status_code == 200
        data    = res.json()
        api_key = data["api_key"]
        assert api_key.startswith("key_")
        assert data["student_id"] == sid

        try:
            keys = requests.get(
                f"{BASE_URL}/admin/keys", headers=admin_headers()
            ).json()["keys"]
            assert any(k["api_key"] == api_key for k in keys)
        finally:
            del_res = requests.delete(
                f"{BASE_URL}/admin/keys/{api_key}", headers=admin_headers()
            )
            assert del_res.status_code == 200

        keys = requests.get(
            f"{BASE_URL}/admin/keys", headers=admin_headers()
        ).json()["keys"]
        assert not any(k["api_key"] == api_key for k in keys)

    def test_create_key_requires_student_id(self):
        res = requests.post(
            f"{BASE_URL}/admin/keys", headers=admin_headers(),
            json={"student_id": ""},
        )
        assert res.status_code == 400

    def test_keys_endpoints_require_admin_auth(self):
        res = requests.get(f"{BASE_URL}/admin/keys")
        assert res.status_code == 401
        res = requests.post(f"{BASE_URL}/admin/keys", json={"student_id": "x"})
        assert res.status_code == 401

    def test_submit_with_api_key_overrides_form_student_id(self):
        """Якщо передано X-API-Key — student_id з ключа має пріоритет над формою."""
        sid = f"trusted_{int(time.time())}"
        res = requests.post(
            f"{BASE_URL}/admin/keys", headers=admin_headers(),
            json={"student_id": sid},
        )
        api_key = res.json()["api_key"]

        try:
            res = requests.post(
                f"{BASE_URL}/submit-job",
                headers={"X-API-Key": api_key},
                json={"code": 'print("with key")', "student_id": "fake_imposter"},
            )
            assert res.status_code == 201
            job_id = res.json()["job_id"]
            try:
                job = requests.get(f"{BASE_URL}/jobs/{job_id}").json()
                # student_id з ключа, не з форми
                assert job["student_id"] == sid
            finally:
                requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")
        finally:
            requests.delete(f"{BASE_URL}/admin/keys/{api_key}", headers=admin_headers())


@admin_required
class TestAdminKill:

    def test_admin_kill_long_running_job(self):
        res = submit_job("import time\ntime.sleep(60)")
        job_id = res.json()["job_id"]
        kill_res = requests.post(
            f"{BASE_URL}/admin/jobs/{job_id}/kill", headers=admin_headers()
        )
        assert kill_res.status_code == 200
        job = wait_for_job(job_id, timeout=40)
        assert job["status"] == "cancelled"

    def test_admin_kill_unknown_job(self):
        res = requests.post(
            f"{BASE_URL}/admin/jobs/non-existent/kill", headers=admin_headers()
        )
        assert res.status_code == 404

    def test_admin_kill_without_auth_rejected(self):
        res = submit_job('print("x")')
        job_id = res.json()["job_id"]
        try:
            kill_res = requests.post(f"{BASE_URL}/admin/jobs/{job_id}/kill")
            assert kill_res.status_code == 401
        finally:
            requests.post(f"{BASE_URL}/jobs/{job_id}/cancel")


# ─────────────────────────────────────────────────────────────────────────────
# UI templates
# ─────────────────────────────────────────────────────────────────────────────
class TestUI:

    def test_dashboard_renders(self):
        res = requests.get(f"{BASE_URL}/")
        assert res.status_code == 200
        assert "GPU Job Scheduler" in res.text

    def test_admin_page_renders(self):
        res = requests.get(f"{BASE_URL}/admin")
        assert res.status_code == 200
        assert "Admin" in res.text
