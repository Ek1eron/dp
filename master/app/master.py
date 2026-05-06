import ast
import asyncio
import io
import json
import logging
import os
import re
import secrets
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import redis
from fastapi import (Depends, FastAPI, File, Form, Header, HTTPException,
                     Request, UploadFile)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse


# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("master")


# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST            = os.getenv("REDIS_HOST", "redis")
REDIS_PORT            = int(os.getenv("REDIS_PORT", "6379"))
GPU_COUNT             = int(os.getenv("GPU_COUNT", "1"))
MAX_QUEUE_PER_STUDENT = int(os.getenv("MAX_QUEUE_PER_STUDENT", "3"))
MAX_DATASET_SIZE_MB   = int(os.getenv("MAX_DATASET_SIZE_MB", "500"))
ADMIN_API_KEY         = os.getenv("ADMIN_API_KEY", "").strip()
REQUIRE_STUDENT_KEY   = os.getenv("REQUIRE_STUDENT_KEY", "false").lower() == "true"
JOB_TTL               = 60 * 60 * 24
JOBS_DIR              = Path("/jobs")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

RUNTIME_PROFILES = {
    "pytorch-cu121": "pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime",
    "pytorch-cu118": "pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime",
    "tensorflow":    "tensorflow/tensorflow:2.15.0-gpu",
}

VALID_MEMORY_SUFFIXES = ("m", "g", "mb", "gb")

ALLOWED_REPO_RE = re.compile(
    r"^https://(github\.com|gitlab\.com)/[\w\-\.]+/[\w\-\.]+(?:\.git)?$"
)

DANGEROUS_BUILTINS = {"exec", "eval", "compile", "__import__"}

MAX_REQUIREMENTS_SIZE = 10 * 1024


# ── App ───────────────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory="app/templates")


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("=" * 50)
    log.info("GPU Job Scheduler Master starting (v2.1)")
    log.info(f"GPU_COUNT={GPU_COUNT} | REDIS={REDIS_HOST}:{REDIS_PORT}")
    log.info(f"MAX_QUEUE_PER_STUDENT={MAX_QUEUE_PER_STUDENT} | MAX_DATASET={MAX_DATASET_SIZE_MB}MB")
    log.info(f"Admin: {'enabled' if ADMIN_API_KEY else 'DISABLED (set ADMIN_API_KEY to enable)'}")
    log.info(f"Student API keys: {'required' if REQUIRE_STUDENT_KEY else 'optional'}")
    log.info(f"Runtimes: {list(RUNTIME_PROFILES.keys())}")
    log.info("=" * 50)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_gpus()
    yield


app = FastAPI(title="GPU Job Scheduler", version="2.1.0", lifespan=lifespan)


# ── Pydantic model (JSON API) ─────────────────────────────────────────────────
class JobRequest(BaseModel):
    code:         str
    runtime:      str   = "pytorch-cu121"
    cpus:         float = 1.0
    memory:       str   = "2g"
    student_id:   str   = "anonymous"
    name:         str   = ""
    description:  str   = ""
    requirements: str   = ""

    @field_validator("code")
    @classmethod
    def code_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("code must not be empty")
        return v

    @field_validator("cpus")
    @classmethod
    def cpus_range(cls, v):
        if not (0.1 <= v <= 8.0):
            raise ValueError("cpus must be between 0.1 and 8.0")
        return v

    @field_validator("memory")
    @classmethod
    def memory_format(cls, v):
        if not any(v.lower().endswith(s) for s in VALID_MEMORY_SUFFIXES):
            raise ValueError("memory must end with m, g, mb or gb")
        return v.lower()


# ── Safety ────────────────────────────────────────────────────────────────────
def check_code_safety(code: str) -> tuple[bool, str]:
    if len(code.encode()) > 200 * 1024:
        return False, "Code exceeds 200 KB size limit"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in DANGEROUS_BUILTINS:
                return False, f"Use of {node.func.id}() is not allowed"
    return True, ""


def check_requirements_safety(requirements: str) -> tuple[bool, str]:
    if not requirements:
        return True, ""
    if len(requirements.encode()) > MAX_REQUIREMENTS_SIZE:
        return False, f"requirements.txt exceeds {MAX_REQUIREMENTS_SIZE} bytes"
    for line in requirements.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Block options that fetch from arbitrary URLs / local files
        if s.startswith(("-e ", "--editable", "-f ", "--find-links", "-i ", "--index-url",
                         "--extra-index-url", "git+", "file:", "/")):
            return False, f"Disallowed line in requirements: '{s[:60]}'"
    return True, ""


# ── Authentication helpers ────────────────────────────────────────────────────
def resolve_student_id(api_key: str | None, fallback: str) -> str:
    """If a valid API key is provided, take student_id from it (trusted).
    Otherwise return fallback (untrusted form value)."""
    if api_key:
        raw = r.get(f"apikey:{api_key}")
        if raw:
            return json.loads(raw)["student_id"]
        if REQUIRE_STUDENT_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    elif REQUIRE_STUDENT_KEY:
        raise HTTPException(status_code=401, detail="API key required (X-API-Key header)")
    return (fallback or "anonymous").strip() or "anonymous"


def require_admin(x_admin_key: str | None = Header(default=None)):
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=503,
                            detail="Admin disabled — set ADMIN_API_KEY in environment")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key")


# ── Helpers ───────────────────────────────────────────────────────────────────
def iter_job_keys():
    """Yield only job:{uuid} keys, skipping job:logs:{uuid} (Redis lists)."""
    for key in r.scan_iter("job:*", count=100):
        if key.startswith("job:logs:"):
            continue
        yield key


def count_student_active_jobs(student_id: str) -> int:
    count = 0
    for key in iter_job_keys():
        raw = r.get(key)
        if raw:
            job = json.loads(raw)
            if job.get("student_id") == student_id and job.get("status") in ("queued", "running"):
                count += 1
    return count


def _build_job_data(
    *,
    job_id:       str,
    code:         str | None,
    repo_url:     str | None,
    runtime:      str,
    cpus:         float,
    memory:       str,
    student_id:   str,
    has_dataset:  bool = False,
    entrypoint:   str  = "main.py",
    name:         str  = "",
    description:  str  = "",
    requirements: str  = "",
) -> dict:
    return {
        "job_id":           job_id,
        "name":             name,
        "description":      description,
        "code":             code or "",
        "repo_url":         repo_url,
        "runtime":          runtime,
        "image":            RUNTIME_PROFILES[runtime],
        "cpus":             cpus,
        "memory":           memory,
        "student_id":       student_id,
        "has_dataset":      has_dataset,
        "has_requirements": bool(requirements),
        "requirements":     requirements,
        "entrypoint":       entrypoint,
        "status":           "queued",
        "gpu_id":           None,
        "container_name":   None,
        "cancel_requested": False,
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "started_at":       None,
        "finished_at":      None,
        "stdout":           "",
        "stderr":           "",
        "return_code":      None,
        "artifacts":        [],
    }


def _enqueue(job_data: dict) -> dict:
    student_id = job_data["student_id"]
    active     = count_student_active_jobs(student_id)
    if active >= MAX_QUEUE_PER_STUDENT:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Queue limit: student '{student_id}' already has {active} active job(s). "
                f"Max: {MAX_QUEUE_PER_STUDENT}"
            ),
        )
    job_id = job_data["job_id"]
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(job_data))
    r.rpush("job_queue", job_id)
    log.info(
        f"Job {job_id[:8]} queued | student={student_id} "
        f"runtime={job_data['runtime']} cpus={job_data['cpus']} mem={job_data['memory']}"
    )
    return {"job_id": job_id, "status": "queued", "runtime": job_data["runtime"],
            "image": job_data["image"]}


def initialize_gpus():
    initialized = 0
    for gpu_id in range(GPU_COUNT):
        key = f"gpu:{gpu_id}"
        if not r.get(key):
            r.set(key, json.dumps({
                "gpu_id":     gpu_id,
                "status":     "idle",
                "job_id":     None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }))
            initialized += 1
    if initialized:
        log.info(f"Initialized {initialized} GPU slot(s) in Redis")
    else:
        log.info("GPU slots already exist in Redis — skipping init")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {})


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return templates.TemplateResponse(request, "admin.html", {})


@app.get("/health")
def health():
    try:
        r.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status":    "ok" if redis_ok else "degraded",
        "redis":     "ok" if redis_ok else "unavailable",
        "gpu_count": GPU_COUNT,
    }


@app.get("/runtimes")
def list_runtimes():
    return {"runtimes": dict(RUNTIME_PROFILES)}


# ── Job submission: JSON (backward-compatible API) ────────────────────────────
@app.post("/submit-job", status_code=201)
def submit_job(job: JobRequest, x_api_key: str | None = Header(default=None)):
    if job.runtime not in RUNTIME_PROFILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported runtime '{job.runtime}'. Allowed: {list(RUNTIME_PROFILES.keys())}",
        )
    safe, reason = check_code_safety(job.code)
    if not safe:
        raise HTTPException(status_code=400, detail=f"Code safety check failed: {reason}")

    safe, reason = check_requirements_safety(job.requirements)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    student_id = resolve_student_id(x_api_key, job.student_id)
    job_id     = str(uuid.uuid4())
    return _enqueue(_build_job_data(
        job_id=job_id, code=job.code, repo_url=None,
        runtime=job.runtime, cpus=job.cpus, memory=job.memory,
        student_id=student_id, name=job.name.strip(),
        description=job.description.strip(), requirements=job.requirements.strip(),
    ))


# ── Job submission: multipart form (dashboard) ────────────────────────────────
@app.post("/submit-job-form", status_code=201)
async def submit_job_form(
    code:         str | None        = Form(None),
    code_file:    UploadFile | None = File(None),
    repo_url:     str | None        = Form(None),
    entrypoint:   str               = Form("main.py"),
    runtime:      str               = Form("pytorch-cu121"),
    cpus:         float             = Form(1.0),
    memory:       str               = Form("2g"),
    student_id:   str               = Form("anonymous"),
    name:         str               = Form(""),
    description:  str               = Form(""),
    requirements: str               = Form(""),
    dataset:      UploadFile | None = File(None),
    x_api_key:    str | None        = Header(default=None),
):
    if runtime not in RUNTIME_PROFILES:
        raise HTTPException(status_code=400, detail=f"Unsupported runtime '{runtime}'")
    if not (0.1 <= cpus <= 8.0):
        raise HTTPException(status_code=400, detail="cpus must be between 0.1 and 8.0")
    memory = memory.strip().lower()
    if not any(memory.endswith(s) for s in VALID_MEMORY_SUFFIXES):
        raise HTTPException(status_code=400, detail="memory must end with m, g, mb or gb")

    # Validate requirements first (cheap)
    safe, reason = check_requirements_safety(requirements)
    if not safe:
        raise HTTPException(status_code=400, detail=reason)

    # ── Resolve code source ───────────────────────────────────────────────────
    final_code: str | None = None
    final_repo: str | None = None

    if repo_url and repo_url.strip():
        repo_url = repo_url.strip()
        if not ALLOWED_REPO_RE.match(repo_url):
            raise HTTPException(
                status_code=400,
                detail="Only public GitHub/GitLab HTTPS URLs are allowed",
            )
        final_repo = repo_url

    elif code_file and code_file.filename:
        if not code_file.filename.endswith(".py"):
            raise HTTPException(status_code=400, detail="Only .py files are accepted")
        raw_bytes = await code_file.read()
        if len(raw_bytes) > 200 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 200 KB")
        final_code = raw_bytes.decode("utf-8", errors="replace")

    elif code and code.strip():
        final_code = code.strip()

    else:
        raise HTTPException(status_code=400, detail="Provide code, code_file, or repo_url")

    if final_code:
        safe, reason = check_code_safety(final_code)
        if not safe:
            raise HTTPException(status_code=400, detail=f"Code safety check failed: {reason}")

    # ── Handle dataset upload ─────────────────────────────────────────────────
    job_id      = str(uuid.uuid4())
    has_dataset = False

    if dataset and dataset.filename:
        max_bytes  = MAX_DATASET_SIZE_MB * 1024 * 1024
        data_bytes = await dataset.read()

        if len(data_bytes) > max_bytes:
            raise HTTPException(status_code=413,
                                detail=f"Dataset exceeds {MAX_DATASET_SIZE_MB} MB limit")
        if not zipfile.is_zipfile(io.BytesIO(data_bytes)):
            raise HTTPException(status_code=400, detail="Dataset must be a .zip archive")

        dataset_dir = JOBS_DIR / f"{job_id}_data"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dataset_dir / "upload.zip"
        zip_path.write_bytes(data_bytes)

        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.infolist():
                member_path = (dataset_dir / member.filename).resolve()
                if not str(member_path).startswith(str(dataset_dir.resolve())):
                    raise HTTPException(status_code=400, detail="Malicious zip file rejected")
            zf.extractall(dataset_dir)

        zip_path.unlink()
        has_dataset = True
        log.info(f"Job {job_id[:8]} dataset extracted ({len(data_bytes) // 1024} KB)")

    resolved_student = resolve_student_id(x_api_key, student_id)

    return _enqueue(_build_job_data(
        job_id=job_id, code=final_code, repo_url=final_repo,
        runtime=runtime, cpus=cpus, memory=memory,
        student_id=resolved_student,
        has_dataset=has_dataset, entrypoint=entrypoint.strip() or "main.py",
        name=name.strip(), description=description.strip(),
        requirements=requirements.strip(),
    ))


# ── Job queries ───────────────────────────────────────────────────────────────
@app.get("/jobs")
def list_jobs(status: str | None = None):
    jobs = []
    for key in iter_job_keys():
        raw = r.get(key)
        if raw:
            job = json.loads(raw)
            if status is None or job.get("status") == status:
                jobs.append(job)
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    job = json.loads(raw)
    if job.get("status") == "queued":
        try:
            pos = r.lpos("job_queue", job_id)
            job["queue_position"] = (pos + 1) if pos is not None else None
        except Exception:
            job["queue_position"] = None
    if job.get("started_at") and job.get("finished_at"):
        try:
            s = datetime.fromisoformat(job["started_at"])
            e = datetime.fromisoformat(job["finished_at"])
            job["execution_time_seconds"] = round((e - s).total_seconds(), 2)
        except Exception:
            job["execution_time_seconds"] = None
    else:
        job["execution_time_seconds"] = None
    return job


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    job = json.loads(raw)
    if job["status"] in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": job["status"], "message": "Job is already finished"}
    job["cancel_requested"] = True
    if job["status"] == "queued":
        job["status"]      = "cancelled"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(job))
    log.info(f"Job {job_id[:8]} cancel requested")
    return {"job_id": job_id, "status": job["status"], "cancel_requested": True}


# ── SSE log stream ────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}/logs/stream")
async def stream_logs(job_id: str, request: Request):
    raw = await asyncio.to_thread(r.get, f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            if await request.is_disconnected():
                break

            new_entries = await asyncio.to_thread(r.lrange, f"job:logs:{job_id}", sent, -1)
            for entry in new_entries:
                yield {"data": entry}
                sent += 1

            job_raw = await asyncio.to_thread(r.get, f"job:{job_id}")
            if job_raw:
                job = json.loads(job_raw)
                if job["status"] in ("completed", "failed", "cancelled"):
                    tail = await asyncio.to_thread(r.lrange, f"job:logs:{job_id}", sent, -1)
                    for entry in tail:
                        yield {"data": entry}
                    yield {"data": json.dumps({"type": "end", "status": job["status"]})}
                    return

            await asyncio.sleep(0.4)

    return EventSourceResponse(event_generator())


# ── Artifacts ─────────────────────────────────────────────────────────────────
@app.get("/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "artifacts": json.loads(raw).get("artifacts", [])}


@app.get("/jobs/{job_id}/artifacts/{filename:path}")
def download_artifact(job_id: str, filename: str):
    if not r.get(f"job:{job_id}"):
        raise HTTPException(status_code=404, detail="Job not found")
    output_dir = JOBS_DIR / f"{job_id}_output"
    file_path  = (output_dir / filename).resolve()
    if not str(file_path).startswith(str(output_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path=str(file_path), filename=file_path.name)


# ── GPU / Cluster status ──────────────────────────────────────────────────────
@app.get("/gpus")
def list_gpus():
    gpus = []
    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if raw:
            gpus.append(json.loads(raw))
    return {"gpus": gpus, "total": GPU_COUNT}


@app.get("/cluster-status")
def cluster_status():
    gpu_total = gpu_idle = gpu_busy = 0
    for gpu_id in range(GPU_COUNT):
        raw = r.get(f"gpu:{gpu_id}")
        if not raw:
            continue
        gpu = json.loads(raw)
        gpu_total += 1
        if gpu["status"] == "idle":
            gpu_idle += 1
        elif gpu["status"] == "busy":
            gpu_busy += 1

    counters = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
    durations = []
    for key in iter_job_keys():
        raw = r.get(key)
        if not raw:
            continue
        job = json.loads(raw)
        s = job.get("status")
        if s in counters:
            counters[s] += 1
        if s == "completed" and job.get("started_at") and job.get("finished_at"):
            try:
                start = datetime.fromisoformat(job["started_at"])
                end   = datetime.fromisoformat(job["finished_at"])
                durations.append((end - start).total_seconds())
            except Exception:
                pass

    avg_exec = round(sum(durations) / len(durations), 1) if durations else None

    return {
        "gpu_summary":  {"total": gpu_total, "idle": gpu_idle, "busy": gpu_busy},
        "job_summary":  {**counters, "total": sum(counters.values())},
        "queue_length": r.llen("job_queue"),
        "avg_execution_time_seconds": avg_exec,
    }


# ── Cleanup ───────────────────────────────────────────────────────────────────
@app.delete("/jobs/cleanup")
def cleanup_jobs(_=Depends(require_admin)):
    removed = 0
    for key in iter_job_keys():
        raw = r.get(key)
        if not raw:
            continue
        job = json.loads(raw)
        if job["status"] in ("completed", "failed", "cancelled"):
            job_id = job["job_id"]
            r.delete(key)
            r.delete(f"job:logs:{job_id}")
            for suffix in ("_output", "_data", "_repo"):
                d = JOBS_DIR / f"{job_id}{suffix}"
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            py_file = JOBS_DIR / f"{job_id}.py"
            if py_file.exists():
                py_file.unlink()
            removed += 1
    log.info(f"Cleanup: removed {removed} finished job(s)")
    return {"removed_jobs": removed}


# ────────────────────────────────────────────────────────────────────────────
# Admin API
# ────────────────────────────────────────────────────────────────────────────
@app.get("/admin/check")
def admin_check(_=Depends(require_admin)):
    return {"ok": True}


@app.get("/admin/stats")
def admin_stats(_=Depends(require_admin)):
    """Aggregated statistics: per-student, runtime usage, daily counts."""
    student_stats: dict[str, dict] = {}
    runtime_usage: dict[str, int]  = {}
    daily_jobs:    dict[str, int]  = {}
    overall_durations: list[float] = []

    for key in iter_job_keys():
        raw = r.get(key)
        if not raw:
            continue
        job = json.loads(raw)
        sid = job.get("student_id") or "anonymous"
        st  = student_stats.setdefault(sid, {
            "total": 0, "queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0,
            "total_gpu_time_seconds": 0.0,
        })
        st["total"] += 1
        status = job.get("status")
        if status in st:
            st[status] += 1

        if job.get("started_at") and job.get("finished_at"):
            try:
                d = (datetime.fromisoformat(job["finished_at"])
                     - datetime.fromisoformat(job["started_at"])).total_seconds()
                st["total_gpu_time_seconds"] += d
                if status == "completed":
                    overall_durations.append(d)
            except Exception:
                pass

        rt = job.get("runtime", "?")
        runtime_usage[rt] = runtime_usage.get(rt, 0) + 1

        date = (job.get("created_at") or "")[:10]
        if date:
            daily_jobs[date] = daily_jobs.get(date, 0) + 1

    students = sorted(
        [{"student_id": k, **v, "total_gpu_time_seconds": round(v["total_gpu_time_seconds"], 1)}
         for k, v in student_stats.items()],
        key=lambda s: s["total"], reverse=True,
    )

    avg = round(sum(overall_durations) / len(overall_durations), 1) if overall_durations else None
    longest = round(max(overall_durations), 1) if overall_durations else None

    return {
        "students":      students,
        "runtime_usage": runtime_usage,
        "daily_jobs":    dict(sorted(daily_jobs.items())),
        "overview": {
            "total_students":        len(student_stats),
            "total_jobs":            sum(s["total"] for s in student_stats.values()),
            "completed_jobs":        sum(s["completed"] for s in student_stats.values()),
            "avg_execution_seconds": avg,
            "longest_job_seconds":   longest,
        },
    }


@app.post("/admin/jobs/{job_id}/kill")
def admin_kill_job(job_id: str, _=Depends(require_admin)):
    raw = r.get(f"job:{job_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Job not found")
    job = json.loads(raw)
    if job["status"] in ("completed", "failed", "cancelled"):
        return {"job_id": job_id, "status": job["status"], "message": "already finished"}
    job["cancel_requested"] = True
    if job["status"] == "queued":
        job["status"]      = "cancelled"
        job["finished_at"] = datetime.now(timezone.utc).isoformat()
    r.setex(f"job:{job_id}", JOB_TTL, json.dumps(job))
    log.warning(f"Admin killed job {job_id[:8]}")
    return {"job_id": job_id, "status": job["status"]}


# ── API key management ────────────────────────────────────────────────────────
class ApiKeyCreate(BaseModel):
    student_id: str
    name:       str = ""


@app.post("/admin/keys")
def create_api_key(payload: ApiKeyCreate, _=Depends(require_admin)):
    student_id = payload.student_id.strip()
    if not student_id:
        raise HTTPException(status_code=400, detail="student_id is required")
    api_key = "key_" + secrets.token_urlsafe(24)
    record = {
        "student_id": student_id,
        "name":       payload.name.strip() or student_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    r.set(f"apikey:{api_key}", json.dumps(record))
    r.sadd("apikeys_set", api_key)
    log.info(f"Admin created API key for student '{student_id}'")
    return {"api_key": api_key, **record}


@app.get("/admin/keys")
def list_api_keys(_=Depends(require_admin)):
    keys = r.smembers("apikeys_set")
    out  = []
    for k in keys:
        raw = r.get(f"apikey:{k}")
        if raw:
            out.append({"api_key": k, **json.loads(raw)})
    out.sort(key=lambda x: x["created_at"], reverse=True)
    return {"keys": out}


@app.delete("/admin/keys/{api_key}")
def delete_api_key(api_key: str, _=Depends(require_admin)):
    r.delete(f"apikey:{api_key}")
    r.srem("apikeys_set", api_key)
    log.info(f"Admin revoked API key {api_key[:14]}...")
    return {"deleted": api_key}
