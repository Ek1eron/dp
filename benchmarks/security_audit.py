#!/usr/bin/env python3
"""
security_audit.py — Automated security audit for GPU Job Scheduler.

Tests 23 attack vectors across 6 categories. Each test sends a crafted HTTP
request to the live API and verifies the expected HTTP status is returned.

Output: console table + benchmarks/results/security_audit.csv
        + benchmarks/results/plots/security_audit.png

Usage:
    python benchmarks/security_audit.py
    ADMIN_API_KEY=xxx python benchmarks/security_audit.py
"""

import csv
import io
import os
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests

BASE_URL   = os.getenv("BASE_URL", "http://localhost:8001")
ADMIN_KEY  = os.getenv("ADMIN_API_KEY", "")
RESULTS    = "benchmarks/results"
PLOTS      = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

SUBMIT     = f"{BASE_URL}/submit-job"
SUBMIT_F   = f"{BASE_URL}/submit-job-form"
ADMIN_C    = f"{BASE_URL}/admin/check"


def _submit_json(code="print('ok')", **kwargs):
    return requests.post(SUBMIT, json={"code": code, **kwargs}, timeout=10)


def _submit_form(**data):
    files = data.pop("_files", {})
    return requests.post(SUBMIT_F, data=data, files=files, timeout=10)


def _make_zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@dataclass
class Check:
    category: str
    name: str
    description: str
    expect_status: int
    fn: object
    result: bool = field(default=False, init=False)
    actual_status: int = field(default=0, init=False)
    detail: str = field(default="", init=False)
    latency_ms: float = field(default=0.0, init=False)
    skipped: bool = field(default=False, init=False)

    def run(self):
        t0 = time.perf_counter()
        try:
            resp = self.fn()
            self.latency_ms = (time.perf_counter() - t0) * 1000
            self.actual_status = resp.status_code
            self.result = (resp.status_code == self.expect_status)
            try:
                body = resp.json()
                self.detail = body.get("detail", "")[:80]
            except Exception:
                self.detail = resp.text[:80]
        except Exception as e:
            self.latency_ms = (time.perf_counter() - t0) * 1000
            self.result = False
            self.detail = f"ERROR: {e}"


# ── Define all checks ─────────────────────────────────────────────────────────

CHECKS: list[Check] = [

    # ── 1. AST Code Safety ────────────────────────────────────────────────────
    Check(
        category="AST Code Safety",
        name="eval() blocked",
        description="Код з eval() відхиляється на рівні AST до постановки в чергу",
        expect_status=400,
        fn=lambda: _submit_json('eval("1+1")'),
    ),
    Check(
        category="AST Code Safety",
        name="exec() blocked",
        description="Код з exec() відхиляється на рівні AST",
        expect_status=400,
        fn=lambda: _submit_json('exec("import os; os.system(\'id\')")'),
    ),
    Check(
        category="AST Code Safety",
        name="compile() blocked",
        description="compile() — динамічна генерація коду заблокована",
        expect_status=400,
        fn=lambda: _submit_json('compile("import os", "f", "exec")'),
    ),
    Check(
        category="AST Code Safety",
        name="__import__() blocked",
        description="__import__() — обхід стандартного import заблокований",
        expect_status=400,
        fn=lambda: _submit_json('__import__("subprocess")'),
    ),
    Check(
        category="AST Code Safety",
        name="Syntax error rejected",
        description="Код з синтаксичною помилкою повертає 400 (не 500)",
        expect_status=400,
        fn=lambda: _submit_json("def foo(\n"),
    ),
    Check(
        category="AST Code Safety",
        name="Code > 200 KB rejected",
        description="Код більший за 200 КБ відхиляється",
        expect_status=400,
        fn=lambda: _submit_json("x = 1\n" * 35_000),  # ~210 KB > 200 KB limit
    ),

    # ── 2. Requirements Safety ────────────────────────────────────────────────
    Check(
        category="Requirements Safety",
        name="-e (editable) blocked",
        description="Локальні editable installs заблоковані",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="-e ./local_pkg"),
    ),
    Check(
        category="Requirements Safety",
        name="git+ URL blocked",
        description="Встановлення з git репозиторіїв заблоковане",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="git+https://github.com/evil/payload"),
    ),
    Check(
        category="Requirements Safety",
        name="-i (index-url) blocked",
        description="Кастомний PyPI-сервер (зловмисний mirror) заблокований",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="-i https://evil.example.com\nrequests"),
    ),
    Check(
        category="Requirements Safety",
        name="--find-links blocked",
        description="--find-links до довільного URL заблокований",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="--find-links http://evil.com\nrequests"),
    ),
    Check(
        category="Requirements Safety",
        name="file: URL blocked",
        description="Встановлення з file:// (локальна файлова система) заблоковане",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="file:///etc/passwd"),
    ),
    Check(
        category="Requirements Safety",
        name="Absolute path blocked",
        description="Абсолютний шлях /path/to/pkg заблокований",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', requirements="/etc/secret_package"),
    ),

    # ── 3. Input Validation ───────────────────────────────────────────────────
    Check(
        category="Input Validation",
        name="Unknown runtime → 400",
        description="Невідомий runtime відхиляється",
        expect_status=400,
        fn=lambda: _submit_json('print("ok")', runtime="malicious-runtime"),
    ),
    Check(
        category="Input Validation",
        name="cpus > 8 → 422",
        description="Запит більше 8 CPU відхиляється валідатором",
        expect_status=422,
        fn=lambda: _submit_json('print("ok")', cpus=99.0),
    ),
    Check(
        category="Input Validation",
        name="Invalid memory format → 422",
        description="Некоректний формат memory (без суфікса) відхиляється",
        expect_status=422,
        fn=lambda: _submit_json('print("ok")', memory="1024"),
    ),
    Check(
        category="Input Validation",
        name="Empty code → 422",
        description="Порожній код відхиляється валідатором",
        expect_status=422,
        fn=lambda: _submit_json("   "),
    ),
    Check(
        category="Input Validation",
        name="Non-.py file → 400",
        description="Завантаження .txt файлу замість .py відхиляється",
        expect_status=400,
        fn=lambda: _submit_form(
            runtime="pytorch-cu121",
            _files={"code_file": ("script.txt", b"print('x')", "text/plain")},
        ),
    ),
    Check(
        category="Input Validation",
        name="Non-zip dataset → 400",
        description="Датасет не у форматі ZIP відхиляється",
        expect_status=400,
        fn=lambda: _submit_form(
            code='print("x")',
            runtime="pytorch-cu121",
            _files={"dataset": ("data.csv", b"a,b\n1,2", "text/csv")},
        ),
    ),

    # ── 4. Repo URL Validation ────────────────────────────────────────────────
    Check(
        category="Repo URL Validation",
        name="Arbitrary host blocked",
        description="git clone з довільного хоста заблокований (тільки GitHub/GitLab)",
        expect_status=400,
        fn=lambda: _submit_form(
            repo_url="https://evil.example.com/user/payload",
            runtime="pytorch-cu121",
        ),
    ),
    Check(
        category="Repo URL Validation",
        name="GitHub URL accepted",
        description="Легітимний GitHub URL приймається (format validation)",
        expect_status=201,
        fn=lambda: requests.post(
            SUBMIT_F,
            data={"repo_url": "https://github.com/octocat/Hello-World",
                  "runtime": "pytorch-cu121"},
            timeout=10,
        ),
    ),

    # ── 5. Rate Limiting ──────────────────────────────────────────────────────
    Check(
        category="Rate Limiting",
        name="Queue limit → 429",
        description="Студент не може поставити >3 задач одночасно",
        expect_status=429,
        fn=None,  # handled specially below
    ),

    # ── 6. Admin Auth ─────────────────────────────────────────────────────────
    Check(
        category="Admin Auth",
        name="No admin key → 401",
        description="Admin endpoint без ключа повертає 401",
        expect_status=401,
        fn=lambda: requests.get(ADMIN_C, timeout=10),
    ),
    Check(
        category="Admin Auth",
        name="Wrong admin key → 401",
        description="Неправильний admin ключ повертає 401",
        expect_status=401,
        fn=lambda: requests.get(ADMIN_C, headers={"X-Admin-Key": "wrong-key-brute-force"}, timeout=10),
    ),
]


def _run_rate_limit_check(check: Check):
    sid = f"audit_{int(time.time())}"
    ids = []
    try:
        for _ in range(3):
            r = _submit_json("import time\ntime.sleep(120)", student_id=sid)
            if r.status_code == 201:
                ids.append(r.json()["job_id"])
        t0 = time.perf_counter()
        resp = _submit_json('print("over limit")', student_id=sid)
        check.latency_ms = (time.perf_counter() - t0) * 1000
        check.actual_status = resp.status_code
        check.result = (resp.status_code == 429)
        try:
            check.detail = resp.json().get("detail", "")[:80]
        except Exception:
            check.detail = resp.text[:80]
    except Exception as e:
        check.result = False
        check.detail = f"ERROR: {e}"
    finally:
        for jid in ids:
            try:
                requests.post(f"{BASE_URL}/jobs/{jid}/cancel", timeout=5)
            except Exception:
                pass


def _run_github_cleanup(check: Check):
    try:
        body = check.fn()
        if body.status_code == 201:
            jid = body.json().get("job_id")
            if jid:
                requests.post(f"{BASE_URL}/jobs/{jid}/cancel", timeout=5)
    except Exception:
        pass


# ── Run all checks ────────────────────────────────────────────────────────────

def run_all() -> list[Check]:
    print(f"\n{'='*70}")
    print(f"  GPU Job Scheduler — Security Audit")
    print(f"  Target : {BASE_URL}")
    print(f"  Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    for i, check in enumerate(CHECKS, 1):
        # Rate limit check requires special setup
        if check.fn is None:
            _run_rate_limit_check(check)
        else:
            check.run()
            # Cleanup the queued GitHub job
            if check.name == "GitHub URL accepted" and check.actual_status == 201:
                try:
                    body = requests.post(SUBMIT_F,
                        data={"repo_url": "https://github.com/octocat/Hello-World",
                              "runtime": "pytorch-cu121"}, timeout=10)
                    # We already called fn() inside check.run() — cancel that job
                except Exception:
                    pass

        icon = "✓" if check.result else "✗"
        status_str = f"got {check.actual_status} (want {check.expect_status})"
        print(f"  [{icon}] [{check.category:22s}] {check.name:<30s} {status_str}  {check.latency_ms:5.0f}ms")

    return CHECKS


# ── Cancel the GitHub job that got queued ────────────────────────────────────

def _cancel_github_jobs():
    try:
        resp = requests.get(f"{BASE_URL}/jobs", timeout=5)
        jobs = resp.json().get("jobs", [])
        for j in jobs:
            if j.get("repo_url", "").startswith("https://github.com") and j.get("status") in ("queued", "running"):
                requests.post(f"{BASE_URL}/jobs/{j['job_id']}/cancel", timeout=5)
    except Exception:
        pass


# ── Save results ──────────────────────────────────────────────────────────────

def save_csv(checks: list[Check]):
    path = f"{RESULTS}/security_audit.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["#", "Категорія", "Перевірка", "Опис", "Очікуваний HTTP",
                         "Отриманий HTTP", "Результат", "Деталь", "Затримка (мс)"])
        for i, c in enumerate(checks, 1):
            writer.writerow([
                i, c.category, c.name, c.description,
                c.expect_status, c.actual_status,
                "PASS" if c.result else "FAIL",
                c.detail, f"{c.latency_ms:.1f}",
            ])
    print(f"\n  CSV збережено: {path}")
    return path


def save_plot(checks: list[Check]):
    categories = {}
    for c in checks:
        categories.setdefault(c.category, []).append(c)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("GPU Job Scheduler — Security Audit Results", fontsize=14, fontweight="bold")

    # ── Left: per-category pass/fail bar chart ────────────────────────────────
    ax = axes[0]
    cat_names   = list(categories.keys())
    pass_counts = [sum(1 for c in v if c.result)      for v in categories.values()]
    fail_counts = [sum(1 for c in v if not c.result)  for v in categories.values()]
    total       = [p + f for p, f in zip(pass_counts, fail_counts)]
    x = range(len(cat_names))

    bars_pass = ax.barh(x, pass_counts, color="#2ecc71", label="PASS", height=0.5)
    bars_fail = ax.barh(x, fail_counts, left=pass_counts, color="#e74c3c", label="FAIL", height=0.5)

    ax.set_yticks(list(x))
    ax.set_yticklabels(cat_names, fontsize=9)
    ax.set_xlabel("Кількість перевірок")
    ax.set_title("Результати по категоріях")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(total) + 1)

    for xi, (p, t) in enumerate(zip(pass_counts, total)):
        ax.text(t + 0.05, xi, f"{p}/{t}", va="center", fontsize=8)

    # ── Right: latency distribution ───────────────────────────────────────────
    ax2 = axes[1]
    latencies   = [c.latency_ms for c in checks]
    colors      = ["#2ecc71" if c.result else "#e74c3c" for c in checks]
    labels      = [c.name for c in checks]
    y_pos       = range(len(checks))

    ax2.barh(y_pos, latencies, color=colors, height=0.6)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_xlabel("Час відповіді (мс)")
    ax2.set_title("Час відповіді на кожну перевірку")
    ax2.axvline(x=100, color="orange", linestyle="--", alpha=0.7, label="100 мс")
    ax2.legend(fontsize=8)

    passed_patch = mpatches.Patch(color="#2ecc71", label="PASS (атака заблокована)")
    failed_patch = mpatches.Patch(color="#e74c3c", label="FAIL (вразливість!)")
    fig.legend(handles=[passed_patch, failed_patch], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    path = f"{PLOTS}/security_audit.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG збережено : {path}")
    return path


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(checks: list[Check]):
    total  = len(checks)
    passed = sum(1 for c in checks if c.result)
    failed = total - passed

    print(f"\n{'='*70}")
    print(f"  ПІДСУМОК: {passed}/{total} перевірок пройдено")
    if failed:
        print(f"\n  ПРОВАЛЕНІ ({failed}):")
        for c in checks:
            if not c.result:
                print(f"    ✗ [{c.category}] {c.name}")
                print(f"      очікував HTTP {c.expect_status}, отримав {c.actual_status}")
                if c.detail:
                    print(f"      {c.detail}")
    else:
        print("  Всі перевірки пройдено — система захищена ✓")
    print(f"{'='*70}\n")

    # Table for diploma
    print("  Таблиця для диплому:\n")
    print(f"  {'№':<3} {'Категорія':<24} {'Атака / Перевірка':<32} {'HTTP':<6} {'Результат'}")
    print(f"  {'-'*3} {'-'*24} {'-'*32} {'-'*6} {'-'*10}")
    for i, c in enumerate(checks, 1):
        res = "PASS ✓" if c.result else "FAIL ✗"
        print(f"  {i:<3} {c.category:<24} {c.name:<32} {c.actual_status:<6} {res}")


if __name__ == "__main__":
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"ПОМИЛКА: не можу підключитись до {BASE_URL}: {e}")
        print("Запустіть: docker compose up -d")
        sys.exit(1)

    checks = run_all()
    _cancel_github_jobs()
    print_summary(checks)
    save_csv(checks)
    save_plot(checks)
