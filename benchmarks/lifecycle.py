#!/usr/bin/env python3
"""
lifecycle.py — Job lifecycle timing benchmark.

Measures time spent in each phase for N jobs:
  submit → queue_wait → running → completed

Each job runs: torch.randn(3000, 3000) @ torch.randn(3000, 3000)
(real GPU computation, ~2–5s on modern hardware)

Output: console + benchmarks/results/lifecycle.csv
                + benchmarks/results/plots/lifecycle.png

Usage:
    python benchmarks/lifecycle.py          # 10 jobs (default)
    N=20 python benchmarks/lifecycle.py
"""

import csv
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

BASE_URL   = os.getenv("BASE_URL",  "http://localhost:8001")
N_JOBS     = int(os.getenv("N",     "10"))
POLL_EVERY = 0.5   # seconds
TIMEOUT    = 180   # seconds per job
RESULTS    = "benchmarks/results"
PLOTS      = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

GPU_CODE = """\
import torch, time, os
t0 = time.perf_counter()
A = torch.randn(3000, 3000, device='cuda')
B = torch.randn(3000, 3000, device='cuda')
C = A @ B
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"matmul 3000x3000 in {elapsed:.3f}s on {torch.cuda.get_device_name(0)}")
"""


def submit_job(run_idx: int) -> str:
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={
            "code":       GPU_CODE,
            "runtime":    "pytorch-cu121",
            "student_id": f"lifecycle_bench_{run_idx}",
            "name":       f"lifecycle-{run_idx:02d}",
        },
        timeout=10,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"Submit failed: {resp.status_code} {resp.text[:200]}")
    return resp.json()["job_id"]


def wait_and_measure(job_id: str, submit_ts: float) -> dict:
    running_ts  = None
    finished_ts = None
    deadline    = time.time() + TIMEOUT

    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job  = resp.json()
        except Exception:
            time.sleep(POLL_EVERY)
            continue

        status = job["status"]

        if status == "running" and running_ts is None:
            running_ts = time.time()

        if status in ("completed", "failed", "cancelled"):
            finished_ts = time.time()
            # Prefer server-side timestamps if available
            if job.get("started_at"):
                try:
                    from datetime import datetime, timezone
                    running_ts = datetime.fromisoformat(
                        job["started_at"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            if job.get("finished_at"):
                try:
                    from datetime import datetime, timezone
                    finished_ts = datetime.fromisoformat(
                        job["finished_at"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            return {
                "job_id":         job_id,
                "status":         status,
                "return_code":    job.get("return_code"),
                "queue_wait_s":   (running_ts or finished_ts) - submit_ts,
                "execution_s":    (finished_ts - (running_ts or submit_ts)) if running_ts else 0,
                "total_s":        finished_ts - submit_ts,
                "gpu_id":         job.get("gpu_id"),
            }

        time.sleep(POLL_EVERY)

    raise TimeoutError(f"Job {job_id[:8]} did not finish within {TIMEOUT}s")


def run_benchmark() -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  Lifecycle Benchmark — {N_JOBS} jobs")
    print(f"  Target: {BASE_URL}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  {'#':<4} {'job_id':>10}  {'queue (s)':>10}  {'exec (s)':>10}  {'total (s)':>10}  status")
    print(f"  {'-'*4} {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'------'}")

    results = []
    for i in range(1, N_JOBS + 1):
        try:
            t_submit = time.time()
            job_id   = submit_job(i)
            row      = wait_and_measure(job_id, t_submit)
            row["run"] = i
            results.append(row)
            ok = "✓" if row["status"] == "completed" else "✗"
            print(
                f"  {i:<4} {job_id[:8]:>10}  "
                f"{row['queue_wait_s']:>10.2f}  "
                f"{row['execution_s']:>10.2f}  "
                f"{row['total_s']:>10.2f}  "
                f"{ok} {row['status']}"
            )
        except Exception as e:
            print(f"  {i:<4} {'ERROR':>10}  {'—':>10}  {'—':>10}  {'—':>10}  {e}")

    return results


def print_summary(results: list[dict]):
    ok = [r for r in results if r["status"] == "completed"]
    if not ok:
        print("\n  Немає успішних задач — перевірте сервіс.")
        return

    q  = [r["queue_wait_s"] for r in ok]
    e  = [r["execution_s"]  for r in ok]
    t  = [r["total_s"]      for r in ok]

    print(f"\n{'='*60}")
    print(f"  ПІДСУМОК ({len(ok)}/{len(results)} успішних)")
    print(f"  {'Метрика':<20} {'avg':>8} {'min':>8} {'max':>8} {'p95':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, vals in [("Очікування (s)", q), ("Виконання (s)", e), ("Загалом (s)", t)]:
        arr = sorted(vals)
        p95 = arr[int(len(arr) * 0.95) - 1] if len(arr) > 1 else arr[0]
        print(f"  {label:<20} {np.mean(vals):>8.2f} {min(vals):>8.2f} {max(vals):>8.2f} {p95:>8.2f}")
    print(f"{'='*60}\n")


def save_csv(results: list[dict]):
    path = f"{RESULTS}/lifecycle.csv"
    if not results:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV збережено: {path}")


def save_plot(results: list[dict]):
    ok = [r for r in results if r["status"] == "completed"]
    if not ok:
        return

    runs = [r["run"] for r in ok]
    q    = [r["queue_wait_s"] for r in ok]
    e    = [r["execution_s"]  for r in ok]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPU Job Scheduler — Job Lifecycle Timing", fontsize=13, fontweight="bold")

    # ── Stacked bar: queue + execution ───────────────────────────────────────
    ax = axes[0]
    x  = np.arange(len(runs))
    b1 = ax.bar(x, q, label="Очікування в черзі", color="#3498db", width=0.6)
    b2 = ax.bar(x, e, bottom=q, label="Виконання (GPU)", color="#e74c3c", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{r}" for r in runs], fontsize=8)
    ax.set_xlabel("Запуск")
    ax.set_ylabel("Час (с)")
    ax.set_title("Розбивка часу по фазах")
    ax.legend()

    for xi, (qi, ei) in enumerate(zip(q, e)):
        total = qi + ei
        ax.text(xi, total + 0.1, f"{total:.1f}s", ha="center", fontsize=7)

    # ── Averages summary bar ──────────────────────────────────────────────────
    ax2 = axes[1]
    avg_q = np.mean(q)
    avg_e = np.mean(e)
    categories = ["Черга", "Виконання", "Загалом"]
    values     = [avg_q, avg_e, avg_q + avg_e]
    colors_    = ["#3498db", "#e74c3c", "#2ecc71"]
    bars = ax2.bar(categories, values, color=colors_, width=0.5)
    ax2.set_ylabel("Середній час (с)")
    ax2.set_title(f"Середні значення (N={len(ok)})")
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Annotation
    ax2.text(0.5, 0.95,
             f"GPU обчислення: matmul(3000×3000)\nРантайм: pytorch-cu121",
             transform=ax2.transAxes, ha="center", va="top",
             fontsize=8, color="gray",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.7))

    plt.tight_layout()
    path = f"{PLOTS}/lifecycle.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG збережено : {path}")


if __name__ == "__main__":
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"ПОМИЛКА: не можу підключитись до {BASE_URL}: {e}")
        sys.exit(1)

    results = run_benchmark()
    print_summary(results)
    save_csv(results)
    save_plot(results)
