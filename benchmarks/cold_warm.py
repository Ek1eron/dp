#!/usr/bin/env python3
"""
cold_warm.py — Cold start vs warm start benchmark.

"Cold start" = перший запуск runtime (Docker image вже є, але шар кешу CPU холодний)
"Warm start"  = наступні запуски після того як перший відпрацював

Метрика: час від submit до статусу running (= час запуску контейнера)

Output: console + benchmarks/results/cold_warm.csv
                + benchmarks/results/plots/cold_warm.png

Usage:
    python benchmarks/cold_warm.py            # 1 cold + 5 warm (default)
    WARM_RUNS=10 python benchmarks/cold_warm.py
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

BASE_URL   = os.getenv("BASE_URL",   "http://localhost:8001")
WARM_RUNS  = int(os.getenv("WARM_RUNS", "5"))
TIMEOUT    = 120
RESULTS    = "benchmarks/results"
PLOTS      = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

# Short job — we measure startup, not compute time
SHORT_CODE = """\
import torch
print(f"container started | GPU: {torch.cuda.get_device_name(0)}")
"""


def submit(run_label: str) -> tuple[str, float]:
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={
            "code":       SHORT_CODE,
            "runtime":    "pytorch-cu121",
            "student_id": "cold_warm_bench",
            "name":       f"cw-{run_label}",
        },
        timeout=10,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"Submit failed: {resp.status_code}")
    return resp.json()["job_id"], t0


def measure_startup(job_id: str, submit_ts: float) -> dict:
    """Poll until running, record time-to-start and time-to-complete."""
    startup_ts  = None
    finished_ts = None
    deadline    = time.time() + TIMEOUT

    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job  = resp.json()
        except Exception:
            time.sleep(0.3)
            continue

        status = job["status"]

        if status == "running" and startup_ts is None:
            startup_ts = time.time()
            # Use server-side timestamp if available
            if job.get("started_at"):
                try:
                    from datetime import datetime, timezone
                    startup_ts = datetime.fromisoformat(
                        job["started_at"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass

        if status in ("completed", "failed", "cancelled"):
            finished_ts = time.time()
            if job.get("finished_at"):
                try:
                    from datetime import datetime, timezone
                    finished_ts = datetime.fromisoformat(
                        job["finished_at"].replace("Z", "+00:00")
                    ).timestamp()
                except Exception:
                    pass
            return {
                "job_id":      job_id,
                "status":      status,
                "startup_s":   (startup_ts or finished_ts) - submit_ts,
                "total_s":     finished_ts - submit_ts,
            }

        time.sleep(0.3)

    raise TimeoutError(f"Job {job_id[:8]} timed out")


def wait_for_gpu_idle(timeout: int = 30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/gpus", timeout=5)
            gpus = resp.json().get("gpus", [])
            if all(g["status"] == "idle" for g in gpus):
                return
        except Exception:
            pass
        time.sleep(1)


def run_benchmark() -> dict:
    print(f"\n{'='*60}")
    print(f"  Cold / Warm Start Benchmark")
    print(f"  Target: {BASE_URL}  |  Warm runs: {WARM_RUNS}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    rows = []

    # ── Cold start ────────────────────────────────────────────────────────────
    print("\n  [Cold] Перший запуск...")
    try:
        wait_for_gpu_idle()
        job_id, t_submit = submit("cold")
        row = measure_startup(job_id, t_submit)
        row["type"] = "cold"
        row["run"]  = 0
        rows.append(row)
        icon = "✓" if row["status"] == "completed" else "✗"
        print(f"  {icon} cold  startup={row['startup_s']:.2f}s  total={row['total_s']:.2f}s  ({row['status']})")
    except Exception as e:
        print(f"  ✗ cold  ERROR: {e}")

    # ── Warm starts ───────────────────────────────────────────────────────────
    print()
    for i in range(1, WARM_RUNS + 1):
        try:
            wait_for_gpu_idle(timeout=60)
            job_id, t_submit = submit(f"warm-{i}")
            row = measure_startup(job_id, t_submit)
            row["type"] = "warm"
            row["run"]  = i
            rows.append(row)
            icon = "✓" if row["status"] == "completed" else "✗"
            print(f"  {icon} warm {i:>2}  startup={row['startup_s']:.2f}s  total={row['total_s']:.2f}s")
        except Exception as e:
            print(f"  ✗ warm {i:>2}  ERROR: {e}")

    return rows


def print_summary(rows: list[dict]):
    cold = [r for r in rows if r["type"] == "cold" and r["status"] == "completed"]
    warm = [r for r in rows if r["type"] == "warm" and r["status"] == "completed"]

    print(f"\n{'='*60}")
    print(f"  ПІДСУМОК")
    if cold:
        print(f"  Cold start: {cold[0]['startup_s']:.2f}s до запуску, {cold[0]['total_s']:.2f}s загалом")
    if warm:
        avg_w = np.mean([r["startup_s"] for r in warm])
        print(f"  Warm start: {avg_w:.2f}s середнє до запуску (N={len(warm)})")
    if cold and warm:
        ratio = cold[0]["startup_s"] / np.mean([r["startup_s"] for r in warm])
        print(f"  Cold/warm ratio: {ratio:.1f}x")
    print(f"{'='*60}\n")


def save_csv(rows: list[dict]):
    if not rows:
        return
    path = f"{RESULTS}/cold_warm.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV збережено: {path}")


def save_plot(rows: list[dict]):
    if not rows:
        return

    cold_rows = [r for r in rows if r["type"] == "cold"]
    warm_rows = [r for r in rows if r["type"] == "warm"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GPU Job Scheduler — Cold vs Warm Container Startup", fontsize=13, fontweight="bold")

    # ── Left: timeline chart ──────────────────────────────────────────────────
    ax = axes[0]
    all_rows = cold_rows + warm_rows
    labels   = ["Cold"] + [f"Warm {r['run']}" for r in warm_rows]
    startups = [r["startup_s"] for r in all_rows]
    colors_  = ["#e74c3c"] + ["#3498db"] * len(warm_rows)

    y = np.arange(len(all_rows))
    ax.barh(y, startups, color=colors_, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Час до переходу у статус 'running' (с)")
    ax.set_title("Час запуску контейнера")

    for yi, val in enumerate(startups):
        ax.text(val + 0.02, yi, f"{val:.2f}s", va="center", fontsize=8)

    # ── Right: box / bar comparison ───────────────────────────────────────────
    ax2 = axes[1]
    cold_val  = [r["startup_s"] for r in cold_rows] or [0]
    warm_vals = [r["startup_s"] for r in warm_rows] or [0]

    categories = ["Cold start", "Warm start\n(середнє)"]
    values     = [np.mean(cold_val), np.mean(warm_vals)]
    errors     = [np.std(cold_val),  np.std(warm_vals)]
    colors2    = ["#e74c3c", "#3498db"]

    bars = ax2.bar(categories, values, color=colors2, width=0.4, yerr=errors,
                   capsize=5, error_kw={"linewidth": 2})
    ax2.set_ylabel("Час запуску (с)")
    ax2.set_title("Cold vs Warm (середнє ± std)")

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(errors) * 0.1,
                 f"{val:.2f}s", ha="center", va="bottom", fontsize=11, fontweight="bold")

    if values[1] > 0:
        ratio = values[0] / values[1]
        ax2.text(0.5, 0.92, f"Cold/Warm = {ratio:.1f}×",
                 transform=ax2.transAxes, ha="center", fontsize=11,
                 color="darkred", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = f"{PLOTS}/cold_warm.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PNG збережено : {path}")


if __name__ == "__main__":
    try:
        requests.get(f"{BASE_URL}/health", timeout=5)
    except Exception as e:
        print(f"ПОМИЛКА: не можу підключитись до {BASE_URL}: {e}")
        sys.exit(1)

    rows = run_benchmark()
    print_summary(rows)
    save_csv(rows)
    save_plot(rows)
