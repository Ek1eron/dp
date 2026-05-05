#!/usr/bin/env python3
"""
concurrency.py — Concurrent submission stress test.

Доводить коректну атомарну резервацію GPU через Redis WATCH/MULTI/EXEC:
- жодна GPU не призначається двом задачам одночасно
- throughput зростає лінійно зі збільшенням GPU (в межах GPU_COUNT)

Сценарії: 1, 5, 10, 20 паралельних submit у кожному раунді.

Output: console + benchmarks/results/concurrency.csv
                + benchmarks/results/plots/concurrency.png

Usage:
    python benchmarks/concurrency.py
    LEVELS=1,5,10 python benchmarks/concurrency.py
"""

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

BASE_URL   = os.getenv("BASE_URL",  "http://localhost:8001")
LEVELS_RAW = os.getenv("LEVELS",    "1,5,10,20")
LEVELS     = [int(x) for x in LEVELS_RAW.split(",")]
JOB_WAIT   = int(os.getenv("JOB_WAIT", "90"))   # max seconds to wait per job
RESULTS    = "benchmarks/results"
PLOTS      = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

# Quick job for throughput: just a matmul to confirm GPU works
QUICK_CODE = """\
import torch
A = torch.randn(1500, 1500, device='cuda')
C = A @ A.T
torch.cuda.synchronize()
print("done")
"""


def submit_one(worker_id: int, level: int) -> dict:
    t0   = time.time()
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={
            "code":       QUICK_CODE,
            "runtime":    "pytorch-cu121",
            "student_id": f"concurrency_w{worker_id}_l{level}",
            "name":       f"cc-l{level}-w{worker_id:02d}",
        },
        timeout=10,
    )
    submit_latency = time.time() - t0
    if resp.status_code != 201:
        return {"worker_id": worker_id, "status": "submit_failed",
                "submit_latency_s": submit_latency, "job_id": None, "error": resp.text[:100]}
    return {
        "worker_id":         worker_id,
        "job_id":            resp.json()["job_id"],
        "submit_latency_s":  submit_latency,
        "status":            "submitted",
        "error":             None,
    }


def wait_for_job(job_id: str, timeout: int) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job  = resp.json()
            if job["status"] in ("completed", "failed", "cancelled"):
                return job
        except Exception:
            pass
        time.sleep(0.5)
    return {"status": "timeout", "gpu_id": None, "started_at": None, "finished_at": None}


def _parse_ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _check_time_overlap(completed: list[dict]) -> bool:
    """True if any two jobs ran concurrently on the same GPU (real race condition)."""
    intervals: dict[int, list[tuple[float, float]]] = {}
    for job in completed:
        gpu  = job.get("gpu_id")
        t_s  = _parse_ts(job.get("started_at"))
        t_e  = _parse_ts(job.get("finished_at"))
        if gpu is None or t_s is None or t_e is None:
            continue
        for prev_s, prev_e in intervals.get(gpu, []):
            # Overlap: one starts before the other ends
            if t_s < prev_e and t_e > prev_s:
                return True
        intervals.setdefault(gpu, []).append((t_s, t_e))
    return False


def run_level(level: int) -> dict:
    print(f"\n  ── Level {level:>2} concurrent submissions ──")

    t_wave_start = time.time()

    # Submit all in parallel
    with ThreadPoolExecutor(max_workers=level) as ex:
        futures  = {ex.submit(submit_one, i, level): i for i in range(level)}
        submitted = []
        for fut in as_completed(futures):
            submitted.append(fut.result())

    submit_done = time.time()
    ok_submitted = [s for s in submitted if s["job_id"]]

    print(f"    Submitted {len(ok_submitted)}/{level}  ({submit_done - t_wave_start:.2f}s)")

    # Wait for all submitted jobs to finish
    with ThreadPoolExecutor(max_workers=min(level, 20)) as ex:
        futures = {ex.submit(wait_for_job, s["job_id"], JOB_WAIT): s for s in ok_submitted}
        finished = []
        for fut in as_completed(futures):
            sub  = futures[fut]
            job  = fut.result()
            finished.append({
                **sub,
                "final_status": job["status"],
                "gpu_id":       job.get("gpu_id"),
                "started_at":   job.get("started_at"),
                "finished_at":  job.get("finished_at"),
            })

    t_wave_end = time.time()
    wave_total = t_wave_end - t_wave_start

    completed = [f for f in finished if f["final_status"] == "completed"]
    failed    = [f for f in finished if f["final_status"] == "failed"]
    timeout_  = [f for f in finished if f["final_status"] == "timeout"]

    # Race condition check: did any two jobs run simultaneously on the same GPU?
    # Correct method: check overlapping [started_at, finished_at] windows per gpu_id.
    # Simply comparing gpu_ids is wrong for single-GPU setups — all jobs use gpu_id=0
    # sequentially, which is correct behavior.
    race_detected = _check_time_overlap(completed)

    throughput = len(completed) / wave_total if wave_total > 0 else 0

    print(f"    Completed={len(completed)}  Failed={len(failed)}  Timeout={len(timeout_)}")
    print(f"    Wave time={wave_total:.1f}s  Throughput={throughput:.2f} jobs/s")
    if race_detected:
        print(f"    !! RACE CONDITION DETECTED — overlapping GPU execution windows !!")
    else:
        print(f"    GPU isolation: OK (no overlapping execution on same GPU)")

    return {
        "level":         level,
        "submitted":     len(ok_submitted),
        "completed":     len(completed),
        "failed":        len(failed),
        "timeout":       len(timeout_),
        "wave_time_s":   wave_total,
        "throughput":    throughput,
        "race_detected": race_detected,
        "submit_rows":   finished,
    }


def run_benchmark() -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  Concurrency / Throughput Benchmark")
    print(f"  Target: {BASE_URL}  |  Levels: {LEVELS}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print("  NOTE: очікування між рівнями щоб GPU звільнились...")

    results = []
    for level in LEVELS:
        row = run_level(level)
        results.append(row)
        # Wait for all GPUs to be free before next level
        _wait_all_idle()

    return results


def _wait_all_idle(timeout: int = 120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/gpus", timeout=5)
            gpus = resp.json().get("gpus", [])
            if all(g["status"] == "idle" for g in gpus):
                return
        except Exception:
            pass
        time.sleep(2)


def print_summary(results: list[dict]):
    print(f"\n{'='*60}")
    print(f"  ПІДСУМОК THROUGHPUT")
    print(f"  {'Level':>8}  {'Submitted':>10}  {'Completed':>10}  {'jobs/s':>8}  {'Race?':>8}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in results:
        race = "!! YES !!" if r["race_detected"] else "OK"
        print(f"  {r['level']:>8}  {r['submitted']:>10}  {r['completed']:>10}  "
              f"{r['throughput']:>8.2f}  {race:>8}")
    print(f"{'='*60}\n")


def save_csv(results: list[dict]):
    path = f"{RESULTS}/concurrency.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "submitted", "completed", "failed", "timeout",
                         "wave_time_s", "throughput_jobs_per_s", "race_detected"])
        for r in results:
            writer.writerow([
                r["level"], r["submitted"], r["completed"], r["failed"], r["timeout"],
                f"{r['wave_time_s']:.2f}", f"{r['throughput']:.3f}", r["race_detected"],
            ])
    print(f"  CSV збережено: {path}")


def save_plot(results: list[dict]):
    if not results:
        return

    levels     = [r["level"]      for r in results]
    throughput = [r["throughput"] for r in results]
    completed  = [r["completed"]  for r in results]
    submitted  = [r["submitted"]  for r in results]
    races      = [r["race_detected"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPU Job Scheduler — Concurrency & Throughput", fontsize=13, fontweight="bold")

    # ── Left: throughput vs concurrency level ─────────────────────────────────
    ax = axes[0]
    bar_colors = ["#e74c3c" if r else "#2ecc71" for r in races]
    bars = ax.bar(range(len(levels)), throughput, color=bar_colors, width=0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(l) for l in levels])
    ax.set_xlabel("Кількість паралельних задач")
    ax.set_ylabel("Пропускна здатність (задач/с)")
    ax.set_title("Throughput vs Concurrency Level")

    for bi, (bar, tp) in enumerate(zip(bars, throughput)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(throughput) * 0.01,
                f"{tp:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    race_patch = plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", label="Race condition виявлено")
    ok_patch   = plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", label="GPU ізоляція OK")
    ax.legend(handles=[ok_patch, race_patch], fontsize=8)

    # ── Right: submitted vs completed stacked ─────────────────────────────────
    ax2 = axes[1]
    x  = np.arange(len(levels))
    ax2.bar(x, completed,                   label="Успішно виконані", color="#2ecc71", width=0.5)
    ax2.bar(x, [s - c for s, c in zip(submitted, completed)],
            bottom=completed,               label="Провалені / timeout", color="#e74c3c", width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in levels])
    ax2.set_xlabel("Кількість паралельних задач")
    ax2.set_ylabel("Кількість задач")
    ax2.set_title("Успішність при різних навантаженнях")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    path = f"{PLOTS}/concurrency.png"
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
