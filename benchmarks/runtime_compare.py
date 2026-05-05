#!/usr/bin/env python3
"""
runtime_compare.py — Compare execution time across all runtime profiles.

Один і той самий GPU-обчислювальний код запускається в трьох контейнерах:
  - pytorch-cu121  (PyTorch 2.2.2 + CUDA 12.1)
  - pytorch-cu118  (PyTorch 2.1.2 + CUDA 11.8)
  - tensorflow     (TensorFlow 2.15.0 + GPU)

Для PyTorch: torch.randn(4000,4000) @ torch.randn(4000,4000)
Для TF     : tf.random.normal + tf.linalg.matmul

Output: console + benchmarks/results/runtime_compare.csv
                + benchmarks/results/plots/runtime_compare.png

Usage:
    python benchmarks/runtime_compare.py         # 3 runs per runtime (default)
    RUNS=5 python benchmarks/runtime_compare.py
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

BASE_URL = os.getenv("BASE_URL", "http://localhost:8001")
RUNS     = int(os.getenv("RUNS", "3"))
TIMEOUT  = 240   # TF image pull can be slow
RESULTS  = "benchmarks/results"
PLOTS    = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

CODES = {
    "pytorch-cu121": """\
import torch, time
device = 'cuda'
N = 4000
A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
C = A @ B
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"RESULT_MS={elapsed*1000:.1f}")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}")
""",
    "pytorch-cu118": """\
import torch, time
device = 'cuda'
N = 4000
A = torch.randn(N, N, device=device)
B = torch.randn(N, N, device=device)
torch.cuda.synchronize()
t0 = time.perf_counter()
C = A @ B
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"RESULT_MS={elapsed*1000:.1f}")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__}")
""",
    "tensorflow": """\
import tensorflow as tf, time
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {[g.name for g in gpus]}")
N = 4000
with tf.device('/GPU:0'):
    A = tf.random.normal((N, N))
    B = tf.random.normal((N, N))
    # warm-up
    _ = tf.linalg.matmul(A, B)
    t0 = time.perf_counter()
    C = tf.linalg.matmul(A, B)
    _ = C.numpy()  # sync
    elapsed = time.perf_counter() - t0
print(f"RESULT_MS={elapsed*1000:.1f}")
print(f"tf={tf.__version__}")
""",
}

RUNTIME_LABELS = {
    "pytorch-cu121": "PyTorch 2.2.2\n(CUDA 12.1)",
    "pytorch-cu118": "PyTorch 2.1.2\n(CUDA 11.8)",
    "tensorflow":    "TensorFlow 2.15\n(GPU)",
}


def submit_and_wait(runtime: str, run_idx: int) -> dict:
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={
            "code":       CODES[runtime],
            "runtime":    runtime,
            "student_id": f"rtcmp_{runtime.replace('-','_')}",
            "name":       f"rtcmp-{runtime}-{run_idx}",
        },
        timeout=10,
    )
    if resp.status_code != 201:
        return {"runtime": runtime, "run": run_idx, "status": "submit_failed",
                "compute_ms": None, "total_s": None}

    job_id  = resp.json()["job_id"]
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            r   = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job = r.json()
            if job["status"] in ("completed", "failed", "cancelled"):
                total_s    = time.time() - t0
                compute_ms = None
                if job["status"] == "completed":
                    for line in job.get("stdout", "").splitlines():
                        if line.startswith("RESULT_MS="):
                            try:
                                compute_ms = float(line.split("=")[1].strip())
                            except Exception:
                                pass
                return {
                    "runtime":    runtime,
                    "run":        run_idx,
                    "status":     job["status"],
                    "compute_ms": compute_ms,
                    "total_s":    total_s,
                    "stdout":     job.get("stdout", "")[:200],
                }
        except Exception:
            pass
        time.sleep(1)

    return {"runtime": runtime, "run": run_idx, "status": "timeout",
            "compute_ms": None, "total_s": time.time() - t0}


def _wait_all_idle(timeout: int = 60):
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


def run_benchmark() -> list[dict]:
    print(f"\n{'='*65}")
    print(f"  Runtime Comparison Benchmark")
    print(f"  Target: {BASE_URL}  |  {RUNS} runs per runtime")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"  Обчислення: matmul(4000×4000) на GPU\n")
    print(f"  {'Runtime':<16} {'Run':>4} {'compute (ms)':>14} {'total (s)':>10}  status")
    print(f"  {'-'*16} {'-'*4} {'-'*14} {'-'*10}  {'------'}")

    all_rows = []
    for runtime in CODES:
        for run_idx in range(1, RUNS + 1):
            _wait_all_idle()
            row = submit_and_wait(runtime, run_idx)
            all_rows.append(row)
            c_ms = f"{row['compute_ms']:.1f}" if row["compute_ms"] else "—"
            t_s  = f"{row['total_s']:.1f}"    if row["total_s"]    else "—"
            icon = "✓" if row["status"] == "completed" else "✗"
            print(f"  {runtime:<16} {run_idx:>4} {c_ms:>14} {t_s:>10}  {icon} {row['status']}")

    return all_rows


def print_summary(rows: list[dict]):
    print(f"\n{'='*65}")
    print(f"  ПІДСУМОК (compute time, мс)")
    print(f"  {'Runtime':<16} {'avg':>8} {'min':>8} {'max':>8}  N")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}  -")
    for runtime in CODES:
        vals = [r["compute_ms"] for r in rows
                if r["runtime"] == runtime and r["compute_ms"] is not None]
        if vals:
            print(f"  {runtime:<16} {np.mean(vals):>8.1f} {min(vals):>8.1f} {max(vals):>8.1f}  {len(vals)}")
        else:
            print(f"  {runtime:<16} {'—':>8} {'—':>8} {'—':>8}  0")
    print(f"{'='*65}\n")


def save_csv(rows: list[dict]):
    path = f"{RESULTS}/runtime_compare.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["runtime", "run", "status", "compute_ms", "total_s"])
        for r in rows:
            writer.writerow([r["runtime"], r["run"], r["status"],
                             r.get("compute_ms", ""), r.get("total_s", "")])
    print(f"  CSV збережено: {path}")


def save_plot(rows: list[dict]):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("GPU Job Scheduler — Runtime Profile Comparison (matmul 4000×4000)",
                 fontsize=12, fontweight="bold")

    runtimes = list(CODES.keys())
    colors   = ["#3498db", "#9b59b6", "#e67e22"]

    # ── Left: compute time per run (grouped bar) ──────────────────────────────
    ax = axes[0]
    x  = np.arange(RUNS)
    w  = 0.25
    for ci, (rt, color) in enumerate(zip(runtimes, colors)):
        vals = [r["compute_ms"] for r in rows
                if r["runtime"] == rt and r["compute_ms"] is not None]
        # pad/trim to RUNS
        vals = (vals + [None] * RUNS)[:RUNS]
        vals_plot = [v if v is not None else 0 for v in vals]
        ax.bar(x + ci * w, vals_plot, w * 0.9, label=RUNTIME_LABELS[rt], color=color)

    ax.set_xticks(x + w)
    ax.set_xticklabels([f"Run {i+1}" for i in range(RUNS)])
    ax.set_ylabel("Час обчислення (мс)")
    ax.set_title("GPU матричне множення — по запусках")
    ax.legend(fontsize=8)

    # ── Right: average comparison bar ────────────────────────────────────────
    ax2 = axes[1]
    avg_vals = []
    std_vals = []
    for rt in runtimes:
        vals = [r["compute_ms"] for r in rows
                if r["runtime"] == rt and r["compute_ms"] is not None]
        avg_vals.append(np.mean(vals) if vals else 0)
        std_vals.append(np.std(vals)  if len(vals) > 1 else 0)

    bars = ax2.bar(range(len(runtimes)), avg_vals, color=colors, width=0.5,
                   yerr=std_vals, capsize=6, error_kw={"linewidth": 2})
    ax2.set_xticks(range(len(runtimes)))
    ax2.set_xticklabels([RUNTIME_LABELS[rt] for rt in runtimes], fontsize=8)
    ax2.set_ylabel("Середній час обчислення (мс)")
    ax2.set_title(f"Середнє ± std (N={RUNS} запусків)")

    best_idx = int(np.argmin([v if v > 0 else float("inf") for v in avg_vals]))
    for bi, (bar, avg, std) in enumerate(zip(bars, avg_vals, std_vals)):
        label = f"{avg:.0f}ms"
        if bi == best_idx:
            label += " ★"
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + std + max(avg_vals) * 0.01,
                 label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = f"{PLOTS}/runtime_compare.png"
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
