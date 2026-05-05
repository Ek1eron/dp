#!/usr/bin/env python3
"""
sse_vs_polling.py — SSE streaming vs HTTP polling latency comparison.

Для кожного запуску вимірюємо:
  SSE     — час від submit до отримання першого лог-рядка через /logs/stream
  Polling — час від submit до виявлення completed через GET /jobs/{id}
            при інтервалах: 0.5s / 1s / 2s / 5s

Показує що SSE дає ~реальний час, а polling вносить затримку ≈ interval/2.

Output: console + benchmarks/results/sse_vs_polling.csv
                + benchmarks/results/plots/sse_vs_polling.png

Usage:
    python benchmarks/sse_vs_polling.py
    N=20 python benchmarks/sse_vs_polling.py
"""

import csv
import os
import sys
import threading
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

BASE_URL     = os.getenv("BASE_URL", "http://localhost:8001")
N            = int(os.getenv("N", "10"))
POLL_IVLS    = [0.5, 1.0, 2.0, 5.0]   # polling intervals to simulate
TIMEOUT      = 120
RESULTS      = "benchmarks/results"
PLOTS        = f"{RESULTS}/plots"
os.makedirs(PLOTS, exist_ok=True)

# Quick job: prints one line immediately, sleeps briefly so SSE stream can open
QUICK_CODE = 'import time\nprint("log line 1")\ntime.sleep(0.1)\nprint("log line 2")\n'


def submit() -> tuple[str, float]:
    t0   = time.perf_counter()
    resp = requests.post(
        f"{BASE_URL}/submit-job",
        json={"code": QUICK_CODE, "runtime": "pytorch-cu121",
              "student_id": "sse_bench", "name": "sse-bench"},
        timeout=10,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"Submit failed: {resp.status_code}")
    return resp.json()["job_id"], t0


def measure_sse(job_id: str, submit_ts: float) -> float | None:
    """Connect to SSE stream, return seconds until first stdout data line."""
    url      = f"{BASE_URL}/jobs/{job_id}/logs/stream"
    deadline = time.perf_counter() + TIMEOUT
    try:
        with requests.get(url, stream=True, timeout=TIMEOUT) as r:
            if r.status_code != 200:
                return None
            for raw in r.iter_lines(decode_unicode=True):
                if time.perf_counter() > deadline:
                    return None
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if '"end"' in payload:
                    return None  # stream ended before we got a log line
                if '"stdout"' in payload or '"stderr"' in payload:
                    return time.perf_counter() - submit_ts
    except Exception:
        return None
    return None


def measure_polling(job_id: str, submit_ts: float, interval: float) -> float | None:
    """Poll /jobs/{id} at given interval, return seconds until completed detected."""
    deadline = time.perf_counter() + TIMEOUT
    while time.perf_counter() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job  = resp.json()
            if job["status"] in ("completed", "failed", "cancelled"):
                return time.perf_counter() - submit_ts
        except Exception:
            pass
        time.sleep(interval)
    return None


def run_one(run_idx: int) -> dict:
    job_id, t0 = submit()

    # SSE measurement runs in a thread so it can connect right after submit
    sse_result = [None]

    def _sse():
        sse_result[0] = measure_sse(job_id, t0)

    t_sse = threading.Thread(target=_sse, daemon=True)
    t_sse.start()

    # Polling measurements (sequential, each with its own job)
    # We use the same job but simulate different intervals by recording
    # when we WOULD have detected it at each interval
    poll_results: dict[float, float | None] = {}

    completed_at = None
    deadline     = time.perf_counter() + TIMEOUT
    # Poll at finest granularity (0.1s), record time for each threshold
    detected     = {ivl: None for ivl in POLL_IVLS}
    next_check   = {ivl: t0 + ivl for ivl in POLL_IVLS}

    while time.perf_counter() < deadline:
        now = time.perf_counter()
        elapsed = now - t0
        try:
            resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=5)
            job  = resp.json()
            if job["status"] in ("completed", "failed", "cancelled"):
                if completed_at is None:
                    completed_at = now
                # Simulate each polling interval: earliest check after completion
                for ivl in POLL_IVLS:
                    if detected[ivl] is None:
                        # next poll time >= completed_at
                        nxt = next_check[ivl]
                        while nxt < completed_at:
                            nxt += ivl
                        detected[ivl] = nxt - t0
                break
        except Exception:
            pass
        # Advance next_check cursors
        for ivl in POLL_IVLS:
            while next_check[ivl] <= time.perf_counter():
                next_check[ivl] += ivl
        time.sleep(0.1)

    t_sse.join(timeout=TIMEOUT)

    return {
        "run":         run_idx,
        "job_id":      job_id,
        "sse_s":       sse_result[0],
        **{f"poll_{ivl}s": detected[ivl] for ivl in POLL_IVLS},
    }


def run_benchmark() -> list[dict]:
    print(f"\n{'='*65}")
    print(f"  SSE vs Polling Latency Benchmark  ({N} runs)")
    print(f"  Target: {BASE_URL}")
    print(f"  Time  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    header_ivls = "  ".join(f"poll@{i}s" for i in POLL_IVLS)
    print(f"  {'#':<4} {'SSE (s)':>8}  {header_ivls}")
    print(f"  {'-'*4} {'-'*8}  " + "  ".join(["-"*7]*len(POLL_IVLS)))

    rows = []
    for i in range(1, N + 1):
        try:
            row = run_one(i)
            rows.append(row)
            sse_str = f"{row['sse_s']:.3f}" if row["sse_s"] else "  —  "
            poll_strs = "  ".join(
                f"{row[f'poll_{ivl}s']:>7.3f}" if row[f"poll_{ivl}s"] else "    — "
                for ivl in POLL_IVLS
            )
            print(f"  {i:<4} {sse_str:>8}  {poll_strs}")
        except Exception as e:
            print(f"  {i:<4} ERROR: {e}")

    return rows


def print_summary(rows: list[dict]):
    print(f"\n{'='*65}")
    print(f"  ПІДСУМОК — середня затримка виявлення завершення задачі")
    print(f"  {'Метод':<18} {'avg (s)':>8}  {'min (s)':>8}  {'max (s)':>8}")
    print(f"  {'-'*18} {'-'*8}  {'-'*8}  {'-'*8}")

    sse_vals = [r["sse_s"] for r in rows if r.get("sse_s") is not None]
    if sse_vals:
        print(f"  {'SSE streaming':<18} {np.mean(sse_vals):>8.3f}  "
              f"{min(sse_vals):>8.3f}  {max(sse_vals):>8.3f}")

    for ivl in POLL_IVLS:
        vals = [r[f"poll_{ivl}s"] for r in rows if r.get(f"poll_{ivl}s") is not None]
        if vals:
            overhead = np.mean(vals) - (np.mean(sse_vals) if sse_vals else 0)
            print(f"  {f'Polling @ {ivl}s':<18} {np.mean(vals):>8.3f}  "
                  f"{min(vals):>8.3f}  {max(vals):>8.3f}  "
                  f"(+{overhead:.3f}s overhead)")

    print(f"{'='*65}\n")


def save_csv(rows: list[dict]):
    if not rows:
        return
    path = f"{RESULTS}/sse_vs_polling.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV збережено: {path}")


def save_plot(rows: list[dict]):
    if not rows:
        return

    sse_vals   = [r["sse_s"] for r in rows if r.get("sse_s") is not None]
    poll_data  = {}
    for ivl in POLL_IVLS:
        vals = [r[f"poll_{ivl}s"] for r in rows if r.get(f"poll_{ivl}s") is not None]
        if vals:
            poll_data[ivl] = vals

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("GPU Job Scheduler — SSE Streaming vs HTTP Polling",
                 fontsize=13, fontweight="bold")

    # ── Left: latency per run ─────────────────────────────────────────────────
    ax = axes[0]
    runs = list(range(1, len(rows) + 1))

    if sse_vals:
        ax.plot(runs[:len(sse_vals)], sse_vals,
                "o-", color="#2ecc71", linewidth=2, markersize=5, label="SSE streaming")

    colors_ = ["#e74c3c", "#e67e22", "#9b59b6", "#3498db"]
    for (ivl, vals), color in zip(poll_data.items(), colors_):
        ax.plot(runs[:len(vals)], vals,
                "s--", color=color, linewidth=1.5, markersize=4,
                label=f"Polling @ {ivl}s", alpha=0.8)

    ax.set_xlabel("Запуск №")
    ax.set_ylabel("Час виявлення завершення (с)")
    ax.set_title("Затримка виявлення по запусках")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Right: average comparison bar ────────────────────────────────────────
    ax2 = axes[1]
    methods = []
    avgs    = []
    stds    = []
    bar_colors = []

    if sse_vals:
        methods.append("SSE\nstreaming")
        avgs.append(np.mean(sse_vals))
        stds.append(np.std(sse_vals))
        bar_colors.append("#2ecc71")

    for (ivl, vals), color in zip(poll_data.items(), colors_):
        methods.append(f"Polling\n@ {ivl}s")
        avgs.append(np.mean(vals))
        stds.append(np.std(vals))
        bar_colors.append(color)

    x    = np.arange(len(methods))
    bars = ax2.bar(x, avgs, color=bar_colors, width=0.5,
                   yerr=stds, capsize=5, error_kw={"linewidth": 2})
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=8)
    ax2.set_ylabel("Середня затримка (с)")
    ax2.set_title(f"Порівняння методів (N={len(rows)} запусків)")

    for bar, avg in zip(bars, avgs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(stds + [0]) * 0.15,
                 f"{avg:.3f}s", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

    # Highlight SSE advantage
    if len(avgs) >= 2:
        worst = max(avgs)
        best  = avgs[0] if sse_vals else avgs[1]
        ratio = worst / best if best > 0 else 1
        ax2.text(0.97, 0.97,
                 f"Polling @ 5s = {ratio:.0f}× повільніше",
                 transform=ax2.transAxes, ha="right", va="top",
                 fontsize=9, color="darkred", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = f"{PLOTS}/sse_vs_polling.png"
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
