#!/usr/bin/env python3
"""
diagram.py — Architecture diagram generator for GPU Job Scheduler.

Generates two PNG diagrams:
  1. system_architecture.png  — компоненти і потоки даних
  2. job_lifecycle.png        — стани задачі (state machine)

Output: benchmarks/results/plots/system_architecture.png
        benchmarks/results/plots/job_lifecycle.png

Usage:
    python benchmarks/diagram.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PLOTS = "benchmarks/results/plots"
os.makedirs(PLOTS, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel="", color="#3498db", text_color="white",
        fontsize=10, radius=0.04):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.07 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.12, sublabel,
                ha="center", va="center", fontsize=7.5,
                color=text_color, alpha=0.85, zorder=4,
                style="italic")


def arrow(ax, x1, y1, x2, y2, label="", color="#555", bidirectional=False,
          lw=1.5, label_offset=(0, 0.06)):
    style  = "<|-|>" if bidirectional else "-|>"
    arr    = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color, linewidth=lw,
        mutation_scale=14, zorder=2,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)
    if label:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=7.5, color=color, zorder=5,
                bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))


def cloud_box(ax, x, y, w, h, label, color="#ecf0f1", border="#bdc3c7"):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor=border, linewidth=1.2,
        linestyle="--", zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x, y + 0.28, label, ha="center", va="center",
            fontsize=8, color=border, fontweight="bold", zorder=3)


# ── Diagram 1: System Architecture ───────────────────────────────────────────

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.text(5, 5.7, "GPU Job Scheduler — Архітектура системи",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#2c3e50")

    # ── Docker Compose boundary ───────────────────────────────────────────────
    cloud_box(ax, 5, 2.8, 9.2, 5.0, "Docker Compose Network")

    # ── Clients ───────────────────────────────────────────────────────────────
    box(ax, 1.1, 5.1, 1.5, 0.55, "Браузер", "Dashboard / Admin",
        color="#2980b9", fontsize=9)
    box(ax, 1.1, 4.2, 1.5, 0.55, "API Client", "curl / Python SDK",
        color="#2980b9", fontsize=9)

    # ── Master ────────────────────────────────────────────────────────────────
    box(ax, 3.6, 4.7, 2.0, 1.8, "Master", "FastAPI :8001",
        color="#16a085", fontsize=11)
    # Sub-items inside master
    for i, (lbl, sub) in enumerate([
        ("REST API", "/submit-job, /jobs"),
        ("AST Safety", "eval/exec блокування"),
        ("SSE Logs", "/jobs/{id}/logs/stream"),
        ("Admin API", "X-Admin-Key"),
    ]):
        yy = 5.1 - i * 0.38
        ax.text(3.6, yy, f"• {lbl}", ha="center", va="center",
                fontsize=7, color="white", zorder=4)

    # ── Redis ─────────────────────────────────────────────────────────────────
    box(ax, 6.0, 4.7, 1.8, 1.8, "Redis", ":6379",
        color="#c0392b", fontsize=11)
    for i, lbl in enumerate(["job_queue (list)", "job:{id} (hash)", "gpu:0..N (state)", "apikey:{k}"]):
        yy = 5.1 - i * 0.38
        ax.text(6.0, yy, f"• {lbl}", ha="center", va="center",
                fontsize=6.8, color="white", zorder=4)

    # ── Worker ────────────────────────────────────────────────────────────────
    box(ax, 6.0, 2.5, 2.0, 1.8, "Worker", "Threading + Docker",
        color="#8e44ad", fontsize=11)
    for i, lbl in enumerate(["BLPOP job_queue", "WATCH/MULTI/EXEC", "docker run --gpus", "Artifact collect"]):
        yy = 2.9 - i * 0.38
        ax.text(6.0, yy, f"• {lbl}", ha="center", va="center",
                fontsize=6.8, color="white", zorder=4)

    # ── Docker Containers (GPU) ───────────────────────────────────────────────
    box(ax, 8.6, 4.2, 1.6, 0.7, "Container 0", "--gpus device=0",
        color="#d35400", fontsize=8.5)
    box(ax, 8.6, 3.2, 1.6, 0.7, "Container 1", "--gpus device=1",
        color="#d35400", fontsize=8.5)
    ax.text(8.6, 2.55, "...", ha="center", va="center",
            fontsize=14, color="#d35400", zorder=4)

    # ── GPU Hardware ──────────────────────────────────────────────────────────
    box(ax, 8.6, 1.5, 1.6, 0.7, "GPU 0\nNVIDIA", "",
        color="#27ae60", fontsize=8, radius=0.02)
    box(ax, 8.6, 0.7, 1.6, 0.7, "GPU 1\nNVIDIA", "",
        color="#27ae60", fontsize=8, radius=0.02)

    # ── Shared Volume ─────────────────────────────────────────────────────────
    box(ax, 3.6, 2.5, 1.8, 1.2, "Jobs Volume", "/workspace\n(Docker named)",
        color="#7f8c8d", fontsize=8.5)

    # ── Arrows ────────────────────────────────────────────────────────────────
    # Client → Master
    arrow(ax, 1.9, 5.1, 2.55, 4.9, "HTTP/SSE", color="#2980b9", lw=2)
    arrow(ax, 1.9, 4.2, 2.55, 4.5, "REST API", color="#2980b9", lw=2,
          label_offset=(0, -0.12))

    # Master ↔ Redis
    arrow(ax, 4.6, 4.7, 5.1, 4.7, "RPUSH / GET / SET", color="#c0392b",
          bidirectional=True, lw=2, label_offset=(0, 0.1))

    # Redis → Worker
    arrow(ax, 6.0, 3.8, 6.0, 3.4, "BLPOP\njob_queue", color="#8e44ad",
          lw=2, label_offset=(0.4, 0))

    # Worker → Containers
    arrow(ax, 6.95, 2.9, 7.8, 4.0, "docker run", color="#d35400", lw=2,
          label_offset=(0.2, 0.05))
    arrow(ax, 6.95, 2.5, 7.8, 3.1, "docker run", color="#d35400", lw=2,
          label_offset=(0.2, -0.12))

    # Container → GPU
    arrow(ax, 8.6, 3.85, 8.6, 1.85, "--gpus device=N", color="#27ae60",
          lw=1.5, label_offset=(0.55, 0))

    # Worker ↔ Volume
    arrow(ax, 5.0, 2.5, 4.5, 2.5, "Write .py\nRead output", color="#7f8c8d",
          bidirectional=True, lw=1.5, label_offset=(0, 0.12))

    # Master ↔ Volume
    arrow(ax, 3.6, 3.6, 3.6, 3.1, "Artifacts\n/jobs/{id}", color="#7f8c8d",
          lw=1.5, label_offset=(0.5, 0))

    # SSE feedback arrow (curved)
    arr = FancyArrowPatch(
        (2.55, 5.1), (1.9, 5.3),
        arrowstyle="-|>",
        color="#2ecc71", linewidth=2,
        mutation_scale=12, zorder=2,
        connectionstyle="arc3,rad=-0.4",
    )
    ax.add_patch(arr)
    ax.text(2.1, 5.45, "SSE logs", ha="center", fontsize=7.5,
            color="#2ecc71", fontweight="bold")

    # Worker → Redis (update job status)
    arr2 = FancyArrowPatch(
        (6.9, 2.5), (6.9, 3.8),
        arrowstyle="-|>",
        color="#e74c3c", linewidth=1.5,
        mutation_scale=11, zorder=2,
        connectionstyle="arc3,rad=-0.3",
    )
    ax.add_patch(arr2)
    ax.text(7.45, 3.15, "Update status\njob:{id}", ha="center", fontsize=7,
            color="#e74c3c")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color="#16a085", label="Master (FastAPI)"),
        mpatches.Patch(color="#c0392b", label="Redis (черга + стан)"),
        mpatches.Patch(color="#8e44ad", label="Worker (threading)"),
        mpatches.Patch(color="#d35400", label="Docker контейнери"),
        mpatches.Patch(color="#27ae60", label="GPU (NVIDIA CUDA)"),
        mpatches.Patch(color="#7f8c8d", label="Shared volume"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8,
              framealpha=0.9, ncol=3)

    plt.tight_layout(pad=0.5)
    path = f"{PLOTS}/system_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  PNG збережено : {path}")


# ── Diagram 2: Job State Machine ─────────────────────────────────────────────

def draw_job_lifecycle():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    ax.text(7, 4.65, "GPU Job Scheduler — Lifecycle задачі",
            ha="center", fontsize=13, fontweight="bold", color="#2c3e50")

    W, H = 1.8, 0.75

    states = [
        (1.5,  2.5, "Submit",    "Client POST\n/submit-job",    "#2980b9"),
        (3.5,  2.5, "QUEUED",    "Redis RPUSH\njob_queue",      "#f39c12"),
        (5.8,  2.5, "RUNNING",   "GPU зарезервована\ncontainer start", "#8e44ad"),
        (8.3,  3.6, "COMPLETED", "return_code=0\nArtifacts saved",     "#27ae60"),
        (8.3,  2.5, "FAILED",    "return_code≠0\nstderr captured",     "#c0392b"),
        (8.3,  1.4, "CANCELLED", "docker stop\njob видалено з черги",  "#7f8c8d"),
    ]

    for x, y, label, sub, color in states:
        box(ax, x, y, W, H, label, sub, color=color, fontsize=9)

    def arr(x1, y1, x2, y2, lbl="", color="#555", offset=(0, 0.1), rad=0.0):
        style = "arc3,rad=" + str(rad)
        p = FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            color=color, linewidth=1.8, mutation_scale=14,
            zorder=2, connectionstyle=style,
        )
        ax.add_patch(p)
        if lbl:
            mx = (x1 + x2) / 2 + offset[0]
            my = (y1 + y2) / 2 + offset[1]
            ax.text(mx, my, lbl, ha="center", va="bottom",
                    fontsize=7.5, color=color,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

    # Submit → Queued
    arr(2.4, 2.5, 2.6, 2.5, "AST check\n+ rate limit", color="#f39c12",
        offset=(0, 0.15))

    # Queued → Running
    arr(4.4, 2.5, 4.7, 2.5, "BLPOP +\nWATCH/MULTI/EXEC", color="#8e44ad",
        offset=(0, 0.15))

    # Running → Completed
    arr(6.7, 2.85, 7.4, 3.4, "exit 0", color="#27ae60", offset=(0.1, 0.05))

    # Running → Failed
    arr(6.7, 2.5, 7.4, 2.5, "exit ≠0\n/ OOM / timeout", color="#c0392b",
        offset=(0.1, 0.12))

    # Running → Cancelled
    arr(6.7, 2.15, 7.4, 1.6, "POST /cancel\n/ admin kill", color="#7f8c8d",
        offset=(0.2, -0.05))

    # Queued → Cancelled (cancel from queue)
    arr(3.5, 2.12, 5.5, 1.4,
        "cancel з черги", color="#7f8c8d", offset=(0.5, -0.05), rad=0.2)

    # Timeout annotation
    ax.annotate(
        "JOB_TIMEOUT_SECONDS\n(env var)",
        xy=(6.7, 2.5), xytext=(5.8, 1.1),
        fontsize=7.5, color="#c0392b",
        arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1),
        bbox=dict(fc="lightyellow", ec="#c0392b", alpha=0.8, pad=3),
    )

    # Redis WATCH annotation
    ax.annotate(
        "Атомарна GPU резервація\nRedis WATCH/MULTI/EXEC",
        xy=(5.1, 2.85), xytext=(4.8, 4.2),
        fontsize=7.5, color="#8e44ad",
        arrowprops=dict(arrowstyle="->", color="#8e44ad", lw=1),
        bbox=dict(fc="lavender", ec="#8e44ad", alpha=0.8, pad=3),
    )

    # SSE annotation
    ax.annotate(
        "SSE streaming\n/jobs/{id}/logs/stream",
        xy=(7.0, 2.85), xytext=(9.8, 4.2),
        fontsize=7.5, color="#2ecc71",
        arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1),
        bbox=dict(fc="honeydew", ec="#2ecc71", alpha=0.8, pad=3),
    )

    # TTL annotation
    for x, y, *_ in states[3:]:
        ax.text(x, y - 0.55, "TTL 24h", ha="center", fontsize=6.5,
                color="#95a5a6", style="italic")

    # Timeline bar at bottom
    ax.axhline(0.35, xmin=0.08, xmax=0.92, color="#bdc3c7", lw=1.5, zorder=1)
    for xi, lbl in [(1.5, "t=0"), (3.5, "~0s"), (5.8, "queue wait"), (8.3, "execution")]:
        ax.text(xi, 0.15, lbl, ha="center", fontsize=7, color="#7f8c8d")
        ax.plot(xi, 0.35, "o", color="#95a5a6", markersize=5, zorder=2)

    plt.tight_layout(pad=0.3)
    path = f"{PLOTS}/job_lifecycle.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  PNG збережено : {path}")


if __name__ == "__main__":
    print("\n  Generating architecture diagrams...")
    draw_system_architecture()
    draw_job_lifecycle()
    print("  Done.\n")
