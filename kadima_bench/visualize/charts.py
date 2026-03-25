"""
Kadima Bench — Publication-Quality Visualizations v3
=====================================================
7 insight-driven charts. Clean labels, no overlap, clear takeaways.

Layout approach:
  1. tight_layout() with no rect — handles chart internals
  2. subplots_adjust(top=) — carves exact space for titles
  3. suptitle + fig.text — placed in the carved-out header space
  This prevents tight_layout and titles from fighting each other.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np

from kadima_bench.visualize.theme import (
    COLORS, setup_style, add_branding,
    get_family_color, make_legend_handles,
)


# ── Shared layout helper ─────────────────────────────────────────────────────

def _finalize(fig, hardware, path, title, subtitle=None, has_legend_row=False):
    """Apply consistent title/subtitle/branding/layout to any chart.

    Order matters:
      1. tight_layout() — auto-fit chart internals
      2. subplots_adjust(top=) — carve header space
      3. suptitle + fig.text — fill the header space
      4. savefig with bbox_inches='tight' — crop whitespace
    """
    add_branding(fig, hardware)
    plt.tight_layout()

    # Reserve top space: enough for title + subtitle + optional legend row
    top = 0.85 if (subtitle and has_legend_row) else 0.88 if subtitle else 0.92
    fig.subplots_adjust(top=top, bottom=0.08)

    fig.suptitle(title, fontsize=20, fontweight="bold", color=COLORS["text"],
                 y=0.98)
    if subtitle:
        fig.text(0.5, top + 0.01, subtitle,
                 fontsize=10, color=COLORS["text_dim"], ha="center")

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()


def generate_all_charts(results_path: str) -> None:
    """Generate all charts from a results JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = os.path.dirname(results_path)
    hardware = data["metadata"]["hardware"]

    print(f"\n  Generating charts in {output_dir}...")

    chart1_leaderboard(data, hardware, output_dir)
    chart2_efficiency_frontier(data, hardware, output_dir)
    chart3_pass_fail_matrix(data, hardware, output_dir)
    chart4_latency_deep_dive(data, hardware, output_dir)
    chart5_speed_heatmap(data, hardware, output_dir)
    chart6_composite_breakdown(data, hardware, output_dir)
    chart7_energy_and_vram(data, hardware, output_dir)

    print(f"  All 7 charts generated!")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1: LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def chart1_leaderboard(data, hardware, output_dir):
    results = data["results"]
    setup_style()
    n = len(results)

    fig, axes = plt.subplots(1, 3, figsize=(22, max(8, n * 0.8)),
                             gridspec_kw={"width_ratios": [1.3, 0.8, 1.0]})

    labels = [r["label"] for r in results]
    families = [r["family"] for r in results]
    bar_colors = [get_family_color(f) for f in families]
    y_pos = np.arange(n)

    # Col 1: Composite
    composites = [r.get("composite_score", 0) for r in results]
    bars = axes[0].barh(y_pos, composites, color=bar_colors, alpha=0.9, height=0.65)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Composite Score", fontsize=12)
    axes[0].set_title("Composite Score\n(50% Quality + 30% Speed + 20% Efficiency)",
                      fontsize=11, fontweight="bold", pad=10)
    axes[0].set_xlim(0, max(composites) * 1.15)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)
    for i, (bar, val) in enumerate(zip(bars, composites)):
        p = " *" if results[i].get("pareto_optimal") else ""
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}{p}", va="center", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Col 2: Accuracy
    accuracies = [r["accuracy_pct"] for r in results]
    bars2 = axes[1].barh(y_pos, accuracies, color=bar_colors, alpha=0.85, height=0.65)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([""] * n)
    axes[1].set_xlabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Accuracy\n(% Tests Passed)", fontsize=11, fontweight="bold", pad=10)
    axes[1].set_xlim(0, 115)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)
    for bar, acc in zip(bars2, accuracies):
        c = COLORS["accent2"] if acc >= 100 else COLORS["text"]
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.0f}%", va="center", fontsize=11, fontweight="bold", color=c)

    # Col 3: Speed
    speeds = [r["avg_tokens_per_second"] for r in results]
    bars3 = axes[2].barh(y_pos, speeds, color=bar_colors, alpha=0.85, height=0.65)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([""] * n)
    axes[2].set_xlabel("Tokens/Second", fontsize=12)
    axes[2].set_title("Inference Speed\n(tokens/s avg)", fontsize=11, fontweight="bold", pad=10)
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)
    for bar, spd in zip(bars3, speeds):
        axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{spd:.0f}", va="center", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Legend in header area — placed after _finalize adjusts top
    handles = make_legend_handles(families)

    date = data["metadata"]["date"][:10]
    sub = f"{n} Models  |  8 Tests + Streaming Latency  |  GPU-Isolated  |  {date}"

    add_branding(fig, hardware)
    plt.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.08)
    fig.suptitle("Local LLM Benchmark — Overall Leaderboard",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.98)
    fig.text(0.5, 0.89, sub, fontsize=10, color=COLORS["text_dim"], ha="center")
    fig.legend(handles=handles, loc="upper center", ncol=min(len(set(families)), 8),
               bbox_to_anchor=(0.5, 0.87), fontsize=10, framealpha=0.3, edgecolor=COLORS["border"])

    path = os.path.join(output_dir, "kadima_1_leaderboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [1/7] Leaderboard")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2: EFFICIENCY FRONTIER
# ═══════════════════════════════════════════════════════════════════════════════

def chart2_efficiency_frontier(data, hardware, output_dir):
    results = data["results"]
    pareto_labels = set(data.get("pareto_frontier", []))
    setup_style()

    fig, ax = plt.subplots(figsize=(16, 11))

    speeds = [r["avg_tokens_per_second"] for r in results]
    accs = [r["accuracy_pct"] for r in results]

    for r in results:
        is_pareto = r["label"] in pareto_labels
        color = get_family_color(r["family"])
        size_gb = r.get("model_size_gb", 3)
        bubble = max(size_gb * 80, 120)
        marker = "*" if is_pareto else "o"
        edge = COLORS["accent2"] if is_pareto else "white"
        msize = bubble * 2.5 if is_pareto else bubble
        ax.scatter(r["avg_tokens_per_second"], r["accuracy_pct"],
                   s=msize, c=color, marker=marker, alpha=0.9,
                   edgecolors=edge, linewidth=2 if is_pareto else 1.2,
                   zorder=10 if is_pareto else 5)

    _place_labels(ax, results, speeds, accs)

    pareto_points = sorted(
        [(r["avg_tokens_per_second"], r["accuracy_pct"])
         for r in results if r["label"] in pareto_labels],
        key=lambda p: p[0])
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, color=COLORS["accent2"], linewidth=2.5, alpha=0.5,
                linestyle="--", zorder=3)

    ax.set_xlabel("Inference Speed (tokens/second)", fontsize=14, labelpad=10)
    ax.set_ylabel("Accuracy (% tests passed)", fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-15, max(speeds) * 1.15)
    ax.set_ylim(min(accs) - 5, 105)
    ax.text(ax.get_xlim()[1] * 0.95, 103, "IDEAL", fontsize=14,
            color=COLORS["accent2"], alpha=0.5, ha="right", fontweight="bold")

    handles = make_legend_handles([r["family"] for r in results])
    handles.append(mpatches.Patch(color=COLORS["accent2"], alpha=0.5, label="Pareto Frontier"))
    ax.legend(handles=handles, loc="lower right", fontsize=11, framealpha=0.4,
              edgecolor=COLORS["border"])

    path = os.path.join(output_dir, "kadima_2_efficiency_frontier.png")
    _finalize(fig, hardware, path,
              "Efficiency Frontier — Speed vs. Accuracy",
              "Top-right = ideal  |  Stars = Pareto-optimal  |  Bubble size = model size on disk")
    print(f"    [2/7] Efficiency Frontier")


def _place_labels(ax, results, speeds, accs):
    """Place scatter labels with cluster-aware staggering."""
    indexed = sorted(enumerate(results), key=lambda t: t[1]["avg_tokens_per_second"])
    labels_info = [None] * len(results)

    offsets_cycle = [
        (14, 15, "left"), (-18, 15, "left"),
        (14, -15, "right"), (-18, -15, "right"),
        (26, 0, "center"), (-30, 0, "center"),
    ]

    clusters = {}
    for orig_i, r in indexed:
        band = round(r["accuracy_pct"] / 5) * 5
        clusters.setdefault(band, []).append(orig_i)

    for band, members in clusters.items():
        for slot, orig_i in enumerate(members):
            r = results[orig_i]
            oy, ox, ha = offsets_cycle[slot % len(offsets_cycle)]
            if len(members) == 1:
                ox, oy, ha = 15, -8, "left"
            labels_info[orig_i] = {"text": r["label"], "x": r["avg_tokens_per_second"],
                                   "y": r["accuracy_pct"], "ox": ox, "oy": oy, "ha": ha}

    for li in labels_info:
        if li is None:
            continue
        ax.annotate(li["text"], (li["x"], li["y"]),
                    xytext=(li["ox"], li["oy"]), textcoords="offset points",
                    fontsize=9.5, color=COLORS["text"], alpha=0.95,
                    ha=li["ha"], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=COLORS["text_dim"], alpha=0.4, lw=0.8))


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3: PASS/FAIL MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def chart3_pass_fail_matrix(data, hardware, output_dir):
    results = data["results"]
    setup_style()
    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [t["category"] for t in results[0]["test_results"]]
    n_m, n_c = len(models), len(categories)
    matrix = np.array([[1 if tr["passed"] else 0 for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(16, max(7, n_m * 0.75)))

    fail_counts = n_m - matrix.sum(axis=0)
    hardest_idx = np.argmax(fail_counts)
    hardest = categories[hardest_idx] if fail_counts[hardest_idx] > 0 else None
    insight = f"Hardest test: {hardest} ({int(fail_counts[hardest_idx])}/{n_m} failed)" if hardest else "All models passed every test"

    cmap = ListedColormap(["#DC3545", "#28A745"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_c))
    ax.set_yticks(np.arange(n_m))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    for i in range(n_m):
        for j in range(n_c):
            txt = "PASS" if matrix[i, j] == 1 else "FAIL"
            clr = "white" if matrix[i, j] == 0 else "#0D1117"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=clr)

    for i, r in enumerate(results):
        acc = r["accuracy_pct"]
        c = COLORS["accent2"] if acc >= 100 else COLORS["accent4"] if acc >= 80 else COLORS["accent5"]
        ax.text(n_c + 0.4, i, f"{acc:.0f}%", ha="left", va="center", fontsize=12, fontweight="bold", color=c)

    ax.grid(False)
    for i in range(n_m + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2.5)
    for j in range(n_c + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2.5)

    path = os.path.join(output_dir, "kadima_3_pass_fail.png")
    _finalize(fig, hardware, path, "Test Category Breakdown — Pass/Fail Matrix", insight)
    print(f"    [3/7] Pass/Fail Matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4: LATENCY DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════

def chart4_latency_deep_dive(data, hardware, output_dir):
    results = data["results"]
    setup_style()
    models = [r for r in results if r.get("speed_metrics")]
    if not models:
        print(f"    [4/7] Latency (skipped)")
        return

    models.sort(key=lambda r: r["speed_metrics"]["ttft_ms"])
    n = len(models)
    labels = [r["label"] for r in models]
    colors = [get_family_color(r["family"]) for r in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, n * 0.75)),
                                    gridspec_kw={"width_ratios": [1, 1.3]})

    fastest, slowest = models[0], models[-1]
    sub = (f"Fastest: {fastest['label']} ({fastest['speed_metrics']['ttft_ms']:.0f}ms)"
           f"  |  Slowest: {slowest['label']} ({slowest['speed_metrics']['ttft_ms']:.0f}ms)")

    # TTFT bars
    ttfts = [r["speed_metrics"]["ttft_ms"] for r in models]
    y_pos = np.arange(n)
    bars = ax1.barh(y_pos, ttfts, color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Time to First Token (ms)", fontsize=12, labelpad=8)
    ax1.set_title("TTFT (lower is better)", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, ttfts):
        ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}ms", va="center", fontsize=10, fontweight="bold", color=COLORS["text"])

    # ITL percentiles
    p50s = [r["speed_metrics"]["itl_p50_ms"] for r in models]
    p95s = [r["speed_metrics"]["itl_p95_ms"] for r in models]
    p99s = [r["speed_metrics"]["itl_p99_ms"] for r in models]
    bh = 0.2
    ax2.barh(y_pos + bh, p50s, bh, color=COLORS["accent2"], alpha=0.9, label="p50 (median)")
    ax2.barh(y_pos, p95s, bh, color=COLORS["accent4"], alpha=0.9, label="p95")
    ax2.barh(y_pos - bh, p99s, bh, color=COLORS["accent5"], alpha=0.85, label="p99 (worst)")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Inter-Token Latency (ms)", fontsize=12, labelpad=8)
    ax2.set_title("ITL Percentiles (lower = smoother)", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(fontsize=10, framealpha=0.4, loc="lower right")
    for i, val in enumerate(p50s):
        ax2.text(p99s[i] + 1, i, f"p50={val:.1f}ms", va="center", fontsize=9, color=COLORS["text_dim"])

    path = os.path.join(output_dir, "kadima_4_latency.png")
    _finalize(fig, hardware, path, "Latency Deep Dive — TTFT & Inter-Token Latency", sub)
    print(f"    [4/7] Latency Deep Dive")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5: SPEED HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def chart5_speed_heatmap(data, hardware, output_dir):
    results = data["results"]
    setup_style()
    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [tr["category"] for tr in results[0]["test_results"]]
    n_m, n_c = len(models), len(categories)
    matrix = np.array([[tr.get("tokens_per_second", 0) for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(18, max(8, n_m * 0.8)))

    avg_speeds = [r["avg_tokens_per_second"] for r in results]
    fi = np.argmax(avg_speeds)
    sub = (f"Fastest: {results[fi]['label']} ({avg_speeds[fi]:.0f} t/s avg)"
           f"  |  Values = tokens/second  |  Darker green = faster")

    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(n_c))
    ax.set_yticks(np.arange(n_m))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    for i in range(n_m):
        for j in range(n_c):
            v = matrix[i, j]
            c = "#0D1117" if v > np.percentile(matrix, 60) else COLORS["text"]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10, fontweight="bold", color=c)

    for i, r in enumerate(results):
        avg = r["avg_tokens_per_second"]
        c = COLORS["accent2"] if avg >= 150 else COLORS["accent"] if avg >= 80 else COLORS["accent5"]
        ax.text(n_c + 0.4, i, f"avg {avg:.0f} t/s", ha="left", va="center",
                fontsize=11, fontweight="bold", color=c)

    ax.grid(False)
    for i in range(n_m + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2.5)
    for j in range(n_c + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.18)
    cbar.set_label("Tokens/second", color=COLORS["text"], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_dim"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLORS["text_dim"])

    path = os.path.join(output_dir, "kadima_5_speed_heatmap.png")
    _finalize(fig, hardware, path, "Inference Speed by Test Category", sub)
    print(f"    [5/7] Speed Heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6: COMPOSITE BREAKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def chart6_composite_breakdown(data, hardware, output_dir):
    results = data["results"]
    w = data["metadata"].get("scoring_weights", {"quality": 0.5, "speed": 0.3, "efficiency": 0.2})
    setup_style()
    n = len(results)

    fig, ax = plt.subplots(figsize=(18, max(8, n * 0.75)))

    sub = (f"Scoring: Quality {w.get('quality', 0.5):.0%}"
           f" + Speed {w.get('speed', 0.3):.0%}"
           f" + Efficiency {w.get('efficiency', 0.2):.0%}"
           f"  |  * = Pareto-optimal")

    labels = [r["label"] for r in results]
    max_spd = max(r.get("avg_tokens_per_second", 0) for r in results)
    max_eff = max(r.get("efficiency_tps_per_gb", 0) for r in results)
    wq, ws, we = w.get("quality", 0.5), w.get("speed", 0.3), w.get("efficiency", 0.2)

    qs = [r.get("accuracy_pct", 0) * wq for r in results]
    ss = [(r.get("avg_tokens_per_second", 0) / max_spd * 100 * ws) if max_spd > 0 else 0 for r in results]
    es = [(r.get("efficiency_tps_per_gb", 0) / max_eff * 100 * we) if max_eff > 0 else 0 for r in results]
    y_pos = np.arange(n)

    ax.barh(y_pos, qs, color=COLORS["accent2"], alpha=0.9, height=0.6, label="Quality (accuracy)")
    ax.barh(y_pos, ss, left=qs, color=COLORS["accent"], alpha=0.9, height=0.6, label="Speed (throughput)")
    lefts = [q + s for q, s in zip(qs, ss)]
    ax.barh(y_pos, es, left=lefts, color=COLORS["accent3"], alpha=0.9, height=0.6, label="Efficiency (t/s per GB)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Composite Score", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.4)

    for i, r in enumerate(results):
        total = r.get("composite_score", 0)
        p = " *" if r.get("pareto_optimal") else ""
        ax.text(total + 0.3, i, f"{total:.1f}{p}", va="center", fontsize=11,
                fontweight="bold", color=COLORS["text"])

    path = os.path.join(output_dir, "kadima_6_composite.png")
    _finalize(fig, hardware, path, "Composite Score Breakdown — What Drives Each Ranking", sub)
    print(f"    [6/7] Composite Breakdown")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7: ENERGY & VRAM
# ═══════════════════════════════════════════════════════════════════════════════

def chart7_energy_and_vram(data, hardware, output_dir):
    results = data["results"]
    setup_style()
    mg = [r for r in results if r.get("gpu_snapshot")]
    if not mg:
        print(f"    [7/7] Energy & VRAM (skipped)")
        return

    mg.sort(key=lambda r: r["gpu_snapshot"]["peak_vram_mb"])
    n = len(mg)
    labels = [r["label"] for r in mg]
    colors = [get_family_color(r["family"]) for r in mg]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(9, n * 0.85)),
                                    gridspec_kw={"width_ratios": [1, 1]})

    best = min(mg, key=lambda r: r["gpu_snapshot"]["avg_power_w"] / max(r.get("avg_tokens_per_second", 1), 1))
    wpv = best["gpu_snapshot"]["avg_power_w"] / max(best.get("avg_tokens_per_second", 1), 1)
    sub = f"Most power-efficient: {best['label']} ({wpv:.2f} W per t/s)  |  16GB VRAM budget shown"

    # VRAM bars
    y_pos = np.arange(n)
    vrams = [r["gpu_snapshot"]["peak_vram_mb"] / 1024 for r in mg]
    bars1 = ax1.barh(y_pos, vrams, color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Peak VRAM (GB)", fontsize=12, labelpad=8)
    ax1.set_title("Peak VRAM Usage (lower = more headroom)", fontsize=11, fontweight="bold", pad=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.axvline(x=16, color=COLORS["accent5"], linestyle="--", linewidth=2, alpha=0.6)
    ax1.text(16.1, -0.5, "16GB", fontsize=9, color=COLORS["accent5"], alpha=0.7, va="top")
    for bar, val in zip(bars1, vrams):
        c = COLORS["accent5"] if val > 14 else COLORS["text"]
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}GB", va="center", fontsize=10, fontweight="bold", color=c)

    # Power vs throughput scatter
    powers = [r["gpu_snapshot"]["avg_power_w"] for r in mg]
    tputs = [r.get("avg_tokens_per_second", 0) for r in mg]

    for i, r in enumerate(mg):
        sz = max(r.get("model_size_gb", 2) * 80, 100)
        ax2.scatter(powers[i], tputs[i], s=sz, c=colors[i], alpha=0.85,
                    edgecolors="white", linewidth=1.2, zorder=5)

    placed = []
    for i, r in enumerate(mg):
        cands = [(10, 8, "left"), (-10, 8, "right"), (10, -12, "left"),
                 (-10, -12, "right"), (0, 16, "center"), (0, -18, "center")]
        bx, by, bh = cands[0]
        pw_range = max(max(powers) - min(powers), 1)
        tp_range = max(max(tputs) - min(tputs), 1)
        for ox, oy, ha in cands:
            hit = False
            for px, py, pox, poy in placed:
                dx = (powers[i] - px) / pw_range * 100
                dy = (tputs[i] - py) / tp_range * 100
                if abs(dx + ox - pox) < 12 and abs(dy + oy - poy) < 12:
                    hit = True
                    break
            if not hit:
                bx, by, bh = ox, oy, ha
                break
        placed.append((powers[i], tputs[i], bx, by))
        ax2.annotate(r["label"], (powers[i], tputs[i]),
                     xytext=(bx, by), textcoords="offset points",
                     fontsize=8.5, color=COLORS["text"], fontweight="bold", ha=bh,
                     arrowprops=dict(arrowstyle="-", color=COLORS["text_dim"], alpha=0.3, lw=0.5))

    ax2.set_xlabel("Average Power (watts)", fontsize=12, labelpad=8)
    ax2.set_ylabel("Throughput (tokens/s)", fontsize=12, labelpad=8)
    ax2.set_title("Power vs. Throughput (top-left = efficient)", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(True, alpha=0.2)

    path = os.path.join(output_dir, "kadima_7_energy_vram.png")
    _finalize(fig, hardware, path, "GPU Resource Usage — VRAM & Power Consumption", sub)
    print(f"    [7/7] Energy & VRAM")
