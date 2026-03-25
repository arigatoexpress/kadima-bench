"""
Kadima Bench — Publication-Quality Visualizations v2
=====================================================
7 insight-driven charts. Clean labels, no overlap, clear takeaways.

Chart 1: Leaderboard — Overall ranking (composite, accuracy, speed)
Chart 2: Efficiency Frontier — Speed vs Accuracy scatter with Pareto line
Chart 3: Pass/Fail Matrix — Which models fail which tasks
Chart 4: Latency Deep Dive — TTFT + ITL percentiles side-by-side
Chart 5: Speed Heatmap — Per-model per-category tokens/s
Chart 6: Composite Breakdown — Stacked quality + speed + efficiency
Chart 7: Energy & VRAM — Power efficiency and memory footprint
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
# Chart 1: LEADERBOARD — The one chart that tells the whole story
# ═══════════════════════════════════════════════════════════════════════════════

def chart1_leaderboard(data, hardware, output_dir):
    """Three-column leaderboard: composite score, accuracy, speed."""
    results = data["results"]
    setup_style()

    n = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(22, max(8, n * 0.8)),
                             gridspec_kw={"width_ratios": [1.3, 0.8, 1.0]})

    date = data["metadata"]["date"][:10]
    fig.suptitle("Local LLM Benchmark — Overall Leaderboard",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.0)
    fig.text(0.5, 0.92,
             f"{n} Models  |  8 Tests + Streaming Latency  |  GPU-Isolated  |  {date}",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    labels = [r["label"] for r in results]
    families = [r["family"] for r in results]
    bar_colors = [get_family_color(f) for f in families]
    y_pos = np.arange(n)

    # Col 1: Composite Score (sorted, this is the primary ranking)
    composites = [r.get("composite_score", 0) for r in results]
    bars = axes[0].barh(y_pos, composites, color=bar_colors, alpha=0.9, height=0.65)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels, fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Composite Score", fontsize=12)
    axes[0].set_title("Composite Score\n(50% Quality + 30% Speed + 20% Efficiency)",
                      fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlim(0, max(composites) * 1.15)
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, composites)):
        pareto = " *" if results[i].get("pareto_optimal") else ""
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}{pareto}", va="center", fontsize=11,
                     fontweight="bold", color=COLORS["text"])

    # Col 2: Accuracy
    accuracies = [r["accuracy_pct"] for r in results]
    bars2 = axes[1].barh(y_pos, accuracies, color=bar_colors, alpha=0.85, height=0.65)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([""] * n)  # no duplicate labels
    axes[1].set_xlabel("Accuracy (%)", fontsize=12)
    axes[1].set_title("Accuracy\n(% Tests Passed)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlim(0, 115)
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    for bar, acc in zip(bars2, accuracies):
        color = COLORS["accent2"] if acc >= 100 else COLORS["text"]
        axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.0f}%", va="center", fontsize=11, fontweight="bold", color=color)

    # Col 3: Speed
    speeds = [r["avg_tokens_per_second"] for r in results]
    bars3 = axes[2].barh(y_pos, speeds, color=bar_colors, alpha=0.85, height=0.65)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels([""] * n)
    axes[2].set_xlabel("Tokens/Second", fontsize=12)
    axes[2].set_title("Inference Speed\n(tokens/s avg)", fontsize=12, fontweight="bold", pad=12)
    axes[2].invert_yaxis()
    axes[2].grid(axis="x", alpha=0.3)

    for bar, spd in zip(bars3, speeds):
        axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                     f"{spd:.0f}", va="center", fontsize=11, fontweight="bold", color=COLORS["text"])

    # Legend
    handles = make_legend_handles(families)
    fig.legend(handles=handles, loc="upper center", ncol=min(len(set(families)), 8),
               bbox_to_anchor=(0.5, 0.94), fontsize=10, framealpha=0.3, edgecolor=COLORS["border"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_1_leaderboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [1/7] Leaderboard")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2: EFFICIENCY FRONTIER — Speed vs Accuracy with Pareto line
# ═══════════════════════════════════════════════════════════════════════════════

def chart2_efficiency_frontier(data, hardware, output_dir):
    """Scatter: speed vs accuracy, Pareto frontier highlighted, insight callouts."""
    results = data["results"]
    pareto_labels = set(data.get("pareto_frontier", []))
    setup_style()

    fig, ax = plt.subplots(figsize=(16, 11))
    fig.suptitle("Efficiency Frontier — Speed vs. Accuracy",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.0)
    fig.text(0.5, 0.92,
             "Top-right = ideal  |  Stars = Pareto-optimal  |  Bubble size = model size on disk",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    speeds = [r["avg_tokens_per_second"] for r in results]
    accs = [r["accuracy_pct"] for r in results]

    # Plot all models
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
                   edgecolors=edge, linewidth=2 if is_pareto else 1.2, zorder=10 if is_pareto else 5)

    # Smart label placement with repulsion
    _place_labels(ax, results, speeds, accs)

    # Draw Pareto frontier line
    pareto_points = sorted(
        [(r["avg_tokens_per_second"], r["accuracy_pct"])
         for r in results if r["label"] in pareto_labels],
        key=lambda p: p[0],
    )
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, color=COLORS["accent2"], linewidth=2.5, alpha=0.5,
                linestyle="--", zorder=3, label="Pareto Frontier")

    ax.set_xlabel("Inference Speed (tokens/second)", fontsize=14, labelpad=10)
    ax.set_ylabel("Accuracy (% tests passed)", fontsize=14, labelpad=10)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(-15, max(speeds) * 1.15)
    ax.set_ylim(min(accs) - 5, 105)

    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(xlim[1] * 0.95, 103, "IDEAL", fontsize=14,
            color=COLORS["accent2"], alpha=0.5, ha="right", fontweight="bold")

    # Legend
    handles = make_legend_handles([r["family"] for r in results])
    handles.append(mpatches.Patch(color=COLORS["accent2"], alpha=0.5, label="Pareto Frontier"))
    ax.legend(handles=handles, loc="lower right", fontsize=11, framealpha=0.4,
              edgecolor=COLORS["border"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    path = os.path.join(output_dir, "kadima_2_efficiency_frontier.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [2/7] Efficiency Frontier")


def _place_labels(ax, results, speeds, accs):
    """Place labels without overlap using multi-pass repulsion."""
    x_range = max(speeds) - min(speeds) if len(set(speeds)) > 1 else 100
    y_range = max(accs) - min(accs) if len(set(accs)) > 1 else 30

    # Sort by x position so we can stagger left-to-right within clusters
    indexed = sorted(enumerate(results), key=lambda t: t[1]["avg_tokens_per_second"])

    labels_info = [None] * len(results)

    # Offset options: (oy_pixels, ox_pixels, ha)
    # Cycle through these for models in the same accuracy band
    offsets_cycle = [
        (14, 15, "left"),     # above-right
        (-18, 15, "left"),    # below-right
        (14, -15, "right"),   # above-left
        (-18, -15, "right"),  # below-left
        (26, 0, "center"),    # far above
        (-30, 0, "center"),   # far below
    ]

    # Group by accuracy band (within 5% = same cluster)
    clusters = {}
    for orig_i, r in indexed:
        band = round(r["accuracy_pct"] / 5) * 5
        clusters.setdefault(band, []).append(orig_i)

    for band, members in clusters.items():
        for slot, orig_i in enumerate(members):
            r = results[orig_i]
            oy, ox, ha = offsets_cycle[slot % len(offsets_cycle)]
            # Singletons just go to the right
            if len(members) == 1:
                ox, oy, ha = 15, -8, "left"
            labels_info[orig_i] = {
                "text": r["label"],
                "x": r["avg_tokens_per_second"],
                "y": r["accuracy_pct"],
                "ox": ox, "oy": oy, "ha": ha,
            }

    # Render with connector lines
    for li in labels_info:
        if li is None:
            continue
        ax.annotate(
            li["text"], (li["x"], li["y"]),
            xytext=(li["ox"], li["oy"]), textcoords="offset points",
            fontsize=9.5, color=COLORS["text"], alpha=0.95,
            ha=li["ha"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLORS["text_dim"], alpha=0.4, lw=0.8),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3: PASS/FAIL MATRIX — Where each model breaks
# ═══════════════════════════════════════════════════════════════════════════════

def chart3_pass_fail_matrix(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [t["category"] for t in results[0]["test_results"]]
    n_models = len(models)
    n_cats = len(categories)

    matrix = np.array([[1 if tr["passed"] else 0 for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(16, max(7, n_models * 0.75)))
    fig.suptitle("Test Category Breakdown — Pass/Fail Matrix",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.0)

    # Count failures per category for subtitle insight
    fail_counts = n_models - matrix.sum(axis=0)
    hardest_idx = np.argmax(fail_counts)
    hardest = categories[hardest_idx] if fail_counts[hardest_idx] > 0 else None
    insight = f"Hardest test: {hardest} ({int(fail_counts[hardest_idx])}/{n_models} models failed)" if hardest else "All models passed every test"
    fig.text(0.5, 0.92, insight,
             fontsize=11, color=COLORS["accent4"], ha="center", fontweight="bold")

    cmap = ListedColormap(["#DC3545", "#28A745"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    for i in range(n_models):
        for j in range(n_cats):
            text = "PASS" if matrix[i, j] == 1 else "FAIL"
            color = "white" if matrix[i, j] == 0 else "#0D1117"
            weight = "bold"
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    fontweight=weight, color=color)

    # Accuracy column on right
    for i, r in enumerate(results):
        acc = r["accuracy_pct"]
        color = COLORS["accent2"] if acc >= 100 else COLORS["accent4"] if acc >= 80 else COLORS["accent5"]
        ax.text(n_cats + 0.4, i, f"{acc:.0f}%",
                ha="left", va="center", fontsize=12, fontweight="bold", color=color)

    ax.grid(False)
    for i in range(n_models + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2.5)
    for j in range(n_cats + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2.5)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    path = os.path.join(output_dir, "kadima_3_pass_fail.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [3/7] Pass/Fail Matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4: LATENCY DEEP DIVE — TTFT bars + ITL percentile grouped bars
# ═══════════════════════════════════════════════════════════════════════════════

def chart4_latency_deep_dive(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    models_with_speed = [r for r in results if r.get("speed_metrics")]
    if not models_with_speed:
        print(f"    [4/7] Latency (skipped: no speed metrics)")
        return

    # Sort by TTFT for readability
    models_with_speed.sort(key=lambda r: r["speed_metrics"]["ttft_ms"])

    n = len(models_with_speed)
    labels = [r["label"] for r in models_with_speed]
    colors = [get_family_color(r["family"]) for r in models_with_speed]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(8, n * 0.75)),
                                    gridspec_kw={"width_ratios": [1, 1.3]})
    fig.suptitle("Latency Deep Dive — Time to First Token & Inter-Token Latency",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=1.0)

    # Find the fastest TTFT for insight callout
    fastest = models_with_speed[0]
    slowest = models_with_speed[-1]
    fig.text(0.5, 0.92,
             f"Fastest first response: {fastest['label']} ({fastest['speed_metrics']['ttft_ms']:.0f}ms)"
             f"  |  Slowest: {slowest['label']} ({slowest['speed_metrics']['ttft_ms']:.0f}ms)",
             fontsize=10, color=COLORS["text_dim"], ha="center")

    # LEFT: TTFT horizontal bars (sorted ascending = fastest at top)
    ttfts = [r["speed_metrics"]["ttft_ms"] for r in models_with_speed]
    y_pos = np.arange(n)
    bars = ax1.barh(y_pos, ttfts, color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Time to First Token (ms)", fontsize=12, labelpad=8)
    ax1.set_title("TTFT — Time to First Token\n(lower is better)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, ttfts):
        ax1.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}ms", va="center", fontsize=10, fontweight="bold", color=COLORS["text"])

    # RIGHT: ITL percentiles as grouped horizontal bars (sorted same order)
    p50s = [r["speed_metrics"]["itl_p50_ms"] for r in models_with_speed]
    p95s = [r["speed_metrics"]["itl_p95_ms"] for r in models_with_speed]
    p99s = [r["speed_metrics"]["itl_p99_ms"] for r in models_with_speed]

    bar_h = 0.2
    ax2.barh(y_pos + bar_h, p50s, bar_h, color=COLORS["accent2"], alpha=0.9, label="p50 (median)")
    ax2.barh(y_pos, p95s, bar_h, color=COLORS["accent4"], alpha=0.9, label="p95")
    ax2.barh(y_pos - bar_h, p99s, bar_h, color=COLORS["accent5"], alpha=0.85, label="p99 (worst case)")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("Inter-Token Latency (ms)", fontsize=12, labelpad=8)
    ax2.set_title("ITL Percentiles — Per-Token Consistency\n(lower = smoother streaming)", fontsize=13, fontweight="bold", pad=12)
    ax2.grid(axis="x", alpha=0.3)
    ax2.legend(fontsize=11, framealpha=0.4, loc="lower right")

    # Add p50 value labels
    for i, val in enumerate(p50s):
        ax2.text(p99s[i] + 1, i, f"p50={val:.1f}ms",
                 va="center", fontsize=9, color=COLORS["text_dim"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    path = os.path.join(output_dir, "kadima_4_latency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [4/7] Latency Deep Dive")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5: SPEED HEATMAP — tokens/s per model per category
# ═══════════════════════════════════════════════════════════════════════════════

def chart5_speed_heatmap(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [tr["category"] for tr in results[0]["test_results"]]
    n_models = len(models)
    n_cats = len(categories)

    matrix = np.array([[tr.get("tokens_per_second", 0)
                         for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(18, max(9, n_models * 0.85)))
    fig.suptitle("Inference Speed by Test Category",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.02)

    # Insight: find the speed champion
    avg_speeds = [r["avg_tokens_per_second"] for r in results]
    fastest_idx = np.argmax(avg_speeds)
    fig.text(0.5, 0.94,
             f"Fastest overall: {results[fastest_idx]['label']} ({avg_speeds[fastest_idx]:.0f} t/s avg)"
             f"  |  Values in tokens/second  |  Darker green = faster",
             fontsize=10, color=COLORS["text_dim"], ha="center")

    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(n_cats))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(categories, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(models, fontsize=11)

    # Cell annotations — larger font, better contrast
    for i in range(n_models):
        for j in range(n_cats):
            val = matrix[i, j]
            # Dark text on bright cells, light text on dark cells
            color = "#0D1117" if val > np.percentile(matrix, 60) else COLORS["text"]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    # Avg speed column
    for i, r in enumerate(results):
        avg = r["avg_tokens_per_second"]
        color = COLORS["accent2"] if avg >= 150 else COLORS["accent"] if avg >= 80 else COLORS["accent5"]
        ax.text(n_cats + 0.4, i, f"avg {avg:.0f} t/s",
                ha="left", va="center", fontsize=11, fontweight="bold", color=color)

    ax.grid(False)
    for i in range(n_models + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2.5)
    for j in range(n_cats + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2.5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.18)
    cbar.set_label("Tokens/second", color=COLORS["text"], fontsize=11)
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_dim"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLORS["text_dim"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    path = os.path.join(output_dir, "kadima_5_speed_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [5/7] Speed Heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6: COMPOSITE BREAKDOWN — Stacked bars showing what drives each score
# ═══════════════════════════════════════════════════════════════════════════════

def chart6_composite_breakdown(data, hardware, output_dir):
    results = data["results"]
    weights = data["metadata"].get("scoring_weights", {"quality": 0.5, "speed": 0.3, "efficiency": 0.2})
    setup_style()

    n = len(results)
    fig, ax = plt.subplots(figsize=(18, max(8, n * 0.75)))
    fig.suptitle("Composite Score Breakdown — What Drives Each Model's Ranking",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.0)
    fig.text(0.5, 0.92,
             f"Scoring: Quality {weights.get('quality', 0.5):.0%}"
             f" + Speed {weights.get('speed', 0.3):.0%}"
             f" + Efficiency {weights.get('efficiency', 0.2):.0%}"
             f"  |  * = Pareto-optimal",
             fontsize=12, color=COLORS["text_dim"], ha="center")

    labels = [r["label"] for r in results]
    max_spd = max(r.get("avg_tokens_per_second", 0) for r in results)
    max_eff = max(r.get("efficiency_tps_per_gb", 0) for r in results)
    wq = weights.get("quality", 0.5)
    ws = weights.get("speed", 0.3)
    we = weights.get("efficiency", 0.2)

    quality_scores = [r.get("accuracy_pct", 0) * wq for r in results]
    speed_scores = [(r.get("avg_tokens_per_second", 0) / max_spd * 100 * ws) if max_spd > 0 else 0 for r in results]
    eff_scores = [(r.get("efficiency_tps_per_gb", 0) / max_eff * 100 * we) if max_eff > 0 else 0 for r in results]

    y_pos = np.arange(n)

    ax.barh(y_pos, quality_scores, color=COLORS["accent2"], alpha=0.9, height=0.6, label="Quality (accuracy)")
    ax.barh(y_pos, speed_scores, left=quality_scores, color=COLORS["accent"], alpha=0.9, height=0.6, label="Speed (throughput)")
    lefts = [q + s for q, s in zip(quality_scores, speed_scores)]
    ax.barh(y_pos, eff_scores, left=lefts, color=COLORS["accent3"], alpha=0.9, height=0.6, label="Efficiency (t/s per GB)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Composite Score", fontsize=13)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.4)

    # Total score labels with Pareto marker
    for i, r in enumerate(results):
        total = r.get("composite_score", 0)
        pareto = " *" if r.get("pareto_optimal") else ""
        ax.text(total + 0.3, i, f"{total:.1f}{pareto}",
                va="center", fontsize=11, fontweight="bold", color=COLORS["text"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.91])
    path = os.path.join(output_dir, "kadima_6_composite.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [6/7] Composite Breakdown")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7: ENERGY & VRAM — Power efficiency and memory footprint
# ═══════════════════════════════════════════════════════════════════════════════

def chart7_energy_and_vram(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    models_with_gpu = [r for r in results if r.get("gpu_snapshot")]
    if not models_with_gpu:
        print(f"    [7/7] Energy & VRAM (skipped: no GPU data)")
        return

    # Sort by peak VRAM ascending
    models_with_gpu.sort(key=lambda r: r["gpu_snapshot"]["peak_vram_mb"])

    n = len(models_with_gpu)
    labels = [r["label"] for r in models_with_gpu]
    colors = [get_family_color(r["family"]) for r in models_with_gpu]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, max(9, n * 0.85)),
                                    gridspec_kw={"width_ratios": [1, 1]})
    fig.suptitle("GPU Resource Usage — VRAM Footprint & Power Consumption",
                 fontsize=22, fontweight="bold", color=COLORS["text"], y=1.02)

    # Insight callout
    most_efficient = min(models_with_gpu,
                         key=lambda r: r["gpu_snapshot"]["avg_power_w"] / max(r.get("avg_tokens_per_second", 1), 1))
    eff_watts_per_tok = most_efficient["gpu_snapshot"]["avg_power_w"] / max(most_efficient.get("avg_tokens_per_second", 1), 1)
    fig.text(0.5, 0.94,
             f"Most power-efficient: {most_efficient['label']}"
             f" ({eff_watts_per_tok:.2f} watts per token/s)"
             f"  |  16GB VRAM budget line shown",
             fontsize=10, color=COLORS["text_dim"], ha="center")

    # LEFT: VRAM usage
    y_pos = np.arange(n)
    vrams = [r["gpu_snapshot"]["peak_vram_mb"] / 1024 for r in models_with_gpu]  # Convert to GB
    bars1 = ax1.barh(y_pos, vrams, color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Peak VRAM Usage (GB)", fontsize=12, labelpad=8)
    ax1.set_title("Peak VRAM Usage\n(lower = more headroom for context)", fontsize=12, fontweight="bold", pad=18)
    ax1.grid(axis="x", alpha=0.3)

    # 16GB budget line
    ax1.axvline(x=16, color=COLORS["accent5"], linestyle="--", linewidth=2, alpha=0.6)
    ax1.text(16.1, -0.5, "16GB\nbudget", fontsize=9, color=COLORS["accent5"], alpha=0.7, va="top")

    for bar, val in zip(bars1, vrams):
        color = COLORS["accent5"] if val > 14 else COLORS["text"]
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}GB", va="center", fontsize=10, fontweight="bold", color=color)

    # RIGHT: Power consumption vs throughput (bubble chart)
    powers = [r["gpu_snapshot"]["avg_power_w"] for r in models_with_gpu]
    throughputs = [r.get("avg_tokens_per_second", 0) for r in models_with_gpu]

    # Plot bubbles first, then labels with repulsion
    for i, r in enumerate(models_with_gpu):
        size = max(r.get("model_size_gb", 2) * 80, 100)
        ax2.scatter(powers[i], throughputs[i], s=size, c=colors[i],
                    alpha=0.85, edgecolors="white", linewidth=1.2, zorder=5)

    # Smart label placement for power vs throughput scatter
    placed = []
    for i, r in enumerate(models_with_gpu):
        # Try multiple offset positions and pick one that doesn't collide
        candidates = [(10, 8, "left"), (-10, 8, "right"), (10, -12, "left"), (-10, -12, "right"),
                      (0, 16, "center"), (0, -18, "center")]
        best_ox, best_oy, best_ha = candidates[0]
        for ox, oy, ha in candidates:
            collision = False
            for px, py, pox, poy in placed:
                dx = (powers[i] - px) / max(max(powers) - min(powers), 1) * 100
                dy = (throughputs[i] - py) / max(max(throughputs) - min(throughputs), 1) * 100
                if abs(dx + ox - pox) < 12 and abs(dy + oy - poy) < 12:
                    collision = True
                    break
            if not collision:
                best_ox, best_oy, best_ha = ox, oy, ha
                break
        placed.append((powers[i], throughputs[i], best_ox, best_oy))
        ax2.annotate(r["label"], (powers[i], throughputs[i]),
                     xytext=(best_ox, best_oy), textcoords="offset points",
                     fontsize=8.5, color=COLORS["text"], fontweight="bold", ha=best_ha,
                     arrowprops=dict(arrowstyle="-", color=COLORS["text_dim"], alpha=0.3, lw=0.5))

    ax2.set_xlabel("Average Power Draw (watts)", fontsize=12, labelpad=8)
    ax2.set_ylabel("Throughput (tokens/s)", fontsize=12, labelpad=8)
    ax2.set_title("Power vs. Throughput\n(top-left = most power efficient)", fontsize=12, fontweight="bold", pad=18)
    ax2.grid(True, alpha=0.2)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    path = os.path.join(output_dir, "kadima_7_energy_vram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [7/7] Energy & VRAM")
