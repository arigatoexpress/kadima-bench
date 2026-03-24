"""Chart generation — 10 publication-quality visualizations."""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
import numpy as np

from kadima_bench.visualize.theme import (
    COLORS, FAMILY_COLORS, setup_style, add_branding,
    get_family_color, get_family_marker, make_legend_handles,
)


def generate_all_charts(results_path: str) -> None:
    """Generate all charts from a results JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = os.path.dirname(results_path)
    hardware = data["metadata"]["hardware"]

    print(f"\n  Generating charts in {output_dir}...")

    chart1_leaderboard(data, hardware, output_dir)
    chart2_scatter(data, hardware, output_dir)
    chart3_category_heatmap(data, hardware, output_dir)
    chart4_nemotron_and_size(data, hardware, output_dir)
    chart5_family_comparison(data, hardware, output_dir)
    chart6_speed_by_category(data, hardware, output_dir)
    chart7_latency_distribution(data, hardware, output_dir)
    chart8_pareto_frontier(data, hardware, output_dir)
    chart9_efficiency_radar(data, hardware, output_dir)
    chart10_composite_breakdown(data, hardware, output_dir)

    print(f"  All 10 charts generated!")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 1: Leaderboard — horizontal bars for accuracy + speed
# ═══════════════════════════════════════════════════════════════════════════════

def chart1_leaderboard(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10), gridspec_kw={"width_ratios": [1.2, 1]})
    n = data["metadata"]["models_tested"]
    date = data["metadata"]["date"][:10]
    fig.suptitle("Local LLM Benchmark -- Overall Leaderboard",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.925, f"{n} Models  |  GPU-Isolated  |  {date}",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    labels = [r["label"] for r in results]
    accuracies = [r["accuracy_pct"] for r in results]
    families = [r["family"] for r in results]
    bar_colors = [get_family_color(f) for f in families]

    y_pos = np.arange(len(labels))

    # Left: Accuracy
    bars = ax1.barh(y_pos, accuracies, color=bar_colors, alpha=0.85, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy (% Tests Passed)", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlim(0, 110)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.0f}%", va="center", fontsize=10, fontweight="bold", color=COLORS["text"])

    # Right: Speed
    speeds = [r["avg_tokens_per_second"] for r in results]
    bars2 = ax2.barh(y_pos, speeds, color=bar_colors, alpha=0.85, height=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=10)
    ax2.set_xlabel("Tokens/Second", fontsize=12)
    ax2.set_title("Inference Speed (tokens/s)", fontsize=14, fontweight="bold", pad=15)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)

    for bar, spd in zip(bars2, speeds):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{spd:.0f}", va="center", fontsize=10, fontweight="bold", color=COLORS["text"])

    # Legend
    handles = make_legend_handles(families)
    fig.legend(handles=handles, loc="upper center", ncol=min(len(set(families)), 8),
               bbox_to_anchor=(0.5, 0.905), fontsize=10, framealpha=0.3, edgecolor=COLORS["border"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.89])
    path = os.path.join(output_dir, "kadima_1_leaderboard.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [1/10] Leaderboard")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 2: Speed vs. Accuracy scatter — with adjustText-style label placement
# ═══════════════════════════════════════════════════════════════════════════════

def chart2_scatter(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle("Speed vs. Accuracy -- Efficiency Frontier",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92, "Top-right = best (fast AND accurate)  |  Bubble size = model disk size",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    # Plot points
    for r in results:
        color = get_family_color(r["family"])
        marker = get_family_marker(r["family"])
        bubble = max(r.get("model_size_gb", 3) * 60, 100)
        ax.scatter(r["avg_tokens_per_second"], r["accuracy_pct"],
                   s=bubble, c=color, marker=marker, alpha=0.85,
                   edgecolors="white", linewidth=1.5, zorder=5)

    # Smart label placement — grid-based collision avoidance
    speeds = [r["avg_tokens_per_second"] for r in results]
    accs = [r["accuracy_pct"] for r in results]
    x_range = max(speeds) - min(speeds) if len(set(speeds)) > 1 else 100
    y_range = max(accs) - min(accs) if len(set(accs)) > 1 else 50

    placed_boxes = []

    for r in results:
        x, y = r["avg_tokens_per_second"], r["accuracy_pct"]
        label = r["label"]

        # Try multiple offset positions to avoid overlap
        candidates = [
            (12, 8, "left"), (12, -12, "left"), (-12, 8, "right"), (-12, -12, "right"),
            (20, 0, "left"), (-20, 0, "right"), (0, 16, "center"), (0, -16, "center"),
        ]

        best_offset = candidates[0]
        for ox, oy, ha in candidates:
            # Convert offset to data coords for collision check
            box_x = x + ox * x_range / 400
            box_y = y + oy * y_range / 200
            collision = False
            for bx, by in placed_boxes:
                if abs(box_x - bx) < x_range * 0.08 and abs(box_y - by) < y_range * 0.08:
                    collision = True
                    break
            if not collision:
                best_offset = (ox, oy, ha)
                break

        ox, oy, ha = best_offset
        ax.annotate(
            label, (x, y), xytext=(ox, oy), textcoords="offset points",
            fontsize=9, color=COLORS["text"], alpha=0.95, ha=ha, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=COLORS["text_dim"], alpha=0.3, lw=0.5),
        )
        placed_boxes.append((x + ox * x_range / 400, y + oy * y_range / 200))

    ax.set_xlabel("Inference Speed (tokens/second)", fontsize=13, labelpad=10)
    ax.set_ylabel("Accuracy (% tests passed)", fontsize=13, labelpad=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-10, max(speeds) + 30)

    # Quadrant markers
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    mid_x, mid_y = (xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2
    ax.axhline(y=mid_y, color=COLORS["grid"], linestyle="--", alpha=0.3)
    ax.axvline(x=mid_x, color=COLORS["grid"], linestyle="--", alpha=0.3)
    ax.text(xlim[1] * 0.95, ylim[1] * 0.98, "IDEAL", fontsize=12,
            color=COLORS["accent2"], alpha=0.4, ha="right", va="top", fontweight="bold")

    handles = make_legend_handles([r["family"] for r in results])
    ax.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.3)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_2_efficiency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [2/10] Speed vs Accuracy")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 3: Category Heatmap — pass/fail matrix
# ═══════════════════════════════════════════════════════════════════════════════

def chart3_category_heatmap(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [t["category"] for t in results[0]["test_results"]]
    matrix = np.array([[1 if tr["passed"] else 0 for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.7)))
    fig.suptitle("Test Category Breakdown -- Pass/Fail Matrix",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92, "Green = Pass  |  Red = Fail  |  Each model tested in GPU isolation",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    cmap = ListedColormap(["#FF4444", "#3FB950"])
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(categories)):
            text = "PASS" if matrix[i, j] == 1 else "FAIL"
            color = "#0D1117" if matrix[i, j] == 1 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    for i, r in enumerate(results):
        ax.text(len(categories) + 0.3, i, f"{r['accuracy_pct']:.0f}%",
                ha="left", va="center", fontsize=11, fontweight="bold",
                color=COLORS["accent2"] if r["accuracy_pct"] >= 70 else COLORS["accent5"])

    ax.grid(False)
    for i in range(len(models) + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2)
    for j in range(len(categories) + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_3_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [3/10] Category Heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 4: Nemotron Deep Dive + All Models Size Efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def chart4_nemotron_and_size(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle("Nemotron Quantization Analysis & Model Size Efficiency",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)

    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2], wspace=0.3)

    # LEFT: Nemotron variants
    ax1 = fig.add_subplot(gs[0])
    nemotron = sorted(
        [r for r in results if r.get("family") == "NVIDIA"],
        key=lambda x: -x["avg_tokens_per_second"],
    )

    if nemotron:
        labels = [r["label"] for r in nemotron]
        acc = [r["accuracy_pct"] for r in nemotron]
        spd = [r["avg_tokens_per_second"] for r in nemotron]
        sizes = [r.get("model_size_gb", 3) for r in nemotron]

        x = np.arange(len(labels))
        width = 0.25

        bars1 = ax1.bar(x - width, acc, width, label="Accuracy %", color="#76B900", alpha=0.85)
        bars2 = ax1.bar(x, spd, width, label="Tokens/s", color=COLORS["accent"], alpha=0.85)
        bars3 = ax1.bar(x + width, [s * 10 for s in sizes], width, label="Size (GB x10)", color=COLORS["accent4"], alpha=0.7)

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax1.set_ylabel("Value", fontsize=11)
        ax1.set_title("NVIDIA Nemotron Family\nQuantization Impact", fontsize=13, pad=15, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        ax1.legend(fontsize=9, framealpha=0.3, loc="upper right")

        for bar, val in zip(bars1, acc):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.0f}%", ha="center", fontsize=8, fontweight="bold", color="#76B900")
        for bar, val in zip(bars2, spd):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{val:.0f}", ha="center", fontsize=8, fontweight="bold", color=COLORS["accent"])
    else:
        ax1.text(0.5, 0.5, "No NVIDIA models in this run", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=14, color=COLORS["text_dim"])
        ax1.set_title("NVIDIA Nemotron Family", fontsize=13, pad=15, fontweight="bold")

    # RIGHT: All models size vs accuracy
    ax2 = fig.add_subplot(gs[1])
    all_sizes = [r["model_size_gb"] for r in results]
    all_accs = [r["accuracy_pct"] for r in results]
    all_speeds = [r["avg_tokens_per_second"] for r in results]
    all_families = [r["family"] for r in results]
    all_labels = [r["label"] for r in results]

    max_spd = max(all_speeds) if all_speeds else 1
    bubble_sizes = [max(s / max_spd * 600, 60) for s in all_speeds]

    for i in range(len(results)):
        ax2.scatter(all_sizes[i], all_accs[i], s=bubble_sizes[i],
                    c=get_family_color(all_families[i]),
                    alpha=0.85, edgecolors="white", linewidth=1.2, zorder=5)
        ax2.annotate(f"{all_labels[i]}\n{all_speeds[i]:.0f} t/s",
                     (all_sizes[i], all_accs[i]),
                     xytext=(8, -3), textcoords="offset points",
                     fontsize=7.5, color=COLORS["text"], alpha=0.9)

    ax2.set_xlabel("Model Size (GB)", fontsize=11, labelpad=8)
    ax2.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)
    ax2.set_title("All Models: Size vs. Accuracy\nBubble = inference speed", fontsize=13, pad=15, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(30, 108)

    handles = make_legend_handles(all_families)
    ax2.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.3)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_4_nemotron_and_size.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [4/10] Nemotron & Size Analysis")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 5: Best-in-Family Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def chart5_family_comparison(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle("Best-in-Family Comparison",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92, "Top performer from each model family  |  Fair cross-vendor comparison",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    families = {}
    for r in results:
        fam = r["family"]
        score = (r["accuracy_pct"], r["avg_tokens_per_second"])
        if fam not in families or score > (families[fam]["accuracy_pct"], families[fam]["avg_tokens_per_second"]):
            families[fam] = r

    best = sorted(families.values(), key=lambda x: (-x["accuracy_pct"], -x["avg_tokens_per_second"]))
    labels = [f"{r['label']}\n({r['family']})" for r in best]
    acc = [r["accuracy_pct"] for r in best]
    spd = [r["avg_tokens_per_second"] for r in best]
    colors = [get_family_color(r["family"]) for r in best]

    x = np.arange(len(labels))
    width = 0.25
    max_spd = max(spd) if spd else 1

    ax.bar(x - width, acc, width, color=colors, alpha=0.9, label="Accuracy (%)")
    ax.bar(x, [s / max_spd * 100 for s in spd], width, color=colors, alpha=0.5, hatch="///", label="Speed (normalized %)")

    composite = [0.6 * a + 0.4 * (s / max_spd * 100) for a, s in zip(acc, spd)]
    bars3 = ax.bar(x + width, composite, width, color=colors, alpha=0.7, edgecolor="white", linewidth=1.5, label="Composite Score")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.3)

    for bar, val in zip(bars3, composite):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{val:.0f}", ha="center", fontsize=10, fontweight="bold", color=COLORS["accent2"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_5_family_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [5/10] Family Comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 6: Speed by Category Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def chart6_speed_by_category(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    if not results[0].get("test_results"):
        return

    models = [r["label"] for r in results]
    categories = [tr["category"] for tr in results[0]["test_results"]]
    matrix = np.array([[tr.get("tokens_per_second", 0) for tr in r["test_results"]] for r in results])

    fig, ax = plt.subplots(figsize=(18, max(8, len(models) * 0.7)))
    fig.suptitle("Inference Speed by Test Category (tokens/second)",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92, "Higher = faster  |  Color intensity = speed  |  GPU-isolated per model",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    im = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0)
    ax.set_xticks(np.arange(len(categories)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = "#0D1117" if val > 100 else COLORS["text"]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    for i, r in enumerate(results):
        avg = r["avg_tokens_per_second"]
        color = COLORS["accent2"] if avg >= 100 else COLORS["accent"] if avg >= 50 else COLORS["accent5"]
        ax.text(len(categories) + 0.3, i, f"avg: {avg:.0f} t/s",
                ha="left", va="center", fontsize=10, fontweight="bold", color=color)

    ax.grid(False)
    for i in range(len(models) + 1):
        ax.axhline(i - 0.5, color=COLORS["bg"], linewidth=2)
    for j in range(len(categories) + 1):
        ax.axvline(j - 0.5, color=COLORS["bg"], linewidth=2)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label("Tokens/second", color=COLORS["text"], fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=COLORS["text_dim"])
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=COLORS["text_dim"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_6_speed_by_category.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [6/10] Speed by Category")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 7: NEW — Latency Distribution (box plot of TTFT and ITL)
# ═══════════════════════════════════════════════════════════════════════════════

def chart7_latency_distribution(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    # Collect models with speed_metrics
    models_with_speed = [r for r in results if r.get("speed_metrics")]
    if not models_with_speed:
        print(f"    [7/10] Latency Distribution (skipped: no speed metrics)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.suptitle("Latency Analysis -- TTFT & Inter-Token Latency",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92, "Lower = better  |  Streaming inference metrics  |  Multiple runs averaged",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    labels = [r["label"] for r in models_with_speed]
    colors = [get_family_color(r["family"]) for r in models_with_speed]

    # Left: TTFT bar chart
    ttfts = [r["speed_metrics"]["ttft_ms"] for r in models_with_speed]
    y_pos = np.arange(len(labels))
    bars = ax1.barh(y_pos, ttfts, color=colors, alpha=0.85, height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_xlabel("Time to First Token (ms)", fontsize=12)
    ax1.set_title("TTFT (Time to First Token)", fontsize=14, fontweight="bold", pad=15)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, ttfts):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}ms", va="center", fontsize=9, fontweight="bold", color=COLORS["text"])

    # Right: ITL percentiles as grouped bars
    p50s = [r["speed_metrics"]["itl_p50_ms"] for r in models_with_speed]
    p95s = [r["speed_metrics"]["itl_p95_ms"] for r in models_with_speed]
    p99s = [r["speed_metrics"]["itl_p99_ms"] for r in models_with_speed]

    x = np.arange(len(labels))
    width = 0.25

    ax2.bar(x - width, p50s, width, color=COLORS["accent2"], alpha=0.85, label="p50 (median)")
    ax2.bar(x, p95s, width, color=COLORS["accent4"], alpha=0.85, label="p95")
    ax2.bar(x + width, p99s, width, color=COLORS["accent5"], alpha=0.85, label="p99")

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Inter-Token Latency (ms)", fontsize=12)
    ax2.set_title("ITL Percentiles (p50 / p95 / p99)", fontsize=14, fontweight="bold", pad=15)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=10, framealpha=0.3, loc="upper right")

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_7_latency.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [7/10] Latency Distribution")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 8: NEW — Pareto Frontier
# ═══════════════════════════════════════════════════════════════════════════════

def chart8_pareto_frontier(data, hardware, output_dir):
    results = data["results"]
    pareto_labels = set(data.get("pareto_frontier", []))
    setup_style()

    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle("Pareto Frontier -- Optimal Speed/Accuracy Trade-offs",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92,
             "Stars = Pareto-optimal (not dominated on speed+accuracy)  |  Connected by frontier line",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    # Plot all models
    for r in results:
        is_pareto = r["label"] in pareto_labels
        color = get_family_color(r["family"])
        marker = "*" if is_pareto else get_family_marker(r["family"])
        size = 400 if is_pareto else 150
        edge = COLORS["accent2"] if is_pareto else "white"
        edge_w = 2.5 if is_pareto else 1.2

        ax.scatter(r["avg_tokens_per_second"], r["accuracy_pct"],
                   s=size, c=color, marker=marker, alpha=0.9,
                   edgecolors=edge, linewidth=edge_w, zorder=10 if is_pareto else 5)

        offset = (12, 5) if is_pareto else (8, -8)
        ax.annotate(r["label"], (r["avg_tokens_per_second"], r["accuracy_pct"]),
                    xytext=offset, textcoords="offset points",
                    fontsize=9 if is_pareto else 8, fontweight="bold" if is_pareto else "normal",
                    color=COLORS["text"], alpha=0.95)

    # Draw Pareto frontier line
    pareto_points = sorted(
        [(r["avg_tokens_per_second"], r["accuracy_pct"]) for r in results if r["label"] in pareto_labels],
        key=lambda p: p[0],
    )
    if len(pareto_points) > 1:
        px, py = zip(*pareto_points)
        ax.plot(px, py, color=COLORS["accent2"], linewidth=2, alpha=0.6, linestyle="--", zorder=3)

    ax.set_xlabel("Inference Speed (tokens/second)", fontsize=13, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=13, labelpad=10)
    ax.grid(True, alpha=0.3)

    # Custom legend
    handles = make_legend_handles([r["family"] for r in results])
    handles.append(plt.scatter([], [], s=300, c="white", marker="*", edgecolors=COLORS["accent2"],
                               linewidth=2, label="Pareto Optimal"))
    ax.legend(handles=handles, loc="lower right", fontsize=10, framealpha=0.3)

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_8_pareto.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [8/10] Pareto Frontier")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 9: NEW — Efficiency Radar (spider chart per family)
# ═══════════════════════════════════════════════════════════════════════════════

def chart9_efficiency_radar(data, hardware, output_dir):
    results = data["results"]
    setup_style()

    # Get best from each family
    families = {}
    for r in results:
        fam = r["family"]
        score = r.get("composite_score", 0)
        if fam not in families or score > families[fam].get("composite_score", 0):
            families[fam] = r

    if len(families) < 2:
        print(f"    [9/10] Efficiency Radar (skipped: <2 families)")
        return

    best = list(families.values())

    # 5 axes: accuracy, speed, efficiency, compactness, consistency
    categories_labels = ["Accuracy", "Speed", "Efficiency\n(t/s per GB)", "Compactness\n(1/size)", "Composite\nScore"]
    N = len(categories_labels)

    # Normalize each axis 0-100
    max_spd = max(r.get("avg_tokens_per_second", 0) for r in best)
    max_eff = max(r.get("efficiency_tps_per_gb", 0) for r in best)
    max_compact = max(1 / r.get("model_size_gb", 1) for r in best)
    max_comp = max(r.get("composite_score", 0) for r in best)

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    fig.suptitle("Multi-Dimensional Efficiency Radar -- Best per Family",
                 fontsize=18, fontweight="bold", color=COLORS["text"], y=0.98)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    ax.set_facecolor(COLORS["card_bg"])

    for r in best:
        values = [
            r.get("accuracy_pct", 0),
            (r.get("avg_tokens_per_second", 0) / max_spd * 100) if max_spd > 0 else 0,
            (r.get("efficiency_tps_per_gb", 0) / max_eff * 100) if max_eff > 0 else 0,
            ((1 / r.get("model_size_gb", 1)) / max_compact * 100) if max_compact > 0 else 0,
            (r.get("composite_score", 0) / max_comp * 100) if max_comp > 0 else 0,
        ]
        values += values[:1]

        color = get_family_color(r["family"])
        ax.plot(angles, values, "o-", linewidth=2, label=f"{r['label']} ({r['family']})", color=color, alpha=0.8)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_labels, fontsize=11, color=COLORS["text"])
    ax.set_ylim(0, 105)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color=COLORS["text_dim"])
    ax.grid(True, alpha=0.3, color=COLORS["grid"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10, framealpha=0.3)

    add_branding(fig, hardware)
    path = os.path.join(output_dir, "kadima_9_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [9/10] Efficiency Radar")


# ═══════════════════════════════════════════════════════════════════════════════
# Chart 10: NEW — Composite Score Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def chart10_composite_breakdown(data, hardware, output_dir):
    results = data["results"]
    weights = data["metadata"].get("scoring_weights", {"quality": 0.5, "speed": 0.3, "efficiency": 0.2})
    setup_style()

    fig, ax = plt.subplots(figsize=(18, 10))
    fig.suptitle("Composite Score Breakdown -- Quality + Speed + Efficiency",
                 fontsize=20, fontweight="bold", color=COLORS["text"], y=0.96)
    fig.text(0.5, 0.92,
             f"Weights: Quality {weights.get('quality', 0.5):.0%} | Speed {weights.get('speed', 0.3):.0%} | Efficiency {weights.get('efficiency', 0.2):.0%}",
             fontsize=11, color=COLORS["text_dim"], ha="center")

    labels = [r["label"] for r in results]
    max_spd = max(r.get("avg_tokens_per_second", 0) for r in results)
    max_eff = max(r.get("efficiency_tps_per_gb", 0) for r in results)

    quality_scores = [r.get("accuracy_pct", 0) * weights.get("quality", 0.5) for r in results]
    speed_scores = [(r.get("avg_tokens_per_second", 0) / max_spd * 100 * weights.get("speed", 0.3)) if max_spd > 0 else 0 for r in results]
    eff_scores = [(r.get("efficiency_tps_per_gb", 0) / max_eff * 100 * weights.get("efficiency", 0.2)) if max_eff > 0 else 0 for r in results]

    y_pos = np.arange(len(labels))

    ax.barh(y_pos, quality_scores, color=COLORS["accent2"], alpha=0.85, height=0.6, label="Quality")
    ax.barh(y_pos, speed_scores, left=quality_scores, color=COLORS["accent"], alpha=0.85, height=0.6, label="Speed")
    lefts = [q + s for q, s in zip(quality_scores, speed_scores)]
    ax.barh(y_pos, eff_scores, left=lefts, color=COLORS["accent3"], alpha=0.85, height=0.6, label="Efficiency")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Composite Score", fontsize=12)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.3)

    # Total score labels
    for i, r in enumerate(results):
        total = r.get("composite_score", 0)
        pareto = " *" if r.get("pareto_optimal") else ""
        ax.text(total + 0.5, i, f"{total:.1f}{pareto}",
                va="center", fontsize=10, fontweight="bold", color=COLORS["text"])

    add_branding(fig, hardware)
    plt.tight_layout(rect=[0, 0.03, 1, 0.90])
    path = os.path.join(output_dir, "kadima_10_composite.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close()
    print(f"    [10/10] Composite Breakdown")
