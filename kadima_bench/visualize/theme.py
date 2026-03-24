"""Kadima brand colors, fonts, and matplotlib style configuration."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# -- Brand Colors -------------------------------------------------------------

COLORS = {
    "bg":           "#0D1117",
    "card_bg":      "#161B22",
    "text":         "#E6EDF3",
    "text_dim":     "#8B949E",
    "accent":       "#58A6FF",
    "accent2":      "#3FB950",
    "accent3":      "#D2A8FF",
    "accent4":      "#F0883E",
    "accent5":      "#FF7B72",
    "grid":         "#21262D",
    "border":       "#30363D",
}

FAMILY_COLORS = {
    "NVIDIA":     "#76B900",
    "Google":     "#4285F4",
    "Meta":       "#0668E1",
    "Microsoft":  "#00BCF2",
    "IBM":        "#BE95FF",
    "Alibaba":    "#FF6A00",
    "DeepSeek":   "#8B5CF6",
    "Zhipu":      "#EF4444",
    "Mistral":    "#FF4500",
}

FAMILY_MARKERS = {
    "NVIDIA":     "D",
    "Google":     "o",
    "Meta":       "^",
    "Microsoft":  "s",
    "IBM":        "P",
    "Alibaba":    "h",
    "DeepSeek":   "v",
    "Zhipu":      "X",
    "Mistral":    "d",
}


def setup_style():
    """Configure matplotlib for dark publication style."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card_bg"],
        "axes.edgecolor": COLORS["border"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text_dim"],
        "ytick.color": COLORS["text_dim"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def add_branding(fig, hardware: dict):
    """Add Kadima Digital Laboratories branding footer."""
    fig.text(
        0.02, 0.01,
        f"Kadima Digital Laboratories  |  {hardware.get('gpu', '')}  |  {hardware.get('cpu', '')}  |  {hardware.get('ram', '')}",
        fontsize=8, color=COLORS["text_dim"], alpha=0.7, fontstyle="italic",
    )
    fig.text(
        0.98, 0.01,
        f"Inference: {hardware.get('inference_engine', 'Ollama')}  |  {hardware.get('os', '')}  |  kadima-bench",
        fontsize=8, color=COLORS["text_dim"], alpha=0.7, ha="right", fontstyle="italic",
    )


def get_family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, COLORS["accent"])


def get_family_marker(family: str) -> str:
    return FAMILY_MARKERS.get(family, "o")


def make_legend_handles(families: list[str]) -> list:
    """Create legend handles for the given family list."""
    seen = set()
    handles = []
    for f in families:
        if f not in seen:
            seen.add(f)
            handles.append(mpatches.Patch(color=get_family_color(f), label=f))
    return handles
