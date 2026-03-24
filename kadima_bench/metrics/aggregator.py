"""Cross-model analytics: composite scores, Pareto frontier, rankings."""
from __future__ import annotations


def compute_composite_score(
    accuracy_pct: float,
    throughput_tps: float,
    max_throughput: float,
    efficiency_tps_per_gb: float = 0.0,
    max_efficiency: float = 1.0,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Compute weighted composite score (0-100).

    Args:
        accuracy_pct: Model accuracy as percentage (0-100)
        throughput_tps: Model throughput in tokens/second
        max_throughput: Max throughput across all models (for normalization)
        efficiency_tps_per_gb: Tokens/sec per GB model size
        max_efficiency: Max efficiency across all models
        weights: (quality_weight, speed_weight, efficiency_weight)
    """
    w_q, w_s, w_e = weights
    norm_speed = (throughput_tps / max_throughput * 100) if max_throughput > 0 else 0
    norm_eff = (efficiency_tps_per_gb / max_efficiency * 100) if max_efficiency > 0 else 0
    return round(w_q * accuracy_pct + w_s * norm_speed + w_e * norm_eff, 1)


def find_pareto_frontier(
    models: list[dict],
    x_key: str = "avg_tokens_per_second",
    y_key: str = "accuracy_pct",
) -> list[str]:
    """Find Pareto-optimal models (not dominated on x and y).

    Returns list of model labels on the frontier.
    """
    points = [(m[x_key], m[y_key], m["label"]) for m in models]
    # Sort by x descending
    points.sort(key=lambda p: -p[0])

    frontier = []
    max_y = -1.0
    for x, y, label in points:
        if y >= max_y:
            frontier.append(label)
            max_y = y

    return frontier
