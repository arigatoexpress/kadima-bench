"""Speed metrics computed from streaming responses."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SpeedMetrics:
    """Production-grade latency metrics from streaming inference."""
    ttft_ms: float = 0.0          # time to first token
    itl_mean_ms: float = 0.0      # mean inter-token latency
    itl_p50_ms: float = 0.0       # median ITL
    itl_p95_ms: float = 0.0       # 95th percentile ITL
    itl_p99_ms: float = 0.0       # 99th percentile ITL
    tpot_ms: float = 0.0          # time per output token (mean)
    throughput_tps: float = 0.0   # tokens per second
    total_tokens: int = 0
    total_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ttft_ms": round(self.ttft_ms, 1),
            "itl_mean_ms": round(self.itl_mean_ms, 2),
            "itl_p50_ms": round(self.itl_p50_ms, 2),
            "itl_p95_ms": round(self.itl_p95_ms, 2),
            "itl_p99_ms": round(self.itl_p99_ms, 2),
            "tpot_ms": round(self.tpot_ms, 2),
            "throughput_tps": round(self.throughput_tps, 1),
            "total_tokens": self.total_tokens,
            "total_time_s": round(self.total_time_s, 3),
        }


def compute_speed_metrics(
    ttft_ms: float,
    itl_ms_list: list[float],
    total_tokens: int,
    total_time_s: float,
) -> SpeedMetrics:
    """Compute aggregated speed metrics from raw streaming data."""
    if not itl_ms_list:
        tps = total_tokens / total_time_s if total_time_s > 0 else 0.0
        return SpeedMetrics(
            ttft_ms=ttft_ms,
            throughput_tps=tps,
            total_tokens=total_tokens,
            total_time_s=total_time_s,
        )

    arr = np.array(itl_ms_list)

    return SpeedMetrics(
        ttft_ms=ttft_ms,
        itl_mean_ms=float(np.mean(arr)),
        itl_p50_ms=float(np.percentile(arr, 50)),
        itl_p95_ms=float(np.percentile(arr, 95)),
        itl_p99_ms=float(np.percentile(arr, 99)),
        tpot_ms=float(np.mean(arr)),  # TPOT ≈ mean ITL
        throughput_tps=total_tokens / total_time_s if total_time_s > 0 else 0.0,
        total_tokens=total_tokens,
        total_time_s=total_time_s,
    )


def merge_speed_metrics(metrics_list: list[SpeedMetrics]) -> SpeedMetrics:
    """Merge multiple speed metric runs (e.g. 3 repeats) into one averaged result."""
    if not metrics_list:
        return SpeedMetrics()
    if len(metrics_list) == 1:
        return metrics_list[0]

    return SpeedMetrics(
        ttft_ms=np.mean([m.ttft_ms for m in metrics_list]),
        itl_mean_ms=np.mean([m.itl_mean_ms for m in metrics_list]),
        itl_p50_ms=np.mean([m.itl_p50_ms for m in metrics_list]),
        itl_p95_ms=np.mean([m.itl_p95_ms for m in metrics_list]),
        itl_p99_ms=np.mean([m.itl_p99_ms for m in metrics_list]),
        tpot_ms=np.mean([m.tpot_ms for m in metrics_list]),
        throughput_tps=np.mean([m.throughput_tps for m in metrics_list]),
        total_tokens=sum(m.total_tokens for m in metrics_list),
        total_time_s=sum(m.total_time_s for m in metrics_list),
    )
