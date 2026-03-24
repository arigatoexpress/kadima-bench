"""GPU monitoring via nvidia-smi polling."""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class GpuSnapshot:
    """Aggregated GPU metrics from a monitoring session."""
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    peak_power_w: float = 0.0
    avg_power_w: float = 0.0
    samples: int = 0
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "avg_vram_mb": round(self.avg_vram_mb, 1),
            "peak_power_w": round(self.peak_power_w, 1),
            "avg_power_w": round(self.avg_power_w, 1),
            "samples": self.samples,
            "duration_seconds": round(self.duration_seconds, 1),
        }


class GpuMonitor:
    """Background thread that polls nvidia-smi every interval_ms milliseconds."""

    def __init__(self, interval_ms: int = 500):
        self.interval = interval_ms / 1000.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._vram_samples: list[float] = []
        self._power_samples: list[float] = []
        self._start_time: float = 0.0

    def _poll(self) -> tuple[float, float] | None:
        """Query nvidia-smi for current VRAM usage and power draw."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=memory.used,power.draw",
                 "--format=csv,nounits,noheader"],
                capture_output=True, encoding="utf-8",
                errors="replace", timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(",")
                vram = float(parts[0].strip())
                power = float(parts[1].strip()) if len(parts) > 1 else 0.0
                return vram, power
        except Exception:
            pass
        return None

    def _monitor_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            sample = self._poll()
            if sample:
                self._vram_samples.append(sample[0])
                self._power_samples.append(sample[1])
            time.sleep(self.interval)

    def start(self) -> None:
        """Start background GPU monitoring."""
        self._vram_samples.clear()
        self._power_samples.clear()
        self._start_time = time.perf_counter()
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> GpuSnapshot:
        """Stop monitoring and return aggregated snapshot."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

        duration = time.perf_counter() - self._start_time

        if not self._vram_samples:
            return GpuSnapshot(duration_seconds=duration)

        return GpuSnapshot(
            peak_vram_mb=max(self._vram_samples),
            avg_vram_mb=sum(self._vram_samples) / len(self._vram_samples),
            peak_power_w=max(self._power_samples) if self._power_samples else 0.0,
            avg_power_w=(sum(self._power_samples) / len(self._power_samples)
                         if self._power_samples else 0.0),
            samples=len(self._vram_samples),
            duration_seconds=duration,
        )
