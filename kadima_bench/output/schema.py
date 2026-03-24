"""Pydantic models for structured benchmark output."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional


class TestResultSchema(BaseModel):
    test_id: str
    category: str
    passed: bool | None = None
    score: float | None = None
    time_seconds: float = 0.0
    tokens: int = 0
    tokens_per_second: float = 0.0
    ttft_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None


class SpeedMetricsSchema(BaseModel):
    ttft_ms: float = 0.0
    itl_mean_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tps: float = 0.0
    total_tokens: int = 0
    total_time_s: float = 0.0


class GpuSnapshotSchema(BaseModel):
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    peak_power_w: float = 0.0
    avg_power_w: float = 0.0
    samples: int = 0
    duration_seconds: float = 0.0


class LmEvalResultSchema(BaseModel):
    task: str
    accuracy: float | None = None
    accuracy_stderr: float | None = None
    num_samples: int = 0


class ModelResultSchema(BaseModel):
    model: str
    label: str
    family: str
    params: str
    quantization: str
    model_size_gb: float = 0.0

    # Quality
    tests_passed: int = 0
    tests_total: int = 0
    accuracy_pct: float = 0.0
    test_results: list[TestResultSchema] = []
    lm_eval_results: list[LmEvalResultSchema] = []

    # Speed
    avg_tokens_per_second: float = 0.0
    avg_response_time: float = 0.0
    total_time_seconds: float = 0.0
    speed_metrics: SpeedMetricsSchema | None = None

    # Efficiency
    efficiency_tps_per_gb: float = 0.0
    energy_per_token_mj: float | None = None  # millijoules per token
    gpu_snapshot: GpuSnapshotSchema | None = None

    # Composite
    composite_score: float = 0.0
    pareto_optimal: bool = False


class HardwareSchema(BaseModel):
    lab: str = "Kadima Digital Laboratories"
    cpu: str = ""
    gpu: str = ""
    ram: str = ""
    motherboard: str = ""
    storage: str = ""
    os: str = ""
    inference_engine: str = "Ollama"


class MetadataSchema(BaseModel):
    lab: str = "Kadima Digital Laboratories"
    hardware: HardwareSchema
    version: str = "0.1.0"
    timestamp: str = ""
    date: str = ""
    models_tested: int = 0
    tests_per_model: int = 0
    categories: list[str] = []
    methodology: str = ""
    suite_preset: str = "quick"
    scoring_weights: dict[str, float] = {}


class BenchmarkOutput(BaseModel):
    metadata: MetadataSchema
    results: list[ModelResultSchema] = []
    pareto_frontier: list[str] = []
    rankings: dict[str, list[str]] = {}
