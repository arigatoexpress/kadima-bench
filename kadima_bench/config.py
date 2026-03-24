"""Configuration management — TOML loading + defaults."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HardwareConfig:
    lab: str = "Kadima Digital Laboratories"
    cpu: str = "AMD Ryzen 9 9900X3D 12-Core"
    gpu: str = "NVIDIA GeForce RTX 5070 Ti (16GB GDDR7)"
    ram: str = "64GB DDR5"
    motherboard: str = "ASUS ROG STRIX B850-A GAMING WIFI"
    storage: str = "Samsung 990 EVO Plus 2TB NVMe"
    os: str = "Windows 11 Pro"
    inference_engine: str = "Ollama"


@dataclass
class BackendConfig:
    type: str = "ollama"
    base_url: str = "http://localhost:11434"
    timeout: int = 120


@dataclass
class SuiteConfig:
    preset: str = "quick"
    custom_tests: bool = True
    lm_eval: bool = False
    lm_eval_tasks: list[str] = field(default_factory=lambda: [
        "mmlu", "gsm8k", "hellaswag", "arc_challenge",
    ])
    lm_eval_limit: int | None = 100
    lm_eval_fewshot: int = 5
    speed_tests: bool = True
    speed_repeats: int = 3


@dataclass
class ScoringConfig:
    quality_weight: float = 0.5
    speed_weight: float = 0.3
    efficiency_weight: float = 0.2


@dataclass
class ModelOverride:
    name: str
    use_chat_api: bool = False
    think_off: bool = False
    label: str | None = None
    family: str | None = None


@dataclass
class RunConfig:
    """Complete configuration for a benchmark run."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    suite: SuiteConfig = field(default_factory=SuiteConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    models: list[str] = field(default_factory=list)  # empty = auto-discover
    model_overrides: dict[str, ModelOverride] = field(default_factory=dict)
    output_dir: str = "."
    gpu_isolation: bool = True
    gpu_monitor: bool = True
    version: str = "0.1.0"


# -- Known model metadata (auto-detected from Ollama, overridable) -----------

KNOWN_MODELS: dict[str, dict] = {
    "nemotron-3-nano:4b":       {"label": "Nemotron 3 Nano 4B",  "family": "NVIDIA",     "params": "4B",   "quant": "Q4_K_M"},
    "nemotron-3-nano:4b-q8_0":  {"label": "Nemotron 3 Nano Q8",  "family": "NVIDIA",     "params": "4B",   "quant": "Q8_0"},
    "nemotron-mini:4b":         {"label": "Nemotron Mini 4B",    "family": "NVIDIA",     "params": "4B",   "quant": "Q4_K_M"},
    "gemma3:4b":                {"label": "Gemma 3 4B",          "family": "Google",     "params": "4B",   "quant": "Q4_K_M"},
    "gemma3n:e4b":              {"label": "Gemma 3n E4B",        "family": "Google",     "params": "~4B",  "quant": "F16"},
    "llama3.2:3b":              {"label": "Llama 3.2 3B",        "family": "Meta",       "params": "3B",   "quant": "Q4_K_M"},
    "phi4-mini:latest":         {"label": "Phi-4 Mini 3.8B",     "family": "Microsoft",  "params": "3.8B", "quant": "Q4_K_M"},
    "phi4:latest":              {"label": "Phi-4 14B",           "family": "Microsoft",  "params": "14B",  "quant": "Q4_K_M"},
    "granite3.3:2b":            {"label": "Granite 3.3 2B",      "family": "IBM",        "params": "2B",   "quant": "Q4_K_M"},
    "qwen3.5:4b":               {"label": "Qwen 3.5 4B",         "family": "Alibaba",    "params": "4B",   "quant": "Q4_K_M"},
    "qwen3:14b":                {"label": "Qwen 3 14B",          "family": "Alibaba",    "params": "14B",  "quant": "Q4_K_M"},
    "qwen2.5:14b":              {"label": "Qwen 2.5 14B",        "family": "Alibaba",    "params": "14B",  "quant": "Q4_K_M"},
    "deepseek-r1:14b":          {"label": "DeepSeek R1 14B",     "family": "DeepSeek",   "params": "14B",  "quant": "Q4_K_M"},
    "deepseek-r1:32b":          {"label": "DeepSeek R1 32B",     "family": "DeepSeek",   "params": "32B",  "quant": "Q4_K_M"},
    "glm4:9b":                  {"label": "GLM-4 9B",            "family": "Zhipu",      "params": "9B",   "quant": "Q4_K_M"},
    "ministral:3b":             {"label": "Ministral 3B",        "family": "Mistral",    "params": "3B",   "quant": "Q4_K_M"},
    "granite3.3:8b":            {"label": "Granite 3.3 8B",      "family": "IBM",        "params": "8B",   "quant": "Q4_K_M"},
}

# Models requiring chat API / think_off
CHAT_MODELS = {"qwen3.5:4b", "qwen3:14b"}


def load_config(path: str | Path | None = None) -> RunConfig:
    """Load config from TOML file, falling back to defaults."""
    config = RunConfig()

    if path is None:
        return config

    path = Path(path)
    if not path.exists():
        return config

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    # Map TOML sections to config dataclasses
    if "hardware" in raw:
        for k, v in raw["hardware"].items():
            if hasattr(config.hardware, k):
                setattr(config.hardware, k, v)

    if "backend" in raw:
        for k, v in raw["backend"].items():
            if hasattr(config.backend, k):
                setattr(config.backend, k, v)

    if "suite" in raw:
        for k, v in raw["suite"].items():
            if hasattr(config.suite, k):
                setattr(config.suite, k, v)

    if "scoring" in raw:
        for k, v in raw["scoring"].items():
            if hasattr(config.scoring, k):
                setattr(config.scoring, k, v)

    if "models" in raw:
        if "include" in raw["models"]:
            config.models = raw["models"]["include"]

    if "general" in raw:
        gen = raw["general"]
        config.output_dir = gen.get("output_dir", config.output_dir)
        config.gpu_isolation = gen.get("gpu_isolation", config.gpu_isolation)
        config.gpu_monitor = gen.get("gpu_monitor", config.gpu_monitor)

    return config
