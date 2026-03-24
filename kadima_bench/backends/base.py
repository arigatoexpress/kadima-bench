"""Abstract backend protocol for inference engines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class GenerateResult:
    """Result from a single non-streaming generation call."""
    text: str
    tokens: int
    prompt_tokens: int
    time_seconds: float
    eval_duration_ns: int
    tokens_per_second: float
    success: bool
    error: str | None = None


@dataclass
class StreamChunk:
    """A single token from a streaming generation call."""
    token: str
    timestamp_ns: int  # time.perf_counter_ns()


@dataclass
class StreamResult:
    """Full result from a streaming generation call, with timing data."""
    text: str
    tokens: int
    time_seconds: float
    tokens_per_second: float
    ttft_ms: float  # time to first token
    itl_ms: list[float] = field(default_factory=list)  # inter-token latencies
    success: bool = True
    error: str | None = None


class Backend(ABC):
    """Abstract base for inference backends (Ollama, SGLang, etc.)."""

    name: str = "base"

    @abstractmethod
    def generate(self, model: str, prompt: str, *,
                 max_tokens: int = 512, temperature: float = 0.0,
                 use_chat: bool = False, think_off: bool = False) -> GenerateResult:
        """Non-streaming generation. Returns complete response with timing."""
        ...

    @abstractmethod
    def generate_stream(self, model: str, prompt: str, *,
                        max_tokens: int = 512, temperature: float = 0.0,
                        use_chat: bool = False, think_off: bool = False) -> StreamResult:
        """Streaming generation. Returns response with per-token timing."""
        ...

    @abstractmethod
    def list_models(self) -> list[dict]:
        """List available models with metadata."""
        ...

    @abstractmethod
    def load_model(self, model: str, *, use_chat: bool = False) -> None:
        """Warmup: load a model into GPU memory."""
        ...

    @abstractmethod
    def unload_all(self) -> None:
        """Unload all models from GPU memory."""
        ...
