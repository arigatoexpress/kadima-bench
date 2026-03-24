"""Base test suite protocol."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TestCase:
    """A single test to run against a model."""
    id: str
    category: str
    prompt: str
    verify: Callable[[str], bool] | None = None
    expected: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class TestResult:
    """Result from running one test against one model."""
    test_id: str
    category: str
    passed: bool | None = None
    score: float | None = None  # 0.0-1.0 for graded evaluation
    response: str = ""
    time_seconds: float = 0.0
    tokens: int = 0
    tokens_per_second: float = 0.0
    ttft_ms: float | None = None
    itl_p50_ms: float | None = None
    itl_p95_ms: float | None = None
    itl_p99_ms: float | None = None
    error: str | None = None


class TestSuite(ABC):
    """Abstract base for test suites."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def get_tests(self) -> list[TestCase]:
        """Return the list of tests in this suite."""
        ...
