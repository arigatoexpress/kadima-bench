"""
Kadima Bench — Open-source local LLM benchmarking framework.

Combines custom quality tests, lm-eval-harness standard benchmarks,
and production inference metrics (TTFT, ITL, percentiles, VRAM tracking).

Usage:
    kadima-bench run --models nemotron-3-nano:4b,phi4:14b --suite quick
    kadima-bench list-models
    kadima-bench report results.json
"""

__version__ = "0.1.0"
__author__ = "Kadima Digital Laboratories"
