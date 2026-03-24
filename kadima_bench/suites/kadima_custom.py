"""Kadima custom test suite — 8 diverse, verifiable tests ported from v3."""
from __future__ import annotations

from kadima_bench.suites.base import TestCase, TestSuite


class KadimaCustomSuite(TestSuite):
    """8 hand-crafted tests covering code, math, logic, language, and structure."""

    name = "kadima-custom"
    description = "8 diverse tests: code gen, reasoning, math, summarization, instruction following, JSON, translation, creative writing"

    def get_tests(self) -> list[TestCase]:
        return [
            TestCase(
                id="code_gen",
                category="Code Generation",
                prompt="Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number using dynamic programming. Include a docstring. Only output the code, no explanation.",
                verify=lambda r: "def fibonacci" in r and "for" in r,
            ),
            TestCase(
                id="reasoning",
                category="Logical Reasoning",
                prompt="A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with just the number.",
                verify=lambda r: "9" in r.strip().split("\n")[-1],
            ),
            TestCase(
                id="math",
                category="Arithmetic",
                prompt="What is 247 * 83? Show your work, then give the final answer on the last line.",
                verify=lambda r: "20501" in r.replace(",", "").replace(" ", ""),
            ),
            TestCase(
                id="summarization",
                category="Summarization",
                prompt="Summarize blockchain technology in exactly 3 sentences. Be precise and technical.",
                verify=lambda r: (
                    len([s for s in r.replace("...", ".").split(".") if s.strip()]) >= 2
                    and any(w in r.lower() for w in ["block", "chain", "ledger", "decentrali", "hash", "distributed"])
                ),
            ),
            TestCase(
                id="instruction",
                category="Instruction Following",
                prompt="List exactly 5 programming languages that start with the letter P. Format: numbered 1-5, one per line. No other text.",
                verify=lambda r: (
                    sum(1 for line in r.strip().split("\n") if line.strip() and any(c.isalpha() for c in line)) >= 4
                    and any(w in r.lower() for w in ["python", "perl", "php", "pascal", "prolog"])
                ),
            ),
            TestCase(
                id="json_output",
                category="Structured Output",
                prompt='Return a valid JSON object with keys "name", "age", and "city" for a fictional person. Output ONLY the JSON, nothing else.',
                verify=lambda r: "{" in r and "}" in r and '"name"' in r and '"age"' in r and '"city"' in r,
            ),
            TestCase(
                id="translation",
                category="Translation",
                prompt="Translate 'The quick brown fox jumps over the lazy dog' into Spanish. Output only the translation.",
                verify=lambda r: any(w in r.lower() for w in ["zorro", "rapido", "perro", "perezoso", "salta"]),
            ),
            TestCase(
                id="creative",
                category="Creative Writing",
                prompt="Write a haiku about artificial intelligence. Follow the 5-7-5 syllable structure.",
                verify=lambda r: len(r.strip().split("\n")) >= 3 and len(r.strip()) > 20,
            ),
        ]
