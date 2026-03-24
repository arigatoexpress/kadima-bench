"""Ollama backend — primary inference engine for kadima-bench."""
from __future__ import annotations

import json
import subprocess
import time
import urllib.request
import urllib.error
from typing import Iterator

from kadima_bench.backends.base import (
    Backend, GenerateResult, StreamChunk, StreamResult,
)


class OllamaBackend(Backend):
    """Ollama REST API backend with streaming support."""

    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- Core API helpers -----------------------------------------------------

    def _api_call(self, endpoint: str, body: dict) -> dict:
        """Make a POST request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _api_get(self, endpoint: str) -> dict:
        """Make a GET request to Ollama API."""
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _build_body(self, model: str, prompt: str, *,
                    max_tokens: int, temperature: float,
                    use_chat: bool, think_off: bool,
                    stream: bool) -> tuple[str, dict]:
        """Build request body and return (endpoint, body)."""
        if use_chat:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }
            if think_off:
                body["think"] = False
            return "/api/chat", body
        else:
            return "/api/generate", {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {"num_predict": max_tokens, "temperature": temperature},
            }

    # -- Non-streaming generation ---------------------------------------------

    def generate(self, model: str, prompt: str, *,
                 max_tokens: int = 512, temperature: float = 0.0,
                 use_chat: bool = False, think_off: bool = False) -> GenerateResult:
        start = time.perf_counter()
        try:
            endpoint, body = self._build_body(
                model, prompt, max_tokens=max_tokens, temperature=temperature,
                use_chat=use_chat, think_off=think_off, stream=False,
            )
            data = self._api_call(endpoint, body)
            elapsed = time.perf_counter() - start

            text = (data.get("message", {}).get("content", "").strip()
                    if use_chat else data.get("response", "").strip())

            eval_count = data.get("eval_count", 0)
            eval_duration_ns = data.get("eval_duration", 0)
            prompt_tokens = data.get("prompt_eval_count", 0)

            if eval_duration_ns > 0 and eval_count > 0:
                tps = eval_count / (eval_duration_ns / 1e9)
            elif elapsed > 0:
                eval_count = max(len(text) // 4, 1)
                tps = eval_count / elapsed
            else:
                tps = 0.0

            return GenerateResult(
                text=text, tokens=eval_count, prompt_tokens=prompt_tokens,
                time_seconds=round(elapsed, 3), eval_duration_ns=eval_duration_ns,
                tokens_per_second=round(tps, 1), success=True,
            )
        except Exception as e:
            return GenerateResult(
                text="", tokens=0, prompt_tokens=0,
                time_seconds=round(time.perf_counter() - start, 3),
                eval_duration_ns=0, tokens_per_second=0.0,
                success=False, error=str(e)[:200],
            )

    # -- Streaming generation (for TTFT / ITL) --------------------------------

    def generate_stream(self, model: str, prompt: str, *,
                        max_tokens: int = 512, temperature: float = 0.0,
                        use_chat: bool = False, think_off: bool = False) -> StreamResult:
        start_ns = time.perf_counter_ns()
        try:
            endpoint, body = self._build_body(
                model, prompt, max_tokens=max_tokens, temperature=temperature,
                use_chat=use_chat, think_off=think_off, stream=True,
            )
            url = f"{self.base_url}{endpoint}"
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=self.timeout)

            chunks: list[StreamChunk] = []
            full_text = []
            ttft_ns: int | None = None

            # Read NDJSON stream
            for line in resp:
                ts = time.perf_counter_ns()
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                data = json.loads(line)

                if use_chat:
                    token = data.get("message", {}).get("content", "")
                else:
                    token = data.get("response", "")

                if token:
                    if ttft_ns is None:
                        ttft_ns = ts - start_ns
                    chunks.append(StreamChunk(token=token, timestamp_ns=ts))
                    full_text.append(token)

                if data.get("done", False):
                    break

            resp.close()
            end_ns = time.perf_counter_ns()

            # Compute inter-token latencies
            itl_ms = []
            for i in range(1, len(chunks)):
                delta_ms = (chunks[i].timestamp_ns - chunks[i - 1].timestamp_ns) / 1e6
                itl_ms.append(round(delta_ms, 2))

            total_time = (end_ns - start_ns) / 1e9
            total_tokens = len(chunks)
            tps = total_tokens / total_time if total_time > 0 else 0.0

            return StreamResult(
                text="".join(full_text).strip(),
                tokens=total_tokens,
                time_seconds=round(total_time, 3),
                tokens_per_second=round(tps, 1),
                ttft_ms=round((ttft_ns or 0) / 1e6, 1),
                itl_ms=itl_ms,
                success=True,
            )
        except Exception as e:
            elapsed = (time.perf_counter_ns() - start_ns) / 1e9
            return StreamResult(
                text="", tokens=0, time_seconds=round(elapsed, 3),
                tokens_per_second=0.0, ttft_ms=0.0, itl_ms=[],
                success=False, error=str(e)[:200],
            )

    # -- Model management -----------------------------------------------------

    def list_models(self) -> list[dict]:
        """List all locally available Ollama models."""
        try:
            data = self._api_get("/api/tags")
            models = []
            for m in data.get("models", []):
                size_bytes = m.get("size", 0)
                models.append({
                    "name": m["name"],
                    "size_gb": round(size_bytes / (1024**3), 1),
                    "family": m.get("details", {}).get("family", "unknown"),
                    "params": m.get("details", {}).get("parameter_size", "?"),
                    "quantization": m.get("details", {}).get("quantization_level", "?"),
                    "modified": m.get("modified_at", ""),
                })
            return models
        except Exception:
            return []

    def load_model(self, model: str, *, use_chat: bool = False) -> None:
        """Warmup: load model into VRAM with a trivial prompt."""
        try:
            if use_chat:
                self._api_call("/api/chat", {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                    "think": False,
                    "options": {"num_predict": 16},
                })
            else:
                self._api_call("/api/generate", {
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 16},
                })
            time.sleep(1)
        except Exception:
            pass

    def unload_all(self) -> None:
        """Unload all models from Ollama to clear VRAM."""
        try:
            result = subprocess.run(
                ["ollama", "ps"], capture_output=True,
                encoding="utf-8", errors="replace", timeout=10,
            )
            for line in result.stdout.strip().split("\n")[1:]:
                model_name = line.split()[0] if line.strip() else None
                if model_name:
                    subprocess.run(
                        ["ollama", "stop", model_name],
                        capture_output=True, encoding="utf-8",
                        errors="replace", timeout=30,
                    )
            time.sleep(3)
        except Exception:
            pass
