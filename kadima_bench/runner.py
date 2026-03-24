"""Runner — orchestrates benchmark execution with GPU isolation and metrics."""
from __future__ import annotations

import datetime
import json
import os
import sys
import time

import numpy as np

from kadima_bench.backends.ollama import OllamaBackend
from kadima_bench.config import RunConfig, KNOWN_MODELS, CHAT_MODELS
from kadima_bench.metrics.gpu_monitor import GpuMonitor
from kadima_bench.metrics.speed import compute_speed_metrics, merge_speed_metrics
from kadima_bench.metrics.aggregator import compute_composite_score, find_pareto_frontier
from kadima_bench.suites.kadima_custom import KadimaCustomSuite

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Speed test prompts for streaming latency measurement
SPEED_PROMPTS = [
    "Explain the concept of recursion in programming in 2-3 sentences.",
    "What are the three laws of thermodynamics? Brief answer.",
    "Write a short Python function that reverses a string.",
]


def resolve_models(config: RunConfig, backend: OllamaBackend) -> list[dict]:
    """Resolve model list: use config.models if specified, else auto-discover."""
    available = backend.list_models()
    available_names = {m["name"] for m in available}
    size_map = {m["name"]: m["size_gb"] for m in available}

    if config.models:
        # Use specified models, filtered to what's available
        target_names = config.models
    else:
        # Auto-discover: all models under 16GB
        target_names = [m["name"] for m in available if m["size_gb"] <= 16]

    resolved = []
    for name in target_names:
        # Match exact or prefix
        matched = None
        for avail_name in available_names:
            if name == avail_name or avail_name.startswith(name.split(":")[0]):
                if name == avail_name:
                    matched = name
                    break
                if matched is None:
                    matched = avail_name

        if not matched:
            print(f"  [--] {name} (not installed, skipping)")
            continue

        # Build model info from known metadata or Ollama API
        known = KNOWN_MODELS.get(matched, {})
        info = {
            "name": matched,
            "label": known.get("label", matched),
            "family": known.get("family", "Unknown"),
            "params": known.get("params", "?"),
            "quant": known.get("quant", "?"),
            "size_gb": size_map.get(matched, 0.0),
            "use_chat": matched in CHAT_MODELS,
            "think_off": matched in CHAT_MODELS,
        }

        # Apply overrides from config
        if matched in config.model_overrides:
            override = config.model_overrides[matched]
            if override.label:
                info["label"] = override.label
            if override.family:
                info["family"] = override.family
            info["use_chat"] = override.use_chat_api
            info["think_off"] = override.think_off

        resolved.append(info)
        print(f"  [OK] {info['label']} ({info['params']} | {info['quant']} | {info['size_gb']}GB)")

    return resolved


def run_custom_tests(backend: OllamaBackend, model_info: dict, suite: KadimaCustomSuite) -> list[dict]:
    """Run custom quality tests (non-streaming)."""
    tests = suite.get_tests()
    results = []
    use_chat = model_info.get("use_chat", False)
    think_off = model_info.get("think_off", False)

    for i, test in enumerate(tests):
        print(f"    [{i+1}/{len(tests)}] {test.category:.<25s}", end=" ", flush=True)

        result = backend.generate(
            model_info["name"], test.prompt,
            use_chat=use_chat, think_off=think_off,
        )

        passed = False
        if result.success and result.text and test.verify:
            try:
                passed = test.verify(result.text)
            except Exception:
                passed = False

        status = "PASS" if passed else "FAIL"
        print(f"{status}  {result.time_seconds:>5.1f}s  {result.tokens_per_second:>6.1f} t/s")

        results.append({
            "test_id": test.id,
            "category": test.category,
            "passed": passed,
            "time_seconds": result.time_seconds,
            "tokens": result.tokens,
            "tokens_per_second": result.tokens_per_second,
        })

    return results


def run_speed_tests(backend: OllamaBackend, model_info: dict,
                    repeats: int = 3) -> dict:
    """Run streaming speed tests for TTFT/ITL/percentiles."""
    all_ttft = []
    all_itl = []
    total_tokens = 0
    total_time = 0.0

    use_chat = model_info.get("use_chat", False)
    think_off = model_info.get("think_off", False)

    for rep in range(repeats):
        for prompt in SPEED_PROMPTS:
            result = backend.generate_stream(
                model_info["name"], prompt,
                use_chat=use_chat, think_off=think_off,
            )
            if result.success and result.tokens > 0:
                all_ttft.append(result.ttft_ms)
                all_itl.extend(result.itl_ms)
                total_tokens += result.tokens
                total_time += result.time_seconds

    if not all_ttft:
        return {"speed_metrics": None}

    arr_ttft = np.array(all_ttft)
    arr_itl = np.array(all_itl) if all_itl else np.array([0.0])

    return {
        "speed_metrics": {
            "ttft_ms": round(float(np.mean(arr_ttft)), 1),
            "ttft_p50_ms": round(float(np.percentile(arr_ttft, 50)), 1),
            "ttft_p95_ms": round(float(np.percentile(arr_ttft, 95)), 1),
            "itl_mean_ms": round(float(np.mean(arr_itl)), 2),
            "itl_p50_ms": round(float(np.percentile(arr_itl, 50)), 2),
            "itl_p95_ms": round(float(np.percentile(arr_itl, 95)), 2),
            "itl_p99_ms": round(float(np.percentile(arr_itl, 99)), 2),
            "tpot_ms": round(float(np.mean(arr_itl)), 2),
            "throughput_tps": round(total_tokens / total_time, 1) if total_time > 0 else 0.0,
            "total_tokens": total_tokens,
            "total_time_s": round(total_time, 3),
            "runs": repeats * len(SPEED_PROMPTS),
        }
    }


def benchmark_model(backend: OllamaBackend, model_info: dict,
                    config: RunConfig, gpu_monitor: GpuMonitor | None) -> dict:
    """Benchmark a single model: quality + speed + GPU metrics."""
    label = model_info["label"]

    print(f"\n{'='*60}")
    print(f"  {label}  ({model_info['params']} | {model_info['quant']} | {model_info['size_gb']}GB)")
    print(f"  Family: {model_info['family']}  |  Model: {model_info['name']}")
    print(f"{'='*60}")

    # GPU isolation
    if config.gpu_isolation:
        print(f"  Clearing VRAM...", end=" ", flush=True)
        backend.unload_all()
        print("done.")

    # Warmup
    print(f"  Loading model...", end=" ", flush=True)
    backend.load_model(model_info["name"], use_chat=model_info.get("use_chat", False))
    print("done.")

    # Start GPU monitoring
    if gpu_monitor:
        gpu_monitor.start()

    result = {
        "model": model_info["name"],
        "label": label,
        "family": model_info["family"],
        "params": model_info["params"],
        "quantization": model_info["quant"],
        "model_size_gb": model_info["size_gb"],
    }

    # 1. Custom quality tests
    if config.suite.custom_tests:
        print(f"  --- Custom Tests ---")
        suite = KadimaCustomSuite()
        test_results = run_custom_tests(backend, model_info, suite)
        correct = sum(1 for t in test_results if t["passed"])
        total = len(test_results)
        acc = round(correct / total * 100, 1) if total > 0 else 0.0

        tps_values = [t["tokens_per_second"] for t in test_results if t["tokens_per_second"] > 0]
        avg_tps = sum(tps_values) / len(tps_values) if tps_values else 0.0
        total_time = sum(t["time_seconds"] for t in test_results)

        result.update({
            "tests_passed": correct,
            "tests_total": total,
            "accuracy_pct": acc,
            "avg_tokens_per_second": round(avg_tps, 1),
            "avg_response_time": round(total_time / total, 2) if total > 0 else 0.0,
            "total_time_seconds": round(total_time, 2),
            "test_results": test_results,
        })

        print(f"  Score: {correct}/{total} ({acc}%)  |  Avg: {avg_tps:.0f} t/s  |  Total: {total_time:.1f}s")

    # 2. Speed tests (streaming)
    if config.suite.speed_tests:
        print(f"  --- Speed Tests ({config.suite.speed_repeats}x{len(SPEED_PROMPTS)} runs) ---", end=" ", flush=True)
        speed_data = run_speed_tests(backend, model_info, repeats=config.suite.speed_repeats)
        if speed_data["speed_metrics"]:
            sm = speed_data["speed_metrics"]
            print(f"TTFT: {sm['ttft_ms']:.0f}ms | ITL p50: {sm['itl_p50_ms']:.1f}ms | p95: {sm['itl_p95_ms']:.1f}ms | {sm['throughput_tps']:.0f} t/s")
        else:
            print("no data")
        result["speed_metrics"] = speed_data["speed_metrics"]

    # Stop GPU monitoring
    gpu_snapshot = None
    if gpu_monitor:
        gpu_snapshot = gpu_monitor.stop()
        result["gpu_snapshot"] = gpu_snapshot.to_dict()
        print(f"  GPU: peak {gpu_snapshot.peak_vram_mb:.0f}MB VRAM | avg {gpu_snapshot.avg_power_w:.0f}W")

    # Efficiency metrics
    size_gb = model_info["size_gb"]
    tps = result.get("avg_tokens_per_second", 0)
    result["efficiency_tps_per_gb"] = round(tps / size_gb, 1) if size_gb > 0 else 0.0

    if gpu_snapshot and gpu_snapshot.avg_power_w > 0 and tps > 0:
        # Energy per token = avg_power / throughput = watts / (tokens/s) = joules/token → mJ
        result["energy_per_token_mj"] = round(gpu_snapshot.avg_power_w / tps * 1000, 1)

    return result


def run_benchmark(config: RunConfig) -> str:
    """Execute full benchmark suite. Returns path to output JSON."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output directory
    output_dir = os.path.join(config.output_dir, f"kadima_bench_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"kadima_bench_{timestamp}.json")

    # Initialize backend
    backend = OllamaBackend(base_url=config.backend.base_url, timeout=config.backend.timeout)

    # Resolve models
    print("Checking model availability...")
    models = resolve_models(config, backend)

    if not models:
        print("ERROR: No models available!")
        sys.exit(1)

    hw = config.hardware
    print(f"\n{'='*60}")
    print(f"  KADIMA DIGITAL LABORATORIES")
    print(f"  kadima-bench v{config.version}")
    print(f"{'='*60}")
    print(f"  {hw.cpu}  |  {hw.gpu}")
    print(f"  {hw.ram}  |  {hw.motherboard}")
    print(f"  {hw.os}  |  {hw.inference_engine}")
    print(f"  {datetime.datetime.now().strftime('%B %d, %Y %I:%M %p')}")
    print(f"  Models: {len(models)}  |  Suite: {config.suite.preset}")
    print(f"  Custom tests: {'Yes' if config.suite.custom_tests else 'No'}")
    print(f"  Speed tests: {'Yes' if config.suite.speed_tests else 'No'} ({config.suite.speed_repeats}x)")
    print(f"  lm-eval: {'Yes' if config.suite.lm_eval else 'No'}")
    print(f"  GPU monitoring: {'Yes' if config.gpu_monitor else 'No'}")
    print(f"  GPU isolation: {'Yes' if config.gpu_isolation else 'No'}")
    print(f"{'='*60}")

    # Run benchmarks
    all_results = []
    gpu_mon = GpuMonitor() if config.gpu_monitor else None

    for i, model in enumerate(models):
        print(f"\n>>> [{i+1}/{len(models)}]")
        try:
            result = benchmark_model(backend, model, config, gpu_mon)
            all_results.append(result)
        except Exception as e:
            print(f"  [ERROR] {model['label']}: {e}")

    # Final VRAM cleanup
    backend.unload_all()

    # Compute cross-model analytics
    if all_results:
        max_tps = max(r.get("avg_tokens_per_second", 0) for r in all_results)
        max_eff = max(r.get("efficiency_tps_per_gb", 0) for r in all_results)
        weights = (config.scoring.quality_weight, config.scoring.speed_weight, config.scoring.efficiency_weight)

        for r in all_results:
            r["composite_score"] = compute_composite_score(
                r.get("accuracy_pct", 0),
                r.get("avg_tokens_per_second", 0),
                max_tps,
                r.get("efficiency_tps_per_gb", 0),
                max_eff,
                weights=weights,
            )

        pareto = find_pareto_frontier(all_results)
        for r in all_results:
            r["pareto_optimal"] = r["label"] in pareto
    else:
        pareto = []

    # Sort by composite score descending
    all_results.sort(key=lambda x: -x.get("composite_score", 0))

    # Build output
    suite_obj = KadimaCustomSuite()
    categories = [t.category for t in suite_obj.get_tests()] if config.suite.custom_tests else []

    output = {
        "metadata": {
            "lab": hw.lab,
            "hardware": {
                "cpu": hw.cpu, "gpu": hw.gpu, "ram": hw.ram,
                "motherboard": hw.motherboard, "storage": hw.storage,
                "os": hw.os, "inference_engine": hw.inference_engine,
            },
            "version": config.version,
            "timestamp": timestamp,
            "date": datetime.datetime.now().isoformat(),
            "models_tested": len(all_results),
            "tests_per_model": len(categories),
            "categories": categories,
            "methodology": (
                "GPU-isolated: VRAM cleared between models. "
                "Streaming for latency metrics (TTFT, ITL percentiles). "
                "nvidia-smi polling for VRAM/power tracking. "
                "512 token output cap."
            ),
            "suite_preset": config.suite.preset,
            "scoring_weights": {
                "quality": config.scoring.quality_weight,
                "speed": config.scoring.speed_weight,
                "efficiency": config.scoring.efficiency_weight,
            },
        },
        "results": all_results,
        "pareto_frontier": pareto,
        "rankings": {
            "by_composite": [r["label"] for r in all_results],
            "by_accuracy": [r["label"] for r in sorted(all_results, key=lambda x: -x.get("accuracy_pct", 0))],
            "by_speed": [r["label"] for r in sorted(all_results, key=lambda x: -x.get("avg_tokens_per_second", 0))],
            "by_efficiency": [r["label"] for r in sorted(all_results, key=lambda x: -x.get("efficiency_tps_per_gb", 0))],
        },
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print leaderboard
    print(f"\n{'='*60}")
    print(f"  LEADERBOARD (by composite score)")
    print(f"{'='*60}")
    print(f"  {'#':<3} {'Model':<25} {'Score':>6} {'Acc':>5} {'Speed':>7} {'Eff':>7} {'Pareto':>7}")
    print(f"  {'-'*3} {'-'*25} {'-'*6} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

    for rank, r in enumerate(all_results, 1):
        p = "*" if r.get("pareto_optimal") else " "
        print(
            f"  {rank:<3} {r['label']:<25} "
            f"{r.get('composite_score', 0):>5.1f} "
            f"{r.get('accuracy_pct', 0):>4.0f}% "
            f"{r.get('avg_tokens_per_second', 0):>5.0f}t/s "
            f"{r.get('efficiency_tps_per_gb', 0):>5.1f}/GB "
            f"{p:>6}"
        )

    print(f"{'='*60}")
    print(f"  Saved: {output_file}")
    print(f"  Pareto frontier: {', '.join(pareto)}")

    return output_file
