# kadima-bench

A local LLM benchmarking framework for consumer GPUs. Runs models through custom quality tests, captures production-grade latency metrics via streaming, monitors GPU resources, and generates publication-ready visualizations — all from a single CLI command.

Built for people who want to know how small open-source models actually perform on their own hardware, not on a rented A100.

## What It Measures

| Dimension | Metrics |
|-----------|---------|
| **Quality** | 8 task categories (math, logic, code, translation, context, creative, instruction following, arithmetic) with pass/fail verification |
| **Speed** | Tokens/second, Time to First Token (TTFT), Inter-Token Latency percentiles (p50/p95/p99) via streaming |
| **Efficiency** | Tokens/s per GB model size, peak VRAM usage, average power draw (watts) |
| **Composite** | Weighted score (50% quality + 30% speed + 20% efficiency) with Pareto frontier analysis |

## Quick Start

```bash
# Install
git clone https://github.com/arigatoexpress/kadima-bench.git
cd kadima-bench
pip install -e .

# Make sure Ollama is running with your models
ollama list

# Run a quick benchmark
kadima-bench run --models nemotron-3-nano:4b,phi4-mini --suite quick

# Benchmark all available models
kadima-bench run --all-models --suite standard

# Regenerate charts from existing results
kadima-bench report results/your_results.json
```

## Requirements

- **Python** 3.11+
- **Ollama** running locally with models pulled
- **NVIDIA GPU** with `nvidia-smi` available (for VRAM/power monitoring)
- **OS**: Windows, Linux, or macOS (tested on Windows 11 + RTX 5070 Ti)

## How It Works

1. **GPU Isolation** — VRAM is cleared between every model (`ollama stop`) so results aren't contaminated by prior model residue
2. **Warmup** — Each model gets a warmup prompt to ensure full VRAM residency before timed tests begin
3. **Streaming Capture** — Tests run with Ollama's streaming API to measure per-token latency, not just aggregate throughput
4. **GPU Monitoring** — A background thread polls `nvidia-smi` every 500ms to capture peak VRAM and power draw
5. **Pareto Analysis** — Models that aren't dominated on speed+accuracy are flagged as Pareto-optimal

## Output

Each run produces:

```
results/
  kadima_bench_YYYYMMDD_HHMMSS.json    # Structured results (all metrics)
  kadima_1_leaderboard.png              # 3-column ranking
  kadima_2_efficiency_frontier.png      # Speed vs accuracy scatter + Pareto line
  kadima_3_pass_fail.png                # Which models fail which tests
  kadima_4_latency.png                  # TTFT + ITL percentile bars
  kadima_5_speed_heatmap.png            # Tokens/s per model per category
  kadima_6_composite.png                # Stacked score breakdown
  kadima_7_energy_vram.png              # Power consumption + VRAM footprint
```

## Sample Results (RTX 5070 Ti, March 2026)

| Model | Accuracy | Speed (t/s) | Size | Composite |
|-------|----------|-------------|------|-----------|
| Nemotron 3 Nano 4B | 100% | 208 | 2.6 GB | 88.8 |
| Nemotron Mini 4B | 63% | 265 | 2.5 GB | 81.2 |
| Gemma 3 4B | 88% | 214 | 3.1 GB | 81.0 |
| Nemotron 3 Nano Q8 | 100% | 155 | 3.9 GB | 75.1 |
| Gemma 3n E4B | 100% | 139 | 7.0 GB | 69.5 |
| Granite 3.3 2B | 100% | 62 | 1.4 GB | 65.3 |
| Phi-4 14B | 100% | 94 | 8.4 GB | 62.7 |
| Qwen 3.5 4B | 88% | 33 | 3.2 GB | 49.5 |
| Phi-4 Mini 3.8B | 75% | 28 | 2.3 GB | 43.0 |
| Llama 3.2 3B | 63% | 34 | 1.9 GB | 38.4 |

## Configuration

Create a `kadima-bench.toml` to customize:

```toml
[general]
lab_name = "Your Lab Name"
gpu_isolation = true
warmup = true
gpu_monitor = true

[models]
include = ["nemotron-3-nano:4b", "phi4:14b", "gemma3:4b"]

[suite]
preset = "standard"     # quick | standard | full
speed_repeats = 3       # repeat speed tests for statistical robustness

[scoring]
quality_weight = 0.5
speed_weight = 0.3
efficiency_weight = 0.2
```

## Project Structure

```
kadima-bench/
  kadima_bench/
    cli.py              # Click CLI entry point
    runner.py           # Orchestrator (GPU isolation, warmup, test sequencing)
    config.py           # TOML config loading
    backends/
      base.py           # Backend protocol (generate, stream, load/unload)
      ollama.py         # Ollama API backend (native + streaming)
    suites/
      base.py           # TestSuite ABC, TestCase/TestResult types
      kadima_custom.py  # 8 custom quality tests
      presets.py        # quick/standard/full definitions
    metrics/
      speed.py          # TTFT, ITL, TPOT, percentiles
      quality.py        # Pass/fail + composite quality scoring
      efficiency.py     # Tokens/s per GB, energy per token
      gpu_monitor.py    # nvidia-smi background polling
      aggregator.py     # Pareto frontier, composite scores, rankings
    visualize/
      theme.py          # Dark theme, brand colors, branding footer
      charts.py         # 7 publication-quality chart generators
    output/
      schema.py         # Pydantic models for results JSON
  configs/
    default.toml        # Default configuration
  pyproject.toml        # Package metadata, dependencies, CLI entry point
```

## Inspired By

This project draws from three excellent tools in the LLM evaluation space:

- **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** by EleutherAI — The standard for academic LLM benchmarks (MMLU, GSM8K, HellaSwag, etc.). kadima-bench integrates with it as an optional backend for standardized scoring.
- **[SGLang](https://github.com/sgl-project/sglang)** by the SGLang team — Production-grade inference serving with detailed latency metrics (TTFT, ITL, throughput under load). Inspired our streaming-based latency capture approach.
- **[Bench360](https://github.com/facebookresearch/bench360)** by Meta — Multi-dimensional evaluation across quality, speed, and efficiency simultaneously. Inspired our composite scoring and Pareto frontier analysis.

kadima-bench combines the practical quality testing of lm-eval, the production latency metrics of SGLang, and the multi-dimensional analysis of Bench360 into a single tool designed for consumer GPU hardware.

## License

MIT — use it however you want.

---

*Built by [Kadima Digital Laboratories](https://github.com/arigatoexpress)*
