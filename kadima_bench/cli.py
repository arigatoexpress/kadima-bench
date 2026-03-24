"""CLI interface for kadima-bench."""
from __future__ import annotations

import sys
import click

from kadima_bench import __version__


@click.group()
@click.version_option(__version__, prog_name="kadima-bench")
def main():
    """Kadima Bench - Open-source local LLM benchmarking framework."""
    pass


@main.command()
@click.option("--models", "-m", default=None, help="Comma-separated model names (e.g., nemotron-3-nano:4b,phi4:14b). Default: auto-discover.")
@click.option("--suite", "-s", default="quick", type=click.Choice(["quick", "standard", "full", "speed-only", "custom"]), help="Test suite preset.")
@click.option("--output", "-o", default=".", help="Output directory for results.")
@click.option("--config", "-c", default=None, help="Path to TOML config file.")
@click.option("--no-gpu-monitor", is_flag=True, help="Disable GPU monitoring.")
@click.option("--no-isolation", is_flag=True, help="Disable GPU isolation between models.")
@click.option("--speed-repeats", default=3, help="Number of speed test repeats.")
@click.option("--lm-eval", is_flag=True, help="Enable lm-eval-harness standard benchmarks.")
@click.option("--lm-eval-limit", default=100, help="Sample limit per lm-eval task.")
def run(models, suite, output, config, no_gpu_monitor, no_isolation, speed_repeats, lm_eval, lm_eval_limit):
    """Run benchmark suite against local LLM models."""
    from kadima_bench.config import load_config, RunConfig
    from kadima_bench.runner import run_benchmark

    # Load config (file or defaults)
    cfg = load_config(config) if config else RunConfig()

    # CLI overrides
    if models:
        cfg.models = [m.strip() for m in models.split(",")]

    cfg.output_dir = output
    cfg.gpu_monitor = not no_gpu_monitor
    cfg.gpu_isolation = not no_isolation
    cfg.suite.speed_repeats = speed_repeats

    # Suite presets
    if suite == "quick":
        cfg.suite.preset = "quick"
        cfg.suite.custom_tests = True
        cfg.suite.speed_tests = True
        cfg.suite.lm_eval = lm_eval
    elif suite == "standard":
        cfg.suite.preset = "standard"
        cfg.suite.custom_tests = True
        cfg.suite.speed_tests = True
        cfg.suite.lm_eval = lm_eval
    elif suite == "full":
        cfg.suite.preset = "full"
        cfg.suite.custom_tests = True
        cfg.suite.speed_tests = True
        cfg.suite.lm_eval = True
        cfg.suite.lm_eval_limit = None
    elif suite == "speed-only":
        cfg.suite.preset = "speed-only"
        cfg.suite.custom_tests = False
        cfg.suite.speed_tests = True
        cfg.suite.lm_eval = False
    elif suite == "custom":
        cfg.suite.preset = "custom"

    if lm_eval:
        cfg.suite.lm_eval = True
    if lm_eval_limit:
        cfg.suite.lm_eval_limit = lm_eval_limit

    # Run
    output_file = run_benchmark(cfg)

    # Generate visualizations
    try:
        from kadima_bench.visualize.charts import generate_all_charts
        generate_all_charts(output_file)
    except Exception as e:
        print(f"\n  [WARN] Chart generation failed: {e}")
        print(f"  Run `kadima-bench report {output_file}` to retry.")


@main.command("list-models")
@click.option("--url", default="http://localhost:11434", help="Ollama base URL.")
def list_models(url):
    """List available Ollama models."""
    from kadima_bench.backends.ollama import OllamaBackend
    from kadima_bench.config import KNOWN_MODELS

    backend = OllamaBackend(base_url=url)
    models = backend.list_models()

    if not models:
        click.echo("No models found. Is Ollama running?")
        return

    click.echo(f"\n  {'Model':<35} {'Size':>6} {'Family':<12} {'Known':>5}")
    click.echo(f"  {'-'*35} {'-'*6} {'-'*12} {'-'*5}")

    for m in sorted(models, key=lambda x: x["size_gb"]):
        known = "Yes" if m["name"] in KNOWN_MODELS else ""
        family = KNOWN_MODELS.get(m["name"], {}).get("family", m.get("family", "?"))
        click.echo(f"  {m['name']:<35} {m['size_gb']:>5.1f}G {family:<12} {known:>5}")

    click.echo(f"\n  Total: {len(models)} models")


@main.command()
@click.argument("results_file")
def report(results_file):
    """Regenerate charts and report from existing results JSON."""
    import json, os

    if not os.path.exists(results_file):
        click.echo(f"File not found: {results_file}")
        sys.exit(1)

    try:
        from kadima_bench.visualize.charts import generate_all_charts
        generate_all_charts(results_file)
        click.echo("Charts regenerated successfully.")
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
