"""Bridge to lm-eval-harness for standardized academic benchmarks."""
from __future__ import annotations


# HF tokenizer mapping for Ollama model names
OLLAMA_TO_HF_TOKENIZER = {
    "nemotron-3-nano": "nvidia/Nemotron-3-Nano-4B-Instruct",
    "nemotron-mini": "nvidia/Nemotron-Mini-4B-Instruct",
    "gemma3": "google/gemma-3-4b-it",
    "gemma3n": "google/gemma-3n-E4B-it",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "phi4-mini": "microsoft/phi-4-mini-instruct",
    "phi4": "microsoft/phi-4",
    "granite3.3": "ibm-granite/granite-3.3-2b-instruct",
    "qwen3.5": "Qwen/Qwen3.5-4B",
    "qwen3": "Qwen/Qwen3-14B",
    "qwen2.5": "Qwen/Qwen2.5-14B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "glm4": "THUDM/glm-4-9b-chat",
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
}

TASK_PRESETS = {
    "quick":    ["hellaswag", "arc_easy"],
    "standard": ["mmlu", "gsm8k", "hellaswag", "arc_challenge"],
    "full":     ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "arc_easy", "winogrande", "truthfulqa"],
}


def get_hf_tokenizer(model_name: str) -> str | None:
    """Map Ollama model name to HuggingFace tokenizer repo."""
    base = model_name.split(":")[0]
    return OLLAMA_TO_HF_TOKENIZER.get(base)


def run_lm_eval(
    model_name: str,
    tasks: list[str] | None = None,
    preset: str = "standard",
    limit: int | None = 100,
    num_fewshot: int = 5,
    base_url: str = "http://localhost:11434/v1",
) -> dict:
    """Run lm-eval-harness against an Ollama model via OpenAI-compat API.

    Requires: pip install lm-eval[api] datasets transformers
    """
    try:
        from lm_eval import evaluator
    except ImportError:
        return {
            "error": "lm-eval not installed. Run: pip install 'lm-eval[api]' datasets transformers",
            "results": {},
        }

    if tasks is None:
        tasks = TASK_PRESETS.get(preset, TASK_PRESETS["standard"])

    # Build model_args string
    tokenizer = get_hf_tokenizer(model_name)
    model_args = f"model={model_name},base_url={base_url},num_concurrent=1"
    if tokenizer:
        model_args += f",tokenizer_backend=huggingface"

    try:
        results = evaluator.simple_evaluate(
            model="local-chat-completions",
            model_args=model_args,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            batch_size=1,
        )

        # Extract per-task metrics
        parsed = {}
        if "results" in results:
            for task_name, task_data in results["results"].items():
                parsed[task_name] = {
                    "accuracy": task_data.get("acc,none") or task_data.get("acc_norm,none"),
                    "accuracy_stderr": task_data.get("acc_stderr,none") or task_data.get("acc_norm_stderr,none"),
                    "num_samples": limit or 0,
                }

        return {"results": parsed, "error": None}

    except Exception as e:
        return {"error": str(e)[:300], "results": {}}
