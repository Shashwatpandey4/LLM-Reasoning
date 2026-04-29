"""Unified config-driven experiment runner.

Usage:
    uv run python benchmarks/run_experiment.py --config configs/gsm8k_self_consistency.yaml

To run a different model, override via CLI:
    uv run python benchmarks/run_experiment.py \
        --config configs/gsm8k_self_consistency.yaml --model gemma-3-4b-it

Any list-valued parameter under `method:` triggers a sweep over those values.
"""

import argparse
import os
import sys

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Side-effect imports: register all components into global registries
import src.common.datasets.gsm8k  # noqa: F401
import src.common.datasets.logiqa  # noqa: F401
import src.common.models.huggingface  # noqa: F401
import src.common.parsing.gsm8k  # noqa: F401
import src.common.parsing.mcq  # noqa: F401
import src.common.prompts.gsm8k_cot  # noqa: F401
import src.common.prompts.mcq_cot  # noqa: F401
from src.registry import DATASETS, MODELS, PARSERS, PROMPTS


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_PROMPT_KEY_OVERRIDES = {"logiqa": "mcq_cot"}
_PARSER_KEY_OVERRIDES = {"logiqa": "mcq"}


def _prompt_key(dataset: str) -> str:
    return _PROMPT_KEY_OVERRIDES.get(dataset, f"{dataset}_cot")


def _parser_key(dataset: str) -> str:
    return _PARSER_KEY_OVERRIDES.get(dataset, f"{dataset}_parser")


def _run_single_cot(config: dict, instances: list, prompt_template, parser, model_name: str):
    from src.experiments.gsm8k_single_cot_sweep import run_gsm8k_single_cot_sweep

    method_cfg = config["method"]
    budgets = method_cfg["max_new_tokens"]
    if not isinstance(budgets, list):
        budgets = [budgets]

    temperature = config["model"].get("temperature", 0.0)

    def model_factory():
        return MODELS.get(model_name)(temperature=temperature)

    return run_gsm8k_single_cot_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=prompt_template,
        parser=parser,
        budgets=budgets,
        experiment_name=config["experiment"]["name"],
        base_dir=config["experiment"]["base_dir"],
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
    )


def _run_self_consistency(config: dict, instances: list, prompt_template, parser, model_name: str):
    from src.experiments.gsm8k_self_consistency_sweep import run_gsm8k_self_consistency_sweep

    method_cfg = config["method"]
    k_values = method_cfg["k"]
    if not isinstance(k_values, list):
        k_values = [k_values]

    max_new_tokens = method_cfg.get("max_new_tokens", 512)
    temperature = config["model"].get("temperature", 0.7)

    def model_factory():
        return MODELS.get(model_name)(temperature=temperature)

    return run_gsm8k_self_consistency_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=prompt_template,
        parser=parser,
        k_values=k_values,
        max_new_tokens=max_new_tokens,
        experiment_name=config["experiment"]["name"],
        base_dir=config["experiment"]["base_dir"],
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
    )


def _run_ear(config: dict, instances: list, prompt_template, parser, model_name: str):
    # Side-effect imports: register EAR components into global registries
    import src.common.parsing.ear_judge  # noqa: F401
    import src.common.prompts.ear_prompts  # noqa: F401
    import src.methods.ear.critique.answer_level  # noqa: F401
    import src.methods.ear.critique.panel  # noqa: F401
    import src.methods.ear.critique.reasoning_level  # noqa: F401
    import src.methods.ear.equilibrium.elo  # noqa: F401
    import src.methods.ear.equilibrium.nash  # noqa: F401
    import src.methods.ear.equilibrium.survival  # noqa: F401
    from src.experiments.gsm8k_ear_sweep import build_ear_configs, run_gsm8k_ear_sweep

    method_cfg = config["method"]
    temperature = config["model"].get("temperature", 0.7)
    max_new_tokens = method_cfg.get("max_new_tokens", 512)
    if isinstance(max_new_tokens, list):
        max_new_tokens = max_new_tokens[0]  # EAR uses a single fixed budget per call

    def _as_list(v):
        return v if isinstance(v, list) else [v]

    ear_configs = build_ear_configs(
        k_values=_as_list(method_cfg["k"]),
        rounds_values=_as_list(method_cfg["rounds"]),
        critique_strategies=_as_list(method_cfg["critique_strategy"]),
        equilibrium_selectors=_as_list(method_cfg["equilibrium_selector"]),
        allow_revision_values=_as_list(method_cfg["allow_revision"]),
        max_new_tokens=max_new_tokens,
    )

    _CRITIQUE_PROMPT_KEYS = {
        "answer_level": "ear_answer_level_critique",
        "reasoning_level": "ear_reasoning_level_critique",
        "panel": "ear_panel_critique",
    }
    needed_strategies = {cfg.critique_strategy for cfg in ear_configs}
    critique_prompts = {s: PROMPTS.get(_CRITIQUE_PROMPT_KEYS[s])() for s in needed_strategies}
    judge_prompt = PROMPTS.get("ear_judge")()
    revision_prompt = PROMPTS.get("ear_revision")()
    judge_parser = PARSERS.get("ear_judge_parser")()

    def model_factory():
        return MODELS.get(model_name)(temperature=temperature)

    print(f"  EAR grid   : {len(ear_configs)} configurations")

    return run_gsm8k_ear_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=prompt_template,
        parser=parser,
        critique_prompts=critique_prompts,
        judge_prompt=judge_prompt,
        judge_parser=judge_parser,
        revision_prompt=revision_prompt,
        ear_configs=ear_configs,
        experiment_name=config["experiment"]["name"],
        base_dir=config["experiment"]["base_dir"],
        dataset_name=config["dataset"]["name"],
        split=config["dataset"]["split"],
    )


_DISPATCHERS = {
    "single_cot": _run_single_cot,
    "self_consistency": _run_self_consistency,
    "ear": _run_ear,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run a benchmarking experiment from a YAML config."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model name from the config (must be registered in src/common/models/).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=None,
        help="Override num_examples from the config. 0 = full split.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.model is not None:
        original_model = config["model"]["name"]
        config["model"]["name"] = args.model
        exp_name = config["experiment"]["name"]
        if original_model in exp_name:
            config["experiment"]["name"] = exp_name.replace(original_model, args.model)
        else:
            config["experiment"]["name"] = f"{exp_name}_{args.model}"
    if args.num_examples is not None:
        config["dataset"]["num_examples"] = args.num_examples

    model_name = config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    split = config["dataset"]["split"]
    num_examples = config["dataset"].get("num_examples", 0)
    method_name = config["method"]["name"]

    if method_name not in _DISPATCHERS:
        raise ValueError(f"Unknown method '{method_name}'. Available: {list(_DISPATCHERS.keys())}")

    # Load data
    dataset = DATASETS.get(dataset_name)(split=split)
    instances = dataset.get_data()
    if num_examples and num_examples > 0:
        instances = instances[:num_examples]

    prompt_template = PROMPTS.get(_prompt_key(dataset_name))()
    answer_parser = PARSERS.get(_parser_key(dataset_name))()

    print(f"Running experiment: {config['experiment']['name']}")
    print(f"  model      : {model_name}")
    print(f"  dataset    : {dataset_name} / {split} ({len(instances)} examples)")
    print(f"  method     : {method_name}")

    result = _DISPATCHERS[method_name](
        config, instances, prompt_template, answer_parser, model_name
    )

    print("\nExperiment complete.")
    print(f"Plot manifest: {result.get('plot_manifest_path')}")


if __name__ == "__main__":
    main()
