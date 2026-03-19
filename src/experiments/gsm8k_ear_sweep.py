"""GSM8K EAR sweep harness.

Runs EAR for every combination in `ear_configs` (the pre-computed Cartesian
product of all experiment dimensions), logging per-instance JSONL and
per-config summary JSON for each, plus an aggregate JSON for the full sweep.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, Iterable, List

from tqdm import tqdm

from src.common.eval.metrics import answers_match, summarize_ear_run
from src.common.utils.storage import ExperimentTracker
from src.methods.EAR import EAR
from src.methods.ear.judge import Judge
from src.registry import CRITIQUE_STRATEGIES, EQUILIBRIUM_SELECTORS


@dataclass
class EarRunConfig:
    k: int
    rounds: int
    critique_strategy: str    # registry key
    equilibrium_selector: str  # registry key
    allow_revision: bool
    max_new_tokens: int

    @property
    def label(self) -> str:
        rev = "rev" if self.allow_revision else "norev"
        return (
            f"k{self.k}_r{self.rounds}"
            f"_{self.critique_strategy}"
            f"_{self.equilibrium_selector}"
            f"_{rev}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "rounds": self.rounds,
            "critique_strategy": self.critique_strategy,
            "equilibrium_selector": self.equilibrium_selector,
            "allow_revision": self.allow_revision,
            "max_new_tokens": self.max_new_tokens,
        }


def build_ear_configs(
    k_values: List[int],
    rounds_values: List[int],
    critique_strategies: List[str],
    equilibrium_selectors: List[str],
    allow_revision_values: List[bool],
    max_new_tokens: int,
) -> List[EarRunConfig]:
    """Build the full Cartesian product of EAR experiment dimensions."""
    return [
        EarRunConfig(
            k=k,
            rounds=r,
            critique_strategy=cs,
            equilibrium_selector=es,
            allow_revision=rev,
            max_new_tokens=max_new_tokens,
        )
        for k, r, cs, es, rev in product(
            k_values,
            rounds_values,
            critique_strategies,
            equilibrium_selectors,
            allow_revision_values,
        )
    ]


def _build_ear_method(
    cfg: EarRunConfig,
    model: Any,
    prompt_template: Any,
    parser: Any,
    critique_prompts: Dict[str, Any],
    judge_prompt: Any,
    judge_parser: Any,
    revision_prompt: Any,
) -> EAR:
    critique_strategy = CRITIQUE_STRATEGIES.get(cfg.critique_strategy)(
        critique_prompt=critique_prompts[cfg.critique_strategy]
    )
    equilibrium_selector = EQUILIBRIUM_SELECTORS.get(cfg.equilibrium_selector)()
    judge = Judge(model=model, judge_prompt=judge_prompt, judge_parser=judge_parser)
    return EAR(
        model=model,
        prompt_template=prompt_template,
        parser=parser,
        critique_strategy=critique_strategy,
        equilibrium_selector=equilibrium_selector,
        judge=judge,
        revision_prompt=revision_prompt,
        k=cfg.k,
        rounds=cfg.rounds,
        allow_revision=cfg.allow_revision,
    )


def evaluate_ear_config(
    instances: List[Dict[str, Any]],
    cfg: EarRunConfig,
    model: Any,
    prompt_template: Any,
    parser: Any,
    critique_prompts: Dict[str, Any],
    judge_prompt: Any,
    judge_parser: Any,
    revision_prompt: Any,
    tracker: ExperimentTracker,
    progress: bool = True,
) -> Dict[str, Any]:
    method = _build_ear_method(
        cfg, model, prompt_template, parser,
        critique_prompts, judge_prompt, judge_parser, revision_prompt,
    )
    records: List[Dict[str, Any]] = []

    iterator = tqdm(instances, desc=cfg.label, leave=False) if progress else instances
    for instance in iterator:
        result = method.run(instance, max_new_tokens=cfg.max_new_tokens)
        result["is_correct"] = answers_match(
            result.get("extracted_answer"), result.get("ground_truth")
        )
        tracker.log_instance(result)
        records.append(result)

    summary = summarize_ear_run(records)
    summary.update(cfg.to_dict())
    summary["raw_path"] = tracker.raw_filepath
    tracker.save_summary(summary)
    return summary


def run_gsm8k_ear_sweep(
    *,
    model_factory: Callable[[], Any],
    instances: Iterable[Dict[str, Any]],
    prompt_template: Any,
    parser: Any,
    critique_prompts: Dict[str, Any],
    judge_prompt: Any,
    judge_parser: Any,
    revision_prompt: Any,
    ear_configs: List[EarRunConfig],
    experiment_name: str,
    base_dir: str = "results",
    dataset_name: str = "gsm8k",
    split: str = "test",
    progress: bool = True,
) -> Dict[str, Any]:
    normalized = list(instances)
    run_summaries: List[Dict[str, Any]] = []

    for cfg in ear_configs:
        model = model_factory()
        tracker = ExperimentTracker(f"{experiment_name}_{cfg.label}", base_dir=base_dir)
        summary = evaluate_ear_config(
            instances=normalized,
            cfg=cfg,
            model=model,
            prompt_template=prompt_template,
            parser=parser,
            critique_prompts=critique_prompts,
            judge_prompt=judge_prompt,
            judge_parser=judge_parser,
            revision_prompt=revision_prompt,
            tracker=tracker,
            progress=progress,
        )
        run_summaries.append(summary)

    aggregate_tracker = ExperimentTracker(f"{experiment_name}_aggregate", base_dir=base_dir)
    aggregate_summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "split": split,
        "method": "ear",
        "num_configs": len(ear_configs),
        "runs": run_summaries,
    }
    aggregate_tracker.save_summary(aggregate_summary)

    with open(aggregate_tracker.summary_filepath, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=4)

    return aggregate_summary
