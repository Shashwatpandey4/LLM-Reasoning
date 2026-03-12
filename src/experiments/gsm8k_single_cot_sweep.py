from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.common.eval.metrics import answers_match, summarize_budget_run
from src.common.utils.storage import ExperimentTracker
from src.methods.SingleCoT import SingleCoT


@dataclass
class SweepRunResult:
    budget: int
    raw_path: str
    summary: Dict[str, Any]


def _normalize_instances(instances: Iterable[Dict[str, Any]], dataset_name: str, split: str) -> List[Dict[str, Any]]:
    normalized = []
    for index, instance in enumerate(instances):
        normalized.append(
            {
                "id": instance.get("id", f"{dataset_name}-{split}-{index}"),
                "dataset": instance.get("dataset", dataset_name),
                "split": instance.get("split", split),
                "question": instance["question"],
                "answer": instance.get("answer"),
                "raw_answer": instance.get("raw_answer"),
            }
        )
    return normalized


def evaluate_single_cot_budget(
    instances: List[Dict[str, Any]],
    model: Any,
    prompt_template: Any,
    parser: Any,
    max_new_tokens: int,
    tracker: ExperimentTracker,
    progress: bool = True,
) -> SweepRunResult:
    method = SingleCoT(model=model, prompt_template=prompt_template, parser=parser)
    records: List[Dict[str, Any]] = []

    iterator = tqdm(instances, desc=f"budget={max_new_tokens}", leave=False) if progress else instances
    for instance in iterator:
        result = method.run(instance, max_new_tokens=max_new_tokens)
        result["is_correct"] = answers_match(result.get("extracted_answer"), result.get("ground_truth"))
        tracker.log_instance(result)
        records.append(result)

    summary = summarize_budget_run(records)
    summary["budget"] = max_new_tokens
    summary["raw_path"] = tracker.raw_filepath
    tracker.save_summary(summary)
    return SweepRunResult(budget=max_new_tokens, raw_path=tracker.raw_filepath, summary=summary)


def generate_sweep_plots(summaries: List[Dict[str, Any]], run_id: str, plots_dir: str) -> Dict[str, str]:
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    df = pd.DataFrame(summaries).sort_values("budget")
    paths: Dict[str, str] = {}

    plot_specs = [
        ("accuracy", "Accuracy", "Accuracy", "accuracy"),
        ("avg_reasoning_tokens", "Average Reasoning Tokens", "Tokens", "avg_reasoning_tokens"),
        ("parse_success_rate", "Parse Success Rate", "Rate", "parse_success_rate"),
    ]

    for column, title, ylabel, suffix in plot_specs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="budget", y=column, marker="o", linewidth=2.5)
        plt.title(f"GSM8K SingleCoT Sweep: {title}")
        plt.xlabel("max_new_tokens budget")
        plt.ylabel(ylabel)
        if column.endswith("rate") or column == "accuracy":
            plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        output_path = os.path.join(plots_dir, f"{run_id}_{suffix}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        paths[suffix] = output_path

    return paths


def run_gsm8k_single_cot_sweep(
    *,
    model_factory: Callable[[], Any],
    instances: Iterable[Dict[str, Any]],
    prompt_template: Any,
    parser: Any,
    budgets: List[int],
    experiment_name: str,
    base_dir: str = "results",
    dataset_name: str = "gsm8k",
    split: str = "test",
    progress: bool = True,
) -> Dict[str, Any]:
    normalized_instances = _normalize_instances(instances, dataset_name=dataset_name, split=split)
    run_summaries: List[Dict[str, Any]] = []
    raw_paths: Dict[int, str] = {}

    for budget in budgets:
        model = model_factory()
        tracker = ExperimentTracker(f"{experiment_name}_budget-{budget}", base_dir=base_dir)
        run_result = evaluate_single_cot_budget(
            instances=normalized_instances,
            model=model,
            prompt_template=prompt_template,
            parser=parser,
            max_new_tokens=budget,
            tracker=tracker,
            progress=progress,
        )
        run_summaries.append(run_result.summary)
        raw_paths[budget] = run_result.raw_path

    aggregate_tracker = ExperimentTracker(f"{experiment_name}_aggregate", base_dir=base_dir)
    aggregate_summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "split": split,
        "method": "single_cot",
        "budgets": budgets,
        "runs": run_summaries,
        "raw_paths": raw_paths,
    }
    aggregate_tracker.save_summary(aggregate_summary)
    plot_paths = generate_sweep_plots(run_summaries, aggregate_tracker.run_id, aggregate_tracker.dirs["plots"])
    plot_manifest_path = aggregate_tracker.save_plot_manifest(plot_paths)
    aggregate_summary["plot_paths"] = plot_paths
    aggregate_summary["plot_manifest_path"] = plot_manifest_path

    with open(aggregate_tracker.summary_filepath, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=4)

    return aggregate_summary
