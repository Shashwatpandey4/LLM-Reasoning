from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.common.eval.metrics import answers_match, summarize_sc_run
from src.common.utils.storage import ExperimentTracker
from src.methods.SelfConsistency import SelfConsistency


@dataclass
class SCSweepRunResult:
    k: int
    raw_path: str
    summary: Dict[str, Any]


def evaluate_self_consistency_k(
    instances: List[Dict[str, Any]],
    model: Any,
    prompt_template: Any,
    parser: Any,
    k: int,
    max_new_tokens: int,
    tracker: ExperimentTracker,
    progress: bool = True,
) -> SCSweepRunResult:
    method = SelfConsistency(model=model, prompt_template=prompt_template, parser=parser, k=k)
    records: List[Dict[str, Any]] = []

    iterator = tqdm(instances, desc=f"k={k}", leave=False) if progress else instances
    for instance in iterator:
        result = method.run(instance, max_new_tokens=max_new_tokens)
        result["is_correct"] = answers_match(
            result.get("extracted_answer"), result.get("ground_truth")
        )
        tracker.log_instance(result)
        records.append(result)

    summary = summarize_sc_run(records)
    summary["k"] = k
    summary["max_new_tokens"] = max_new_tokens
    summary["raw_path"] = tracker.raw_filepath
    tracker.save_summary(summary)
    return SCSweepRunResult(k=k, raw_path=tracker.raw_filepath, summary=summary)


def generate_sc_sweep_plots(
    summaries: List[Dict[str, Any]], run_id: str, plots_dir: str
) -> Dict[str, str]:
    os.makedirs(plots_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    df = pd.DataFrame(summaries).sort_values("k")
    paths: Dict[str, str] = {}

    plot_specs = [
        ("accuracy", "Accuracy vs Number of Samples (k)", "Accuracy", "accuracy"),
        ("avg_total_tokens", "Total Tokens vs k", "Tokens", "avg_total_tokens"),
        ("parse_success_rate", "Parse Success Rate vs k", "Rate", "parse_success_rate"),
        (
            "avg_candidate_parse_success_rate",
            "Avg Candidate Parse Rate vs k",
            "Rate",
            "avg_candidate_parse_success_rate",
        ),
    ]

    for column, title, ylabel, suffix in plot_specs:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="k", y=column, marker="o", linewidth=2.5)
        plt.title(f"GSM8K Self-Consistency Sweep: {title}")
        plt.xlabel("k (number of sampled reasoning paths)")
        plt.ylabel(ylabel)
        if "rate" in column or column == "accuracy":
            plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        output_path = os.path.join(plots_dir, f"{run_id}_{suffix}.png")
        plt.savefig(output_path, dpi=300)
        plt.close()
        paths[suffix] = output_path

    return paths


def run_gsm8k_self_consistency_sweep(
    *,
    model_factory: Callable[[], Any],
    instances: Iterable[Dict[str, Any]],
    prompt_template: Any,
    parser: Any,
    k_values: List[int],
    max_new_tokens: int,
    experiment_name: str,
    base_dir: str = "results",
    dataset_name: str = "gsm8k",
    split: str = "test",
    progress: bool = True,
) -> Dict[str, Any]:
    normalized = list(instances)
    run_summaries: List[Dict[str, Any]] = []
    raw_paths: Dict[int, str] = {}

    for k in k_values:
        model = model_factory()
        tracker = ExperimentTracker(f"{experiment_name}_k-{k}", base_dir=base_dir)
        run_result = evaluate_self_consistency_k(
            instances=normalized,
            model=model,
            prompt_template=prompt_template,
            parser=parser,
            k=k,
            max_new_tokens=max_new_tokens,
            tracker=tracker,
            progress=progress,
        )
        run_summaries.append(run_result.summary)
        raw_paths[k] = run_result.raw_path

    aggregate_tracker = ExperimentTracker(f"{experiment_name}_aggregate", base_dir=base_dir)
    aggregate_summary = {
        "experiment_name": experiment_name,
        "dataset": dataset_name,
        "split": split,
        "method": "self_consistency",
        "k_values": k_values,
        "max_new_tokens": max_new_tokens,
        "runs": run_summaries,
        "raw_paths": {str(k): p for k, p in raw_paths.items()},
    }
    aggregate_tracker.save_summary(aggregate_summary)
    plot_paths = generate_sc_sweep_plots(
        run_summaries, aggregate_tracker.run_id, aggregate_tracker.dirs["plots"]
    )
    plot_manifest_path = aggregate_tracker.save_plot_manifest(plot_paths)
    aggregate_summary["plot_paths"] = plot_paths
    aggregate_summary["plot_manifest_path"] = plot_manifest_path

    with open(aggregate_tracker.summary_filepath, "w", encoding="utf-8") as f:
        json.dump(aggregate_summary, f, indent=4)

    return aggregate_summary
