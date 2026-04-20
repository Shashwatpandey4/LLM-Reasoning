"""Pareto frontier: accuracy vs total tokens per instance.

One point per (method, config) combination. Upper-left dominates.

Usage:
    uv run python benchmarks/plot_pareto.py
    uv run python benchmarks/plot_pareto.py --summaries_dir results/summaries --output_dir results/plots
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MODEL_SIZES = {
    "qwen3-0.6b": 0.6,
    "qwen3-1.7b": 1.7,
    "qwen3-4b": 4.0,
    "qwen3-8b": 8.0,
    "qwen3-14b": 14.0,
}

KNOWN_MODELS = list(MODEL_SIZES.keys())


def extract_model(experiment_name: str) -> str | None:
    for m in KNOWN_MODELS:
        if m in experiment_name:
            return m
    return None


def load_summaries(summaries_dir: str) -> list[dict]:
    records = []
    for fname in os.listdir(summaries_dir):
        if not fname.endswith("_metrics.json"):
            continue
        path = os.path.join(summaries_dir, fname)
        with open(path) as f:
            data = json.load(f)

        model = extract_model(data.get("experiment_name", ""))
        if model is None:
            continue

        method = data.get("method", "unknown")
        dataset = data.get("dataset", "unknown")

        for run in data.get("runs", []):
            acc = run.get("accuracy", 0.0)
            tokens = run.get("avg_total_tokens", 0.0)
            if tokens == 0:
                continue

            strategy = run.get("critique_strategy", "")
            selector = run.get("equilibrium_selector", "")
            k = run.get("k", "")
            label = method
            if method == "ear":
                label = f"EAR ({strategy[:3]}/{selector[:3]})"
            elif method == "self_consistency":
                label = f"SC (k={k})"
            elif method == "single_cot":
                label = f"CoT (B={run.get('max_new_tokens', '')})"

            records.append(
                {
                    "model": model,
                    "model_size": MODEL_SIZES[model],
                    "method": method,
                    "dataset": dataset,
                    "accuracy": acc * 100,
                    "avg_total_tokens": tokens,
                    "label": label,
                }
            )

    return records


def plot_pareto(summaries_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    records = load_summaries(summaries_dir)

    if not records:
        print("No matching summary files found.")
        return

    datasets = sorted(set(r["dataset"] for r in records))
    models = sorted(set(r["model"] for r in records), key=lambda m: MODEL_SIZES.get(m, 0))

    method_markers = {
        "ear": "s",
        "self_consistency": "^",
        "single_cot": "o",
    }
    method_labels = {
        "ear": "EAR",
        "self_consistency": "SC",
        "single_cot": "SingleCoT",
    }

    for dataset in datasets:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
        if len(models) == 1:
            axes = [axes]

        sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
        palette = sns.color_palette("Set2", 3)
        color_map = {"ear": palette[0], "self_consistency": palette[1], "single_cot": palette[2]}

        for ax, model in zip(axes, models):
            model_records = [r for r in records if r["dataset"] == dataset and r["model"] == model]
            if not model_records:
                ax.set_title(f"{model}\n(no data)")
                continue

            for method in ["single_cot", "self_consistency", "ear"]:
                pts = [r for r in model_records if r["method"] == method]
                if not pts:
                    continue
                xs = [r["avg_total_tokens"] for r in pts]
                ys = [r["accuracy"] for r in pts]
                ax.scatter(
                    xs,
                    ys,
                    marker=method_markers[method],
                    color=color_map[method],
                    label=method_labels[method],
                    alpha=0.8,
                    s=70,
                    edgecolors="white",
                    linewidths=0.5,
                )

            ax.set_title(f"{model}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Avg Tokens / Instance", fontsize=10)
            if ax == axes[0]:
                ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 100)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=3,
                fontsize=10,
                bbox_to_anchor=(0.5, -0.08),
            )

        fig.suptitle(
            f"Accuracy vs. Compute Pareto — {dataset.upper()}", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        out = os.path.join(output_dir, f"pareto_{dataset}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", default="results/summaries")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    plot_pareto(args.summaries_dir, args.output_dir)
