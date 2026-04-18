"""Scale curve: accuracy vs model size for EAR vs SC vs SingleCoT.

Reads all summary JSONs from results/summaries/ and plots accuracy against
model size (in billions of parameters) for each method family.

Usage:
    uv run python benchmarks/plot_scale_curve.py
    uv run python benchmarks/plot_scale_curve.py --summaries_dir results/summaries --output_dir results/plots
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
            record = {
                "model": model,
                "model_size": MODEL_SIZES[model],
                "method": method,
                "dataset": dataset,
                "accuracy": run.get("accuracy", 0.0),
                "avg_total_tokens": run.get("avg_total_tokens", 0.0),
                "k": run.get("k"),
                "critique_strategy": run.get("critique_strategy"),
                "equilibrium_selector": run.get("equilibrium_selector"),
                "max_new_tokens": run.get("max_new_tokens"),
            }
            records.append(record)

    return records


def best_config(records: list[dict], method: str, dataset: str) -> list[dict]:
    """For each model, pick the config with the highest accuracy for the given method."""
    from collections import defaultdict

    by_model = defaultdict(list)
    for r in records:
        if r["method"] == method and r["dataset"] == dataset:
            by_model[r["model"]].append(r)

    best = []
    for model, runs in by_model.items():
        best.append(max(runs, key=lambda x: x["accuracy"]))
    return sorted(best, key=lambda x: x["model_size"])


def plot_scale_curve(summaries_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    records = load_summaries(summaries_dir)

    if not records:
        print("No matching summary files found.")
        return

    datasets = sorted(set(r["dataset"] for r in records))
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.1)

    for dataset in datasets:
        fig, ax = plt.subplots(figsize=(9, 6))

        method_styles = {
            "ear": dict(marker="s", linestyle="-", label="EAR (best config)"),
            "self_consistency": dict(marker="^", linestyle="--", label="SC (best k)"),
            "single_cot": dict(marker="o", linestyle=":", label="SingleCoT (best budget)"),
        }
        palette = sns.color_palette("Set2", len(method_styles))
        colors = dict(zip(method_styles.keys(), palette))

        plotted = False
        for method, style in method_styles.items():
            best = best_config(records, method, dataset)
            if not best:
                continue
            sizes = [r["model_size"] for r in best]
            accs = [r["accuracy"] * 100 for r in best]
            ax.plot(sizes, accs, color=colors[method], **style, linewidth=2, markersize=9)
            plotted = True

        if not plotted:
            print(f"No data for dataset '{dataset}', skipping.")
            plt.close()
            continue

        ax.set_xlabel("Model Size (B parameters)", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(f"Accuracy vs. Model Scale — {dataset.upper()}", fontsize=14, fontweight="bold")
        ax.set_xticks([MODEL_SIZES[m] for m in KNOWN_MODELS])
        ax.set_xticklabels([f"{MODEL_SIZES[m]}B" for m in KNOWN_MODELS])
        ax.legend(fontsize=11)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        out = os.path.join(output_dir, f"scale_curve_{dataset}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", default="results/summaries")
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    plot_scale_curve(args.summaries_dir, args.output_dir)
