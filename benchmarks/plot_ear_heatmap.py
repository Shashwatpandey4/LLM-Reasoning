"""EAR ablation heatmap: critique strategy x equilibrium selector.

For each (dataset, model), plots a 3x3 heatmap of accuracy at k=5, R=3,
revision=true. Shows which strategy/selector combination works best.

Usage:
    uv run python benchmarks/plot_ear_heatmap.py
    uv run python benchmarks/plot_ear_heatmap.py --k 5 --rounds 3
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
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
STRATEGIES = ["answer_level", "reasoning_level", "panel"]
SELECTORS = ["elo", "nash", "survival"]

STRATEGY_LABELS = {
    "answer_level": "Answer",
    "reasoning_level": "Reasoning",
    "panel": "Panel",
}
SELECTOR_LABELS = {
    "elo": "Elo",
    "nash": "Nash",
    "survival": "Survival",
}


def extract_model(experiment_name: str) -> str | None:
    for m in KNOWN_MODELS:
        if m in experiment_name:
            return m
    return None


def load_ear_runs(summaries_dir: str, k: int, rounds: int) -> list[dict]:
    records = []
    for fname in os.listdir(summaries_dir):
        if not fname.endswith("_metrics.json"):
            continue
        path = os.path.join(summaries_dir, fname)
        with open(path) as f:
            data = json.load(f)

        if data.get("method") != "ear":
            continue

        model = extract_model(data.get("experiment_name", ""))
        if model is None:
            continue

        dataset = data.get("dataset", "unknown")

        for run in data.get("runs", []):
            if run.get("k") != k or run.get("rounds") != rounds:
                continue
            if not run.get("allow_revision", True):
                continue
            records.append({
                "model": model,
                "dataset": dataset,
                "strategy": run.get("critique_strategy"),
                "selector": run.get("equilibrium_selector"),
                "accuracy": run.get("accuracy", 0.0) * 100,
            })

    return records


def plot_heatmap(summaries_dir: str, output_dir: str, k: int, rounds: int):
    os.makedirs(output_dir, exist_ok=True)
    records = load_ear_runs(summaries_dir, k, rounds)

    if not records:
        print(f"No EAR runs found with k={k}, rounds={rounds}.")
        return

    datasets = sorted(set(r["dataset"] for r in records))
    models = sorted(set(r["model"] for r in records), key=lambda m: MODEL_SIZES.get(m, 0))

    for dataset in datasets:
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
        if n_models == 1:
            axes = [axes]

        for ax, model in zip(axes, models):
            grid = np.full((len(STRATEGIES), len(SELECTORS)), np.nan)
            for r in records:
                if r["dataset"] != dataset or r["model"] != model:
                    continue
                if r["strategy"] not in STRATEGIES or r["selector"] not in SELECTORS:
                    continue
                i = STRATEGIES.index(r["strategy"])
                j = SELECTORS.index(r["selector"])
                # Keep best if multiple runs match
                if np.isnan(grid[i, j]) or r["accuracy"] > grid[i, j]:
                    grid[i, j] = r["accuracy"]

            mask = np.isnan(grid)
            sns.heatmap(
                grid,
                ax=ax,
                annot=True,
                fmt=".1f",
                mask=mask,
                cmap="YlGn",
                vmin=0,
                vmax=100,
                xticklabels=[SELECTOR_LABELS[s] for s in SELECTORS],
                yticklabels=[STRATEGY_LABELS[s] for s in STRATEGIES],
                linewidths=0.5,
                cbar=False,
                annot_kws={"size": 12, "weight": "bold"},
            )
            ax.set_title(f"{model}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Selector", fontsize=10)
            ax.set_ylabel("Critique Strategy" if ax == axes[0] else "", fontsize=10)

        fig.suptitle(
            f"EAR Ablation: Strategy × Selector — {dataset.upper()} (k={k}, R={rounds})",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        out = os.path.join(output_dir, f"ear_heatmap_{dataset}_k{k}_r{rounds}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", default="results/summaries")
    parser.add_argument("--output_dir", default="results/plots")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    plot_heatmap(args.summaries_dir, args.output_dir, args.k, args.rounds)
