"""Compare two experiment runs side-by-side.

Reads two raw JSONL files (one per method) and prints a comparison table
plus a bar chart saved to results/plots/.

Usage:
    uv run python benchmarks/compare_methods.py \\
        --ear  results/raw/<ear_run>.jsonl \\
        --sc   results/raw/<sc_run>.jsonl
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.common.eval.metrics import answers_match


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_stats(records: list, label: str) -> dict:
    n = len(records)
    correct = sum(
        1 for r in records if answers_match(r.get("extracted_answer"), r.get("ground_truth"))
    )
    parsed = sum(1 for r in records if r.get("parse_success"))
    total_tokens = sum(r["metrics"]["total_tokens"] for r in records)
    total_calls = sum(r["metrics"]["total_model_calls"] for r in records)

    return {
        "label": label,
        "n": n,
        "accuracy": correct / n if n else 0.0,
        "parse_success_rate": parsed / n if n else 0.0,
        "avg_total_tokens": total_tokens / n if n else 0.0,
        "avg_model_calls": total_calls / n if n else 0.0,
        "correct": correct,
    }


def print_table(stats_list: list):
    col_w = 28
    metrics = [
        ("accuracy", "Accuracy", ".1%"),
        ("parse_success_rate", "Parse Success Rate", ".1%"),
        ("avg_total_tokens", "Avg Total Tokens", ".1f"),
        ("avg_model_calls", "Avg Model Calls", ".1f"),
    ]

    header = f"{'Metric':<{col_w}}" + "".join(f"{s['label']:>{col_w}}" for s in stats_list)
    print("\n" + "=" * (col_w * (len(stats_list) + 1)))
    print(header)
    print("=" * (col_w * (len(stats_list) + 1)))
    for key, display, fmt in metrics:
        row = f"{display:<{col_w}}"
        for s in stats_list:
            val = s[key]
            row += f"{format(val, fmt):>{col_w}}"
        print(row)
    print("-" * (col_w * (len(stats_list) + 1)))
    row = f"{'Examples (n)':<{col_w}}"
    for s in stats_list:
        row += f"{s['n']:>{col_w}}"
    print(row)
    print("=" * (col_w * (len(stats_list) + 1)))


def save_comparison_plot(stats_list: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    metrics = [
        ("accuracy", "Accuracy", "%.0f%%"),
        ("avg_total_tokens", "Avg Total Tokens", "%.0f"),
        ("avg_model_calls", "Avg Model Calls", "%.1f"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 6))
    labels = [s["label"] for s in stats_list]
    colors = sns.color_palette("Set2", len(stats_list))

    for ax, (key, title, val_fmt) in zip(axes, metrics):
        values = [s[key] for s in stats_list]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("")
        if key == "accuracy":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                val_fmt % val,
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    fig.suptitle("Method Comparison: EAR vs Self-Consistency (GSM8K)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "method_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compare EAR and Self-Consistency results.")
    parser.add_argument("--ear", required=True, help="Path to EAR raw JSONL file.")
    parser.add_argument("--sc", required=True, help="Path to SC raw JSONL file.")
    parser.add_argument("--output_dir", default="results/plots", help="Directory to save plots.")
    args = parser.parse_args()

    ear_records = load_jsonl(args.ear)
    sc_records = load_jsonl(args.sc)

    ear_stats = compute_stats(ear_records, "EAR (reasoning+Elo)")
    sc_stats = compute_stats(sc_records, "Self-Consistency")

    print_table([ear_stats, sc_stats])
    save_comparison_plot([ear_stats, sc_stats], args.output_dir)


if __name__ == "__main__":
    main()
