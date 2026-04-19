"""Token efficiency table: accuracy vs compute across all methods.

Reads all summary JSONs and prints a markdown table per dataset with columns:
  Method | Model | Acc (%) | Tok/inst | Calls/inst | Acc/kTok

Usage:
    uv run python benchmarks/build_efficiency_table.py
    uv run python benchmarks/build_efficiency_table.py --summaries_dir results/summaries
"""

import argparse
import json
import os
import sys
from collections import defaultdict

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


def method_label(method: str, run: dict) -> str:
    if method == "single_cot":
        return f"SingleCoT (B={run.get('max_new_tokens', '?')})"
    if method == "self_consistency":
        return f"SC (k={run.get('k', '?')})"
    if method == "ear":
        s = run.get("critique_strategy", "?")[:3]
        sel = run.get("equilibrium_selector", "?")[:3]
        k = run.get("k", "?")
        r = run.get("rounds", "?")
        rev = "rev" if run.get("allow_revision") else "norev"
        return f"EAR ({s}/{sel}, k={k}, R={r}, {rev})"
    return method


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
            calls = run.get("avg_model_calls", 0.0)
            acc_ktok = (acc / tokens * 1000) if tokens > 0 else 0.0

            records.append({
                "dataset": dataset,
                "model": model,
                "model_size": MODEL_SIZES[model],
                "method": method,
                "method_label": method_label(method, run),
                "accuracy": acc * 100,
                "avg_total_tokens": tokens,
                "avg_model_calls": calls,
                "acc_per_ktok": acc_ktok * 100,
            })

    return records


def best_per_family(records: list[dict], dataset: str) -> list[dict]:
    """For each (model, method_family), keep the best-accuracy config."""
    by_key = defaultdict(list)
    for r in records:
        if r["dataset"] != dataset:
            continue
        key = (r["model"], r["method"])
        by_key[key].append(r)

    best = []
    for runs in by_key.values():
        best.append(max(runs, key=lambda x: x["accuracy"]))

    return sorted(best, key=lambda x: (x["model_size"], x["method"]))


def print_table(records: list[dict], dataset: str):
    rows = best_per_family(records, dataset)
    if not rows:
        print(f"No data for {dataset}.\n")
        return

    print(f"\n### {dataset.upper()}\n")
    header = f"| {'Method':<45} | {'Model':<14} | {'Acc (%)':<9} | {'Tok/inst':<10} | {'Calls':<7} | {'Acc/kTok':<9} |"
    sep    = f"|{'-'*47}|{'-'*16}|{'-'*11}|{'-'*12}|{'-'*9}|{'-'*11}|"
    print(header)
    print(sep)

    prev_model = None
    for r in rows:
        if prev_model and r["model"] != prev_model:
            print(sep)
        print(
            f"| {r['method_label']:<45} "
            f"| {r['model']:<14} "
            f"| {r['accuracy']:>7.1f}  "
            f"| {r['avg_total_tokens']:>8.0f}   "
            f"| {r['avg_model_calls']:>5.1f}   "
            f"| {r['acc_per_ktok']:>7.3f}    |"
        )
        prev_model = r["model"]
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dir", default="results/summaries")
    args = parser.parse_args()

    records = load_summaries(args.summaries_dir)
    if not records:
        print("No matching summary files found.")
        sys.exit(0)

    datasets = sorted(set(r["dataset"] for r in records))
    for dataset in datasets:
        print_table(records, dataset)
