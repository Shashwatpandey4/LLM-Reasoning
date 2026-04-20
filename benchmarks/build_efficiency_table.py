"""Build token-efficiency markdown tables from summary JSONs.

Reads aggregate summary files from results/summaries/ and prints one markdown
table per benchmark (e.g. GSM8K, LogiQA).

Columns:
    Method (config) | Model | Acc (%) | Tok/inst | Calls/inst | Acc/kTok

Usage:
    python benchmarks/build_efficiency_table.py
    python benchmarks/build_efficiency_table.py --summaries_dir results/summaries
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Skipping {path.name}: failed to read JSON ({e})", file=sys.stderr)
        return None


def _infer_model(experiment_name: str) -> str:
    """
    Extract model name from experiment_name.

    Works for names like:
      gsm8k_qwen3-0.6b_single-cot
      logiqa_qwen3-8b_ear
      gsm8k_llama-3.1-8b_self-consistency
      xyz_gemma-3-27b-it_aggregate
    """
    if not experiment_name:
        return "unknown"

    model_patterns = [
        r"(qwen3-\d+(?:\.\d+)?b)",
        r"(llama-\d+(?:\.\d+)?-\d+b)",
        r"(gemma-\d+-\d+b-it)",
        r"(phi-\d+(?:\.\d+)?-[a-z0-9.-]+)",
        r"(deepseek-r1-distill-\d+b)",
        r"(mixtral-\d+x\d+b)",
    ]

    for pattern in model_patterns:
        match = re.search(pattern, experiment_name)
        if match:
            return match.group(1)

    # fallback: try to recover some model-ish token
    for part in experiment_name.split("_"):
        if "-" in part and any(ch.isdigit() for ch in part):
            return part

    return "unknown"


def _format_method_config(method: str, run: Dict[str, Any]) -> str:
    if method == "single_cot":
        budget = run.get("budget", run.get("max_new_tokens", "?"))
        return f"SingleCoT (B={budget})"

    if method == "self_consistency":
        k = run.get("k", "?")
        budget = run.get("max_new_tokens", "?")
        return f"SelfConsistency (k={k}, B={budget})"

    if method == "ear":
        k = run.get("k", "?")
        rounds = run.get("rounds", "?")
        critique = run.get("critique_strategy", "?")
        selector = run.get("equilibrium_selector", "?")
        allow_revision = run.get("allow_revision", False)
        rev = "rev" if allow_revision else "norev"
        return f"EAR ({critique}, {selector}, k={k}, R={rounds}, {rev})"

    return method


def _flatten_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    dataset = summary.get("dataset", "unknown")
    benchmark = str(dataset).upper()
    experiment_name = summary.get("experiment_name", "")
    model = _infer_model(experiment_name)
    method = summary.get("method", "unknown")
    runs = summary.get("runs", [])

    rows: List[Dict[str, Any]] = []

    if not isinstance(runs, list):
        return rows

    for run in runs:
        if not isinstance(run, dict):
            continue

        accuracy = float(run.get("accuracy", 0.0))
        avg_tokens = float(run.get("avg_total_tokens", 0.0))
        avg_calls = float(run.get("avg_model_calls", 0.0))

        acc_percent = accuracy * 100.0
        acc_per_ktok = (acc_percent / avg_tokens * 1000.0) if avg_tokens > 0 else 0.0

        rows.append(
            {
                "benchmark": benchmark,
                "dataset": dataset,
                "model": model,
                "method": method,
                "method_config": _format_method_config(method, run),
                "accuracy_percent": acc_percent,
                "avg_tokens": avg_tokens,
                "avg_calls": avg_calls,
                "acc_per_ktok": acc_per_ktok,
            }
        )

    return rows


def load_all_rows(summaries_dir: str) -> List[Dict[str, Any]]:
    summaries_path = Path(summaries_dir)
    if not summaries_path.exists():
        print(f"Directory not found: {summaries_dir}", file=sys.stderr)
        return []

    rows: List[Dict[str, Any]] = []

    # Read all JSON files so partial/incremental outputs still work
    for path in sorted(summaries_path.glob("*.json")):
        summary = _safe_read_json(path)
        if summary is None:
            continue
        rows.extend(_flatten_summary(summary))

    return rows


def _sort_key(row: Dict[str, Any]):
    return (
        row["benchmark"],
        row["model"],
        row["method"],
        row["method_config"],
    )


def print_markdown_table(benchmark: str, rows: List[Dict[str, Any]]) -> None:
    print(f"\n### {benchmark}\n")

    if not rows:
        print("No results available.\n")
        return

    rows = sorted(rows, key=_sort_key)

    print("| Method (config) | Model | Acc (%) | Tok/inst | Calls/inst | Acc/kTok |")
    print("|---|---:|---:|---:|---:|---:|")

    for row in rows:
        print(
            f"| {row['method_config']} "
            f"| {row['model']} "
            f"| {row['accuracy_percent']:.2f} "
            f"| {row['avg_tokens']:.2f} "
            f"| {row['avg_calls']:.2f} "
            f"| {row['acc_per_ktok']:.3f} |"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Build markdown token-efficiency tables from experiment summaries."
    )
    parser.add_argument(
        "--summaries_dir",
        default="results/summaries",
        help="Directory containing summary JSON files.",
    )
    args = parser.parse_args()

    rows = load_all_rows(args.summaries_dir)

    if not rows:
        print("No summary rows found.")
        return

    benchmarks = sorted({row["benchmark"] for row in rows})

    for benchmark in benchmarks:
        benchmark_rows = [row for row in rows if row["benchmark"] == benchmark]
        print_markdown_table(benchmark, benchmark_rows)


if __name__ == "__main__":
    main()