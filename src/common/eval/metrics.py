from typing import Any, Dict, List, Optional


def normalize_numeric_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().replace(",", "")


def answers_match(prediction: Optional[str], ground_truth: Optional[str]) -> bool:
    pred_clean = normalize_numeric_text(prediction)
    truth_clean = normalize_numeric_text(ground_truth)

    if not pred_clean or not truth_clean:
        return False

    try:
        p_float = float(pred_clean)
        t_float = float(truth_clean)
        return abs(p_float - t_float) < 1e-5
    except ValueError:
        return pred_clean == truth_clean


def exact_match_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Computes exact match accuracy.
    Assumes inputs are already somewhat clean strings representing numbers.
    """
    if not predictions or not ground_truths:
        return 0.0

    correct = 0
    for pred, truth in zip(predictions, ground_truths):
        if answers_match(pred, truth):
            correct += 1

    return correct / len(predictions)


def summarize_sc_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics for a single Self-Consistency k-sweep run."""
    if not records:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "parse_success_rate": 0.0,
            "avg_total_tokens": 0.0,
            "avg_tokens_per_candidate": 0.0,
            "avg_model_calls": 0.0,
            "avg_candidate_parse_success_rate": 0.0,
        }

    predictions = [r.get("extracted_answer") for r in records]
    ground_truths = [r.get("ground_truth") for r in records]
    num_examples = len(records)

    return {
        "num_examples": num_examples,
        "accuracy": exact_match_accuracy(predictions, ground_truths),
        "parse_success_rate": sum(1 for r in records if r.get("parse_success")) / num_examples,
        "avg_total_tokens": sum(r["metrics"]["total_tokens"] for r in records) / num_examples,
        "avg_tokens_per_candidate": sum(r["metrics"]["avg_tokens_per_candidate"] for r in records)
        / num_examples,
        "avg_model_calls": sum(r["metrics"]["total_model_calls"] for r in records) / num_examples,
        "avg_candidate_parse_success_rate": sum(r["metrics"]["parse_success_rate"] for r in records)
        / num_examples,
    }


def summarize_ear_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics for a single EAR configuration run."""
    if not records:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "parse_success_rate": 0.0,
            "avg_total_tokens": 0.0,
            "avg_model_calls": 0.0,
            "avg_successful_critiques": 0.0,
            "avg_total_critiques": 0.0,
            "avg_revisions": 0.0,
            "avg_rounds_until_convergence": 0.0,
        }

    predictions = [r.get("extracted_answer") for r in records]
    ground_truths = [r.get("ground_truth") for r in records]
    num_examples = len(records)

    return {
        "num_examples": num_examples,
        "accuracy": exact_match_accuracy(predictions, ground_truths),
        "parse_success_rate": sum(1 for r in records if r.get("parse_success")) / num_examples,
        "avg_total_tokens": sum(r["metrics"]["total_tokens"] for r in records) / num_examples,
        "avg_model_calls": sum(r["metrics"]["total_model_calls"] for r in records) / num_examples,
        "avg_successful_critiques": sum(r["metrics"]["successful_critiques"] for r in records)
        / num_examples,
        "avg_total_critiques": sum(r["metrics"]["total_critiques"] for r in records) / num_examples,
        "avg_revisions": sum(r["metrics"]["total_revisions"] for r in records) / num_examples,
        "avg_rounds_until_convergence": sum(
            r["metrics"]["rounds_until_convergence"] for r in records
        )
        / num_examples,
    }


def summarize_budget_run(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "num_examples": 0,
            "accuracy": 0.0,
            "parse_success_rate": 0.0,
            "avg_total_tokens": 0.0,
            "avg_reasoning_tokens": 0.0,
            "avg_answer_tokens": 0.0,
        }

    predictions = [record.get("extracted_answer") for record in records]
    ground_truths = [record.get("ground_truth") for record in records]
    num_examples = len(records)

    return {
        "num_examples": num_examples,
        "accuracy": exact_match_accuracy(predictions, ground_truths),
        "parse_success_rate": sum(1 for record in records if record.get("parse_success"))
        / num_examples,
        "avg_total_tokens": sum(record["metrics"]["total_tokens"] for record in records)
        / num_examples,
        "avg_reasoning_tokens": sum(record["metrics"]["reasoning_tokens"] for record in records)
        / num_examples,
        "avg_answer_tokens": sum(record["metrics"]["answer_tokens"] for record in records)
        / num_examples,
    }
