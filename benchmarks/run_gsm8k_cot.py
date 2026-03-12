import argparse
import os
import sys

from tqdm import tqdm

# Add project root to sys path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import everything so registries get populated
from src.common.eval.metrics import exact_match_accuracy
from src.common.utils.storage import ExperimentTracker
from src.registry import DATASETS, METHODS, MODELS, PARSERS, PROMPTS


def main():
    parser = argparse.ArgumentParser(description="Run GSM8K with Gemma-3-1b-it")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help=("Number of examples from the test set to evaluate. Use 0 for the full dataset."),
    )
    args = parser.parse_args()

    # 1. Load Model
    print("-" * 50)
    print("Initiating models and data...")
    get_model_fn = MODELS.get("gemma-3-1b-it")
    # Setting max_new_tokens relatively high for reasoning
    model = get_model_fn(max_new_tokens=1024, temperature=0.7)

    # 2. Load Dataset
    get_dataset_fn = DATASETS.get("gsm8k")
    dataset = get_dataset_fn(split="test")
    raw_data = dataset.get_data()

    if args.num_examples > 0:
        raw_data = raw_data[: args.num_examples]
        print(f"Subsampling to {args.num_examples} examples for testing run.")

    # 3. Load Prompts & Parsing
    get_prompt_fn = PROMPTS.get("gsm8k_cot")
    prompt_template = get_prompt_fn()

    get_parser_fn = PARSERS.get("gsm8k_parser")
    answer_parser = get_parser_fn()

    # 4. Initialize Reasoning Method
    method_class = METHODS.get("single_cot")
    reasoning_method = method_class(
        model=model, prompt_template=prompt_template, parser=answer_parser
    )

    # 5. Initialize Storage
    tracker = ExperimentTracker("gsm8k_gemma-3-1b-it")
    print(f"Logging results to: {tracker.run_id}")

    # 6. Execute Evaluation Loop
    print("-" * 50)
    print("Starting evaluation loop...")

    predictions = []
    ground_truths = []
    token_stats = {"total": [], "reasoning": [], "answer": []}

    for i, instance in enumerate(tqdm(raw_data, desc="Evaluating")):
        # Run Single CoT reasoning
        result = reasoning_method.run(instance)

        predictions.append(result["extracted_answer"])
        ground_truths.append(result["ground_truth"])

        # Track Tokens
        metrics = result.get("metrics", {})
        token_stats["total"].append(metrics.get("total_tokens", 0))
        token_stats["reasoning"].append(metrics.get("reasoning_tokens", 0))
        token_stats["answer"].append(metrics.get("answer_tokens", 0))

        # Save exact result immediately
        tracker.log_instance(result)

        # Periodically log to show progress
        if i < 3:
            print(f"\n[Example {i + 1}]")
            print(f"Q: {result['question'][:100]}...")
            print(f"Ground-Truth: {result['ground_truth']}")
            print(f"Prediction:   {result['extracted_answer']}")
            print("-" * 20)

    # 7. Calculate Metrics
    accuracy = exact_match_accuracy(predictions, ground_truths)

    avg_total = sum(token_stats["total"]) / len(token_stats["total"]) if token_stats["total"] else 0
    avg_reasoning = (
        sum(token_stats["reasoning"]) / len(token_stats["reasoning"])
        if token_stats["reasoning"]
        else 0
    )
    avg_answer = (
        sum(token_stats["answer"]) / len(token_stats["answer"]) if token_stats["answer"] else 0
    )

    summary = {
        "num_examples": len(predictions),
        "exact_match_accuracy": accuracy,
        "avg_total_tokens": avg_total,
        "avg_reasoning_tokens": avg_reasoning,
        "avg_answer_tokens": avg_answer,
    }

    tracker.save_summary(summary)

    print("-" * 50)
    print("Evaluation Complete!")
    print(f"Total Examples Evaluated: {len(predictions)}")
    print(f"Exact Match Accuracy: {accuracy * 100:.2f}%")
    print(f"Avg Reasoning Tokens: {avg_reasoning:.1f}")
    print(f"Avg Total Tokens: {avg_total:.1f}")


if __name__ == "__main__":
    # Ensure Huggingface tokens are handled if required by user's environment
    main()
