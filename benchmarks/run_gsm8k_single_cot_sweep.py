import argparse
import os
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.experiments.gsm8k_single_cot_sweep import run_gsm8k_single_cot_sweep
from src.registry import DATASETS, MODELS, PARSERS, PROMPTS


def parse_budgets(raw_value: str) -> List[int]:
    return [int(token.strip()) for token in raw_value.split(",") if token.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run a GSM8K SingleCoT max_new_tokens sweep.")
    parser.add_argument("--budgets", default="8,16,32,64,128,256,512,1000")
    parser.add_argument(
        "--num_examples", type=int, default=0, help="0 uses the full official split."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--base_dir", default="results")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    budgets = parse_budgets(args.budgets)

    dataset = DATASETS.get("gsm8k")(split=args.split)
    instances = dataset.get_data()
    if args.num_examples > 0:
        instances = instances[: args.num_examples]

    prompt_template = PROMPTS.get("gsm8k_cot")()
    answer_parser = PARSERS.get("gsm8k_parser")()

    def model_factory():
        return MODELS.get("gemma-3-1b-it")(temperature=args.temperature)

    aggregate_summary = run_gsm8k_single_cot_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=prompt_template,
        parser=answer_parser,
        budgets=budgets,
        experiment_name="gsm8k_gemma-3-1b-it_single-cot",
        base_dir=args.base_dir,
        dataset_name="gsm8k",
        split=args.split,
        progress=True,
    )

    print("Sweep complete.")
    print(f"Aggregate summary: {aggregate_summary['plot_manifest_path']}")


if __name__ == "__main__":
    main()
