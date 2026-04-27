from typing import Any, Dict, List

from datasets import load_dataset

from src.registry import DATASETS

LETTERS = ["A", "B", "C", "D"]


class LogiQADataset:
    """Wrapper for the LogiQA dataset."""

    def __init__(self, split: str = "test"):
        self.split = split
        print(f"Loading LogiQA dataset ({split} split)...")
        self.dataset = load_dataset(
            "parquet",
            data_files={
                split: f"hf://datasets/lucasmccabe/logiqa@refs/convert/parquet/default/{split}/*.parquet"
            },
            split=split,
        )
        print(f"Loaded {len(self.dataset)} examples.")

    def get_data(self) -> List[Dict[str, Any]]:
        processed_data = []
        for index, item in enumerate(self.dataset):
            context = item["context"]
            query = item["query"]
            options = item["options"]
            correct_option = item["correct_option"]

            question = f"{context}\n\n{query}"
            answer = LETTERS[correct_option]
            choices = {
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
            }

            processed_data.append(
                {
                    "id": str(index),
                    "dataset": "logiqa",
                    "split": self.split,
                    "question": question,
                    "answer": answer,
                    "choices": choices,
                }
            )
        return processed_data


@DATASETS.register("logiqa")
def load_logiqa(split: str = "test", **kwargs) -> LogiQADataset:
    return LogiQADataset(split=split)
