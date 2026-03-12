from typing import List, Dict, Any
from datasets import load_dataset
from src.registry import DATASETS

class GSM8KDataset:
    """Wrapper for the GSM8K dataset."""
    def __init__(self, split: str = "test"):
        self.split = split
        print(f"Loading GSM8K dataset ({split} split)...")
        self.dataset = load_dataset("gsm8k", "main", split=split)
        print(f"Loaded {len(self.dataset)} examples.")

    def get_data(self) -> List[Dict[str, Any]]:
        """
        Returns a list of dictionaries with 'question' and 'answer'.
        The 'answer' in the raw dataset contains the reasoning and the final number.
        We extract the final number here for evaluation (it comes after ####).
        """
        processed_data = []
        for item in self.dataset:
            question = item['question']
            raw_answer = item['answer']
            
            # Extract ground truth number
            # GSM8K format: [Reasoning...] #### [Answer]
            if "####" in raw_answer:
                ground_truth = raw_answer.split("####")[-1].strip()
            else:
                ground_truth = None
                
            processed_data.append({
                "question": question,
                "answer": ground_truth,
                "raw_answer": raw_answer
            })
        return processed_data

@DATASETS.register("gsm8k")
def load_gsm8k(split: str = "test", **kwargs) -> GSM8KDataset:
    return GSM8KDataset(split=split)
