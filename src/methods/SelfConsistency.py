from typing import Any, Dict, List

from src.common.eval.voting import majority_vote, vote_distribution
from src.methods.Base import BaseReasoningMethod
from src.registry import METHODS


@METHODS.register("self_consistency")
class SelfConsistency(BaseReasoningMethod):
    """
    Self-Consistency: sample k independent reasoning paths and select the
    answer by majority vote.

    Each candidate is a full SingleCoT-style generation. The final answer is
    the one that appears most frequently across all k candidates (with
    numeric-aware canonicalization for vote counting).
    """

    def __init__(self, model: Any, prompt_template: Any, parser: Any, k: int = 5):
        super().__init__(model, prompt_template, parser)
        self.k = k

    def run(self, instance: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        question = instance.get("question")
        if not question:
            raise ValueError("Instance must contain a 'question' key.")

        candidates: List[Dict[str, Any]] = []

        for i in range(self.k):
            choices = instance.get("choices")
            if choices is not None:
                prompt = self.prompt_template.format(question=question, choices=choices)
            else:
                prompt = self.prompt_template.format(question=question)

            generation = self.model.generate(prompt, **kwargs)
            extracted_answer, answer_start_idx = self.parser.parse(generation)
            reasoning_text = generation[:answer_start_idx]
            answer_text = generation[answer_start_idx:]

            candidates.append(
                {
                    "id": i,
                    "prompt": prompt,
                    "raw_generation": generation,
                    "reasoning_text": reasoning_text,
                    "answer_text": answer_text,
                    "extracted_answer": extracted_answer,
                    "parse_success": extracted_answer is not None,
                    "tokens": {
                        "total": self.model.count_tokens(generation),
                        "reasoning": self.model.count_tokens(reasoning_text),
                        "answer": self.model.count_tokens(answer_text),
                    },
                }
            )

        all_answers = [c["extracted_answer"] for c in candidates]
        selected_answer = majority_vote(all_answers)
        votes = vote_distribution(all_answers)

        total_tokens = sum(c["tokens"]["total"] for c in candidates)
        parse_success_count = sum(1 for c in candidates if c["parse_success"])

        return {
            "instance_id": instance.get("id"),
            "dataset": instance.get("dataset"),
            "split": instance.get("split"),
            "question": question,
            "ground_truth": instance.get("answer"),
            "candidates": candidates,
            "vote_distribution": votes,
            "extracted_answer": selected_answer,
            "parse_success": selected_answer is not None,
            "run_config": {
                "k": self.k,
                "max_new_tokens": kwargs.get("max_new_tokens"),
            },
            "metrics": {
                "total_tokens": total_tokens,
                "total_model_calls": self.k,
                "avg_tokens_per_candidate": total_tokens / self.k if self.k > 0 else 0,
                "parse_success_count": parse_success_count,
                "parse_success_rate": parse_success_count / self.k if self.k > 0 else 0,
            },
        }
