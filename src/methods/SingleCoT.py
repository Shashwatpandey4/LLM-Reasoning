from typing import Any, Dict

from src.methods.Base import BaseReasoningMethod
from src.registry import METHODS


@METHODS.register("single_cot")
class SingleCoT(BaseReasoningMethod):
    """
    Implements a single step Chain of Thought reasoning method.
    """

    def run(self, instance: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Runs the Single CoT method.
        Requires 'question' in the instance dictionary.
        """
        question = instance.get("question")
        if not question:
            raise ValueError("Instance must contain a 'question' key.")

        # 1. Format the prompt
        choices = instance.get("choices")
        if choices is not None:
            prompt = self.prompt_template.format(question=question, choices=choices)
        else:
            prompt = self.prompt_template.format(question=question)

        # 2. Run generation
        generation_output = self.model.generate(prompt, **kwargs)

        # 3. Parse the result to extract the answer and find the split point
        extracted_answer, answer_start_idx = self.parser.parse(generation_output)

        # 4. Split Generation into reasoning and answer parts
        reasoning_text = generation_output[:answer_start_idx]
        answer_text = generation_output[answer_start_idx:]

        # 5. Calculate Token Metrics
        total_tokens = self.model.count_tokens(generation_output)
        reasoning_tokens = self.model.count_tokens(reasoning_text)
        answer_tokens = self.model.count_tokens(answer_text)

        return {
            "instance_id": instance.get("id"),
            "dataset": instance.get("dataset"),
            "split": instance.get("split"),
            "question": question,
            "prompt": prompt,
            "raw_generation": generation_output,
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "extracted_answer": extracted_answer,
            "ground_truth": instance.get("answer"),
            "parse_success": extracted_answer is not None,
            "run_config": {
                "max_new_tokens": kwargs.get("max_new_tokens"),
            },
            "metrics": {
                "total_tokens": total_tokens,
                "reasoning_tokens": reasoning_tokens,
                "answer_tokens": answer_tokens,
            },
        }
