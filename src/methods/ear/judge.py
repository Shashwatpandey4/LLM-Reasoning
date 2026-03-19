"""Judge: evaluates whether a critique successfully refutes a candidate."""

from typing import Any, Dict

from src.methods.ear.types import CritiqueResult, JudgeResult


class Judge:
    """Wraps a model + prompt + parser to produce a JudgeResult for one critique."""

    def __init__(self, model: Any, judge_prompt: Any, judge_parser: Any):
        self.model = model
        self.judge_prompt = judge_prompt
        self.judge_parser = judge_parser

    def evaluate(
        self,
        question: str,
        target: Dict[str, Any],
        critique: CritiqueResult,
        **kwargs: Any,
    ) -> JudgeResult:
        prompt = self.judge_prompt.format(
            question=question,
            reasoning=target.get("reasoning_text") or target.get("raw_generation") or "",
            answer=target.get("extracted_answer") or "",
            critique=critique.critique_text,
        )
        judge_output = self.model.generate(prompt, **kwargs)
        is_successful, confidence = self.judge_parser.parse(judge_output)

        return JudgeResult(
            critiquer_id=critique.critiquer_id,
            target_id=critique.target_id,
            is_successful=is_successful,
            confidence=confidence,
            judge_text=judge_output,
            tokens=self.model.count_tokens(judge_output),
        )
