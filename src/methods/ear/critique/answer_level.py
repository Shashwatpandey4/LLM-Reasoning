"""Answer-level critique strategy.

An external critic evaluates each candidate's final answer independently.
The critic does not see other candidates' solutions or the reasoning chain —
only the question and the extracted answer.

Produces k critique calls per round (one per candidate).
critiquer_id is None because there is no candidate doing the critiquing.
"""

from typing import Any, Dict, List

from src.methods.ear.critique.base import BaseCritiqueStrategy
from src.methods.ear.types import CritiqueResult
from src.registry import CRITIQUE_STRATEGIES


@CRITIQUE_STRATEGIES.register("answer_level")
class AnswerLevelCritique(BaseCritiqueStrategy):
    def run(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        model: Any,
        **kwargs: Any,
    ) -> List[CritiqueResult]:
        results = []
        for target in candidates:
            prompt = self.critique_prompt.format(
                question=question,
                answer=target.get("extracted_answer") or "",
            )
            critique_text = model.generate(prompt, **kwargs)
            results.append(
                CritiqueResult(
                    critiquer_id=None,
                    target_id=target["id"],
                    critique_text=critique_text,
                    tokens=model.count_tokens(critique_text),
                )
            )
        return results
