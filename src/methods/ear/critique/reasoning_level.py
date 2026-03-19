"""Reasoning-level critique strategy.

An external critic evaluates each candidate's full reasoning chain.
Richer than answer_level — the critic can identify specific logical errors
rather than just checking the final number.

Produces k critique calls per round (one per candidate).
critiquer_id is None (external critic, no candidate gaining Elo).
"""

from typing import Any, Dict, List

from src.methods.ear.critique.base import BaseCritiqueStrategy
from src.methods.ear.types import CritiqueResult
from src.registry import CRITIQUE_STRATEGIES


@CRITIQUE_STRATEGIES.register("reasoning_level")
class ReasoningLevelCritique(BaseCritiqueStrategy):
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
                reasoning=target.get("reasoning_text") or target.get("raw_generation") or "",
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
