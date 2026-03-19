"""Panel cross-candidate critique strategy.

Each candidate acts as a critiquer and critiques every other candidate,
with full visibility into all candidates' solutions as context.

Produces k*(k-1) critique calls per round — the richest signal for Elo
because critiquer_id is meaningful (inter-candidate competition).
"""

from typing import Any, Dict, List

from src.methods.ear.critique.base import BaseCritiqueStrategy
from src.methods.ear.types import CritiqueResult
from src.registry import CRITIQUE_STRATEGIES


def _format_panel_context(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for c in candidates:
        lines.append(f"Candidate {c['id']}:")
        lines.append(c.get("reasoning_text") or c.get("raw_generation") or "(no reasoning)")
        lines.append(f"Answer: {c.get('extracted_answer') or '(unparsed)'}")
        lines.append("")
    return "\n".join(lines).strip()


@CRITIQUE_STRATEGIES.register("panel")
class PanelCritique(BaseCritiqueStrategy):
    def run(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        model: Any,
        **kwargs: Any,
    ) -> List[CritiqueResult]:
        panel_context = _format_panel_context(candidates)
        results = []

        for critiquer in candidates:
            for target in candidates:
                if critiquer["id"] == target["id"]:
                    continue
                prompt = self.critique_prompt.format(
                    question=question,
                    panel_context=panel_context,
                    target_id=target["id"],
                    target_reasoning=target.get("reasoning_text")
                    or target.get("raw_generation")
                    or "",
                    target_answer=target.get("extracted_answer") or "",
                )
                critique_text = model.generate(prompt, **kwargs)
                results.append(
                    CritiqueResult(
                        critiquer_id=critiquer["id"],
                        target_id=target["id"],
                        critique_text=critique_text,
                        tokens=model.count_tokens(critique_text),
                    )
                )
        return results
