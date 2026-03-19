"""Survival equilibrium selector.

The winner is the candidate with the fewest successful critiques against it —
i.e. the one that was hardest to refute.
Ties broken by candidate id (lower wins).
"""

from typing import Any, Dict, List

from src.methods.ear.equilibrium.base import BaseEquilibriumSelector
from src.methods.ear.types import JudgeResult
from src.registry import EQUILIBRIUM_SELECTORS


@EQUILIBRIUM_SELECTORS.register("survival")
class SurvivalSelector(BaseEquilibriumSelector):
    def initialize(self, candidate_ids: List[int]) -> None:
        # Count of successful critiques received by each candidate
        self._hits: Dict[int, int] = {cid: 0 for cid in candidate_ids}

    def update(self, judge_results: List[JudgeResult]) -> None:
        for jr in judge_results:
            if jr.is_successful:
                self._hits[jr.target_id] += 1

    def get_scores(self) -> Dict[int, float]:
        # Invert: fewer hits = higher score
        max_hits = max(self._hits.values(), default=0)
        return {cid: float(max_hits - hits) for cid, hits in self._hits.items()}

    def select_winner_id(self, candidates: List[Dict[str, Any]]) -> int:
        return min(self._hits, key=lambda cid: (self._hits[cid], cid))

    def is_converged(self) -> bool:
        return False
