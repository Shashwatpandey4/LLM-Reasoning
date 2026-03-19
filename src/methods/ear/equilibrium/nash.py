"""Nash equilibrium selector.

Convergence is defined as a fixed point under revision: no candidate changes
its answer in a round. This mirrors the game-theoretic notion of Nash
equilibrium — a state where no player wants to deviate given the others' strategies.

At convergence the winner is the most stable candidate (did not change last round)
with the fewest successful critiques against it. If the max-rounds budget is
exhausted without convergence, the same score is used as a fallback.
"""

from typing import Any, Dict, List

from src.methods.ear.equilibrium.base import BaseEquilibriumSelector
from src.methods.ear.types import JudgeResult, RevisionResult
from src.registry import EQUILIBRIUM_SELECTORS


@EQUILIBRIUM_SELECTORS.register("nash")
class NashSelector(BaseEquilibriumSelector):
    def initialize(self, candidate_ids: List[int]) -> None:
        self._candidate_ids = list(candidate_ids)
        # Track successful critiques per candidate (used for tie-breaking)
        self._hits: Dict[int, int] = {cid: 0 for cid in candidate_ids}
        # Whether each candidate changed its answer in the most recent revision step
        self._changed_last_round: Dict[int, bool] = {cid: False for cid in candidate_ids}
        self._converged = False

    def update(self, judge_results: List[JudgeResult]) -> None:
        for jr in judge_results:
            if jr.is_successful:
                self._hits[jr.target_id] += 1

    def notify_revisions(self, revision_results: List[RevisionResult]) -> None:
        self._changed_last_round = {cid: False for cid in self._candidate_ids}
        for rv in revision_results:
            self._changed_last_round[rv.candidate_id] = rv.answer_changed
        self._converged = not any(self._changed_last_round.values())

    def get_scores(self) -> Dict[int, float]:
        # Higher score = fewer hits (more stable)
        max_hits = max(self._hits.values(), default=0)
        return {cid: float(max_hits - hits) for cid, hits in self._hits.items()}

    def select_winner_id(self, candidates: List[Dict[str, Any]]) -> int:
        # Prefer candidates that did NOT change last round (more stable)
        stable = [cid for cid, changed in self._changed_last_round.items() if not changed]
        pool = stable if stable else self._candidate_ids
        # Among stable candidates, fewest successful critiques wins; tie-break by lower id
        return min(pool, key=lambda cid: (self._hits.get(cid, 0), cid))

    def is_converged(self) -> bool:
        return self._converged
