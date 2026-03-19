"""Elo equilibrium selector.

When a critique is judged successful:
- If critiquer_id is not None (panel strategy): critiquer gains Elo, target loses Elo.
- If critiquer_id is None (external critic): only the target is penalized against a
  fixed reference Elo, since there is no opponent candidate to reward.

Winner = candidate with highest Elo at the end of all rounds.
"""

from typing import Any, Dict, List

from src.methods.ear.equilibrium.base import BaseEquilibriumSelector
from src.methods.ear.types import JudgeResult
from src.registry import EQUILIBRIUM_SELECTORS

_INITIAL_RATING = 1200.0
_K_FACTOR = 32.0
_EXTERNAL_REFERENCE_RATING = 1200.0  # notional rating of the external critic


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@EQUILIBRIUM_SELECTORS.register("elo")
class EloSelector(BaseEquilibriumSelector):
    def initialize(self, candidate_ids: List[int]) -> None:
        self._ratings: Dict[int, float] = {cid: _INITIAL_RATING for cid in candidate_ids}

    def update(self, judge_results: List[JudgeResult]) -> None:
        for jr in judge_results:
            if not jr.is_successful:
                continue

            target_rating = self._ratings[jr.target_id]

            if jr.critiquer_id is not None:
                # Cross-candidate: both parties get rating adjustment
                critiquer_rating = self._ratings[jr.critiquer_id]
                exp_critiquer = _expected(critiquer_rating, target_rating)
                exp_target = 1.0 - exp_critiquer
                self._ratings[jr.critiquer_id] += _K_FACTOR * (1.0 - exp_critiquer)
                self._ratings[jr.target_id] += _K_FACTOR * (0.0 - exp_target)
            else:
                # External critic: only penalize the target
                exp_target = _expected(target_rating, _EXTERNAL_REFERENCE_RATING)
                self._ratings[jr.target_id] += _K_FACTOR * (0.0 - exp_target)

    def get_scores(self) -> Dict[int, float]:
        return dict(self._ratings)

    def select_winner_id(self, candidates: List[Dict[str, Any]]) -> int:
        return max(self._ratings, key=lambda cid: (self._ratings[cid], -cid))

    def is_converged(self) -> bool:
        return False
