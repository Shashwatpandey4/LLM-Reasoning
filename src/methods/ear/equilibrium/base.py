from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.methods.ear.types import JudgeResult, RevisionResult


class BaseEquilibriumSelector(ABC):
    """Abstract base for equilibrium selection strategies.

    Lifecycle per EAR run:
        selector.initialize(candidate_ids)
        for each round:
            selector.update(judge_results)
            selector.notify_revisions(revision_results)  # optional hook
            if selector.is_converged(): break
        winner_id = selector.select_winner_id(candidates)
    """

    @abstractmethod
    def initialize(self, candidate_ids: List[int]) -> None:
        """Set up internal state for a new problem instance."""

    @abstractmethod
    def update(self, judge_results: List[JudgeResult]) -> None:
        """Update scores based on this round's judge verdicts."""

    def notify_revisions(self, revision_results: List[RevisionResult]) -> None:
        """Optional hook called after each revision step.

        Selectors that track answer stability (e.g. Nash) override this.
        Default: no-op.
        """

    @abstractmethod
    def get_scores(self) -> Dict[int, float]:
        """Return current score per candidate id (higher = better)."""

    @abstractmethod
    def select_winner_id(self, candidates: List[Dict[str, Any]]) -> int:
        """Return the id of the winning candidate."""

    @abstractmethod
    def is_converged(self) -> bool:
        """True when the selector has reached its stopping condition.

        For most selectors this is always False (run all rounds).
        Nash overrides this to signal fixed-point convergence.
        """
