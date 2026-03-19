from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.methods.ear.types import CritiqueResult


class BaseCritiqueStrategy(ABC):
    """Abstract base for all critique strategies.

    Each strategy receives the question and the current candidate list, and
    returns a flat list of CritiqueResult objects — one per (critiquer, target)
    pair that the strategy chooses to generate.
    """

    def __init__(self, critique_prompt: Any):
        self.critique_prompt = critique_prompt

    @abstractmethod
    def run(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        model: Any,
        **kwargs: Any,
    ) -> List[CritiqueResult]:
        pass
