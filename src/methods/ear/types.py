"""Shared dataclasses for EAR intermediate results."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CritiqueResult:
    """Output of one critique call targeting a single candidate."""

    critiquer_id: Optional[int]  # None for external (non-cross-candidate) strategies
    target_id: int
    critique_text: str
    tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critiquer_id": self.critiquer_id,
            "target_id": self.target_id,
            "critique_text": self.critique_text,
            "tokens": self.tokens,
        }


@dataclass
class JudgeResult:
    """Output of the judge evaluating one critique."""

    critiquer_id: Optional[int]
    target_id: int
    is_successful: bool
    confidence: float
    judge_text: str
    tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critiquer_id": self.critiquer_id,
            "target_id": self.target_id,
            "is_successful": self.is_successful,
            "confidence": self.confidence,
            "judge_text": self.judge_text,
            "tokens": self.tokens,
        }


@dataclass
class RevisionResult:
    """Output of a candidate revising its answer after a successful critique."""

    candidate_id: int
    old_answer: Optional[str]
    new_answer: Optional[str]
    revised_raw: str
    reasoning_text: str
    answer_text: str
    tokens: int
    answer_changed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "old_answer": self.old_answer,
            "new_answer": self.new_answer,
            "revised_raw": self.revised_raw,
            "reasoning_text": self.reasoning_text,
            "answer_text": self.answer_text,
            "tokens": self.tokens,
            "answer_changed": self.answer_changed,
        }
