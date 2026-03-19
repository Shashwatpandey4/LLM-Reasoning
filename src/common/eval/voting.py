from collections import Counter
from typing import Dict, List, Optional


def _canonical(answer: Optional[str]) -> Optional[str]:
    """Normalize an answer string to a canonical form for vote counting.

    Converts numeric strings to a stable representation so that "42" and "42.0"
    are counted as the same vote.
    """
    if answer is None:
        return None
    cleaned = str(answer).strip().replace(",", "")
    try:
        f = float(cleaned)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return cleaned


def majority_vote(answers: List[Optional[str]]) -> Optional[str]:
    """Return the most common answer by canonical form.

    Returns None if no valid (non-None) answers are present.
    """
    canonical_to_original: Dict[str, str] = {}
    counts: Counter = Counter()

    for answer in answers:
        key = _canonical(answer)
        if key is not None:
            counts[key] += 1
            if key not in canonical_to_original:
                canonical_to_original[key] = answer  # type: ignore[assignment]

    if not counts:
        return None

    best_key = counts.most_common(1)[0][0]
    return canonical_to_original[best_key]


def vote_distribution(answers: List[Optional[str]]) -> Dict[str, int]:
    """Return a count of votes per canonical answer value."""
    counts: Counter = Counter()
    for answer in answers:
        key = _canonical(answer)
        if key is not None:
            counts[key] += 1
    return dict(counts)
