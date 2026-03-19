"""Parser for the EAR judge model output.

The judge is asked to respond with "VALID" or "INVALID".
This parser extracts that verdict and assigns a confidence score.
"""

import re
from typing import Tuple

from src.registry import PARSERS


class JudgeParser:
    """Extracts VALID / INVALID verdict from judge model output."""

    def parse(self, text: str) -> Tuple[bool, float]:
        """Return (is_successful, confidence).

        Confidence reflects how clearly the model expressed the verdict:
        - 0.9  explicit VALID / INVALID keyword found
        - 0.6  softer yes / no signal found
        - 0.0  no signal — defaults to not successful (conservative)
        """
        upper = text.strip().upper()

        # Check INVALID first because it contains the substring "VALID".
        if re.search(r"\bINVALID\b", upper):
            return False, 0.9
        if re.search(r"\bVALID\b", upper):
            return True, 0.9

        # Softer signals
        if re.search(r"\bYES\b", upper):
            return True, 0.6
        if re.search(r"\bNO\b", upper):
            return False, 0.6

        # No signal — conservative default
        return False, 0.0


@PARSERS.register("ear_judge_parser")
def get_ear_judge_parser(**kwargs) -> JudgeParser:
    return JudgeParser()
