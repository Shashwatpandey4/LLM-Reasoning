import re

from src.registry import PARSERS


class MCQParser:
    def parse(self, text: str) -> str | None:
        match = re.search(r"the answer is\s*\(([ABCD])\)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        match = re.search(r"the answer is\s+([ABCD])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        matches = list(re.finditer(r"\b([ABCD])\b", text))
        if matches:
            return matches[-1].group(1).upper()

        return None


def extract_mcq_answer(text: str) -> str | None:
    return MCQParser().parse(text)


@PARSERS.register("mcq")
def get_mcq_parser(**kwargs) -> MCQParser:
    return MCQParser()
