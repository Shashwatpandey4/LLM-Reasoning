import re

from src.registry import PARSERS


class MCQParser:
    def parse(self, text: str) -> tuple[str | None, int]:
        match = re.search(r"the answer is\s*\(([ABCD])\)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.start()

        match = re.search(r"the answer is\s+([ABCD])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), match.start()

        matches = list(re.finditer(r"\b([ABCD])\b", text))
        if matches:
            last = matches[-1]
            return last.group(1).upper(), last.start()

        return None, len(text)


def extract_mcq_answer(text: str) -> str | None:
    answer, _ = MCQParser().parse(text)
    return answer


@PARSERS.register("mcq")
def get_mcq_parser(**kwargs) -> MCQParser:
    return MCQParser()
