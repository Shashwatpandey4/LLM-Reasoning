import pytest
from src.common.parsing.mcq import extract_mcq_answer


@pytest.mark.parametrize("text,expected", [
    ("The answer is (B)", "B"),
    ("The answer is C", "C"),
    ("...therefore the answer is (A).", "A"),
    ("blah blah D blah", "D"),
    ("no letter anywhere", None),
    ("THE ANSWER IS (d)", "D"),
    ("The answer is (A) because reasons", "A"),
])
def test_extract_mcq_answer(text, expected):
    assert extract_mcq_answer(text) == expected
