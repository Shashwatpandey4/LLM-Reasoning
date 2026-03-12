from src.common.parsing.gsm8k import GSM8KParser


def test_parser_extracts_prompt_format_answer():
    parser = GSM8KParser()
    answer, start_idx = parser.parse("Work here. The answer is 1,234.")
    assert answer == "1234"
    assert start_idx == 11


def test_parser_falls_back_to_last_number():
    parser = GSM8KParser()
    answer, start_idx = parser.parse("Reasoning with no marker ends in 42")
    assert answer == "42"
    assert start_idx == 33
