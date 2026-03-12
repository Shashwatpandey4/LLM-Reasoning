from src.common.prompts.gsm8k_cot import get_gsm8k_cot_prompt


def test_gsm8k_prompt_includes_question_and_answer_instruction():
    prompt = get_gsm8k_cot_prompt()
    rendered = prompt.format(question="What is 2 + 2?")
    assert "What is 2 + 2?" in rendered
    assert "The answer is [number]" in rendered
