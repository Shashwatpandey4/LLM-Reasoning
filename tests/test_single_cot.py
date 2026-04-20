from src.methods.SingleCoT import SingleCoT


class FakePrompt:
    def format(self, **kwargs):
        return f"Question: {kwargs['question']}"


class FakeParser:
    def parse(self, text):
        marker = "The answer is "
        start = text.index(marker)
        return text[start + len(marker) :].strip(), start


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return "Reasoning steps. The answer is 7"

    def count_tokens(self, text):
        return len(text.split())


def test_single_cot_passes_budget_to_model_and_returns_metrics():
    method = SingleCoT(model=FakeModel(), prompt_template=FakePrompt(), parser=FakeParser())
    result = method.run({"question": "3 + 4?", "answer": "7"}, max_new_tokens=32)

    assert result["extracted_answer"] == "7"
    assert result["run_config"]["max_new_tokens"] == 32
    assert result["metrics"]["total_tokens"] >= result["metrics"]["answer_tokens"]


def test_single_cot_uses_choices_when_present():
    model = FakeModel()
    method = SingleCoT(model=model, prompt_template=FakePrompt(), parser=FakeParser())

    class MCQPrompt:
        def format(self, **kwargs):
            if "choices" in kwargs:
                return f"Question: {kwargs['question']}\nChoices: {kwargs['choices']}"
            return f"Question: {kwargs['question']}"

    method = SingleCoT(model=model, prompt_template=MCQPrompt(), parser=FakeParser())
    result = method.run(
        {
            "question": "Pick one",
            "choices": ["A", "B", "C"],
            "answer": "A",
        },
        max_new_tokens=32,
    )

    assert "Choices:" in model.calls[0]["prompt"]
    assert result["prompt"] == model.calls[0]["prompt"]
