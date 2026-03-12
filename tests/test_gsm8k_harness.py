import json
from pathlib import Path

from src.experiments.gsm8k_single_cot_sweep import run_gsm8k_single_cot_sweep


class FakePrompt:
    def format(self, **kwargs):
        return f"Solve: {kwargs['question']}"


class FakeParser:
    def parse(self, text):
        marker = "The answer is "
        start = text.index(marker)
        return text[start + len(marker) :].strip(), start


class FakeModel:
    def __init__(self, answer_by_budget):
        self.answer_by_budget = answer_by_budget

    def generate(self, prompt, max_new_tokens=None, **kwargs):
        answer = self.answer_by_budget[max_new_tokens]
        return f"Reasoning for {prompt}. The answer is {answer}"

    def count_tokens(self, text):
        return len(text.split())


def test_gsm8k_budget_sweep_writes_results_and_plots(tmp_path):
    instances = [
        {"id": "ex-1", "question": "1+1", "answer": "2"},
        {"id": "ex-2", "question": "2+2", "answer": "4"},
    ]

    def model_factory():
        return FakeModel({8: "2", 32: "4"})

    summary = run_gsm8k_single_cot_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=FakePrompt(),
        parser=FakeParser(),
        budgets=[8, 32],
        experiment_name="gsm8k_test",
        base_dir=str(tmp_path),
        progress=False,
    )

    assert summary["budgets"] == [8, 32]
    assert len(summary["runs"]) == 2
    assert Path(summary["plot_manifest_path"]).exists()

    for run in summary["runs"]:
        assert Path(run["raw_path"]).exists()
        assert run["num_examples"] == 2

    manifest = json.loads(Path(summary["plot_manifest_path"]).read_text(encoding="utf-8"))
    assert set(manifest) == {"accuracy", "avg_reasoning_tokens", "parse_success_rate"}
    for plot_path in manifest.values():
        assert Path(plot_path).exists()
