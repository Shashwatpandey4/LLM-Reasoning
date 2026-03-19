import json
from pathlib import Path

from src.common.eval.voting import majority_vote, vote_distribution
from src.experiments.gsm8k_self_consistency_sweep import run_gsm8k_self_consistency_sweep


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class FakePrompt:
    def format(self, **kwargs):
        return f"Solve: {kwargs['question']}"


class FakeParser:
    def parse(self, text):
        marker = "The answer is "
        start = text.index(marker)
        return text[start + len(marker) :].strip(), start


class FakeModel:
    """Returns answers from a cycling list to simulate diversity across k samples."""

    def __init__(self, answers: list):
        self._answers = answers
        self._call_count = 0

    def generate(self, prompt, **kwargs):
        answer = self._answers[self._call_count % len(self._answers)]
        self._call_count += 1
        return f"Reasoning. The answer is {answer}"

    def count_tokens(self, text):
        return len(text.split())


# ---------------------------------------------------------------------------
# Voting utilities
# ---------------------------------------------------------------------------


def test_majority_vote_clear_winner():
    assert majority_vote(["42", "42", "37"]) == "42"


def test_majority_vote_numeric_normalization():
    # "42" and "42.0" should be treated as the same vote
    assert majority_vote(["42", "42.0", "37"]) == "42"


def test_majority_vote_all_none():
    assert majority_vote([None, None]) is None


def test_majority_vote_single():
    assert majority_vote(["7"]) == "7"


def test_vote_distribution_counts():
    dist = vote_distribution(["42", "42", "37", None])
    assert dist["42"] == 2
    assert dist["37"] == 1
    assert None not in dist


# ---------------------------------------------------------------------------
# SelfConsistency method structure
# ---------------------------------------------------------------------------


def test_self_consistency_result_structure():
    from src.methods.SelfConsistency import SelfConsistency

    model = FakeModel(["2", "2", "4"])
    method = SelfConsistency(model=model, prompt_template=FakePrompt(), parser=FakeParser(), k=3)
    instance = {"id": "ex-1", "question": "1+1", "answer": "2", "dataset": "gsm8k", "split": "test"}
    result = method.run(instance, max_new_tokens=64)

    assert result["extracted_answer"] == "2"
    assert result["parse_success"] is True
    assert len(result["candidates"]) == 3
    assert result["run_config"]["k"] == 3
    assert result["run_config"]["max_new_tokens"] == 64
    assert result["metrics"]["total_model_calls"] == 3
    assert result["metrics"]["total_tokens"] > 0
    assert "vote_distribution" in result

    for candidate in result["candidates"]:
        assert "raw_generation" in candidate
        assert "reasoning_text" in candidate
        assert "answer_text" in candidate
        assert "extracted_answer" in candidate
        assert "parse_success" in candidate
        assert "tokens" in candidate
        assert set(candidate["tokens"]) == {"total", "reasoning", "answer"}


def test_self_consistency_majority_wins():
    from src.methods.SelfConsistency import SelfConsistency

    # 2 votes for "4", 1 vote for "2" → "4" should win
    model = FakeModel(["4", "4", "2"])
    method = SelfConsistency(model=model, prompt_template=FakePrompt(), parser=FakeParser(), k=3)
    instance = {"id": "ex-1", "question": "2+2", "answer": "4"}
    result = method.run(instance, max_new_tokens=64)
    assert result["extracted_answer"] == "4"


# ---------------------------------------------------------------------------
# Full sweep harness
# ---------------------------------------------------------------------------


def test_gsm8k_self_consistency_sweep_writes_results_and_plots(tmp_path):
    instances = [
        {"id": "ex-1", "question": "1+1", "answer": "2"},
        {"id": "ex-2", "question": "2+2", "answer": "4"},
    ]

    # k=3: answers cycle [2, 2, 4] → majority "2" for ex-1 and "4" for ex-2
    def model_factory():
        return FakeModel(["2", "2", "4"])

    summary = run_gsm8k_self_consistency_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=FakePrompt(),
        parser=FakeParser(),
        k_values=[3, 5],
        max_new_tokens=64,
        experiment_name="gsm8k_sc_test",
        base_dir=str(tmp_path),
        progress=False,
    )

    assert summary["k_values"] == [3, 5]
    assert len(summary["runs"]) == 2
    assert Path(summary["plot_manifest_path"]).exists()

    for run in summary["runs"]:
        assert Path(run["raw_path"]).exists()
        assert run["num_examples"] == 2
        assert "k" in run
        assert "accuracy" in run

    # Validate raw JSONL records
    for run in summary["runs"]:
        lines = Path(run["raw_path"]).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "candidates" in record
        assert "vote_distribution" in record
        assert "metrics" in record

    manifest = json.loads(Path(summary["plot_manifest_path"]).read_text(encoding="utf-8"))
    expected_plots = {
        "accuracy",
        "avg_total_tokens",
        "parse_success_rate",
        "avg_candidate_parse_success_rate",
    }
    assert set(manifest) == expected_plots
    for plot_path in manifest.values():
        assert Path(plot_path).exists()
