"""Tests for EAR components and full pipeline.

All tests use fake models — no GPU or model downloads required.
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class FakePrompt:
    def format(self, **kwargs) -> str:
        return "prompt"


class FakeGenerationModel:
    """Returns a fixed generation string for every call."""

    def __init__(self, response: str = "Step 1. The answer is 42"):
        self._response = response
        self.call_count = 0

    def generate(self, prompt, **kwargs) -> str:
        self.call_count += 1
        return self._response

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeParser:
    def parse(self, text):
        marker = "The answer is "
        if marker in text:
            start = text.index(marker)
            return text[start + len(marker) :].strip(), start
        return None, len(text)


class FakeJudgeModel:
    """Always returns VALID so every critique is judged successful."""

    def generate(self, prompt, **kwargs) -> str:
        return "VALID - the critique identifies a real error."

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeInvalidJudgeModel:
    """Always returns INVALID — no critiques succeed."""

    def generate(self, prompt, **kwargs) -> str:
        return "INVALID - the candidate's answer is correct."

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# ---------------------------------------------------------------------------
# JudgeParser
# ---------------------------------------------------------------------------


def test_judge_parser_valid():
    from src.common.parsing.ear_judge import JudgeParser

    p = JudgeParser()
    assert p.parse("VALID - good critique") == (True, 0.9)


def test_judge_parser_invalid():
    from src.common.parsing.ear_judge import JudgeParser

    p = JudgeParser()
    assert p.parse("INVALID - the candidate is correct") == (False, 0.9)


def test_judge_parser_invalid_takes_priority_over_valid_substring():
    from src.common.parsing.ear_judge import JudgeParser

    p = JudgeParser()
    # "INVALID" contains "VALID" — parser must match INVALID first
    is_successful, _ = p.parse("INVALID")
    assert is_successful is False


def test_judge_parser_no_signal_defaults_false():
    from src.common.parsing.ear_judge import JudgeParser

    p = JudgeParser()
    is_successful, confidence = p.parse("The answer seems reasonable.")
    assert is_successful is False
    assert confidence == 0.0


# ---------------------------------------------------------------------------
# Critique strategies
# ---------------------------------------------------------------------------


def test_answer_level_produces_k_critiques():
    from src.methods.ear.critique.answer_level import AnswerLevelCritique

    candidates = [
        {"id": 0, "extracted_answer": "42", "reasoning_text": ""},
        {"id": 1, "extracted_answer": "37", "reasoning_text": ""},
        {"id": 2, "extracted_answer": "42", "reasoning_text": ""},
    ]
    strategy = AnswerLevelCritique(critique_prompt=FakePrompt())
    results = strategy.run("What is x?", candidates, FakeGenerationModel())

    assert len(results) == 3
    for r in results:
        assert r.critiquer_id is None
        assert r.target_id in {0, 1, 2}
        assert r.tokens > 0


def test_reasoning_level_produces_k_critiques():
    from src.methods.ear.critique.reasoning_level import ReasoningLevelCritique

    candidates = [{"id": i, "extracted_answer": str(i), "reasoning_text": f"step {i}"} for i in range(3)]
    strategy = ReasoningLevelCritique(critique_prompt=FakePrompt())
    results = strategy.run("What is x?", candidates, FakeGenerationModel())

    assert len(results) == 3
    assert all(r.critiquer_id is None for r in results)


def test_panel_produces_k_times_k_minus_1_critiques():
    from src.methods.ear.critique.panel import PanelCritique

    k = 3
    candidates = [{"id": i, "extracted_answer": str(i), "reasoning_text": f"step {i}"} for i in range(k)]
    strategy = PanelCritique(critique_prompt=FakePrompt())
    results = strategy.run("What is x?", candidates, FakeGenerationModel())

    assert len(results) == k * (k - 1)
    # All critiquer_ids should be meaningful (not None)
    assert all(r.critiquer_id is not None for r in results)
    # No self-critiques
    assert all(r.critiquer_id != r.target_id for r in results)


# ---------------------------------------------------------------------------
# Equilibrium selectors
# ---------------------------------------------------------------------------


def test_survival_winner_is_least_critiqued():
    from src.methods.ear.equilibrium.survival import SurvivalSelector
    from src.methods.ear.types import JudgeResult

    sel = SurvivalSelector()
    sel.initialize([0, 1, 2])

    # Candidate 0 gets 2 successful critiques, 1 gets 1, 2 gets 0
    sel.update([
        JudgeResult(critiquer_id=None, target_id=0, is_successful=True, confidence=0.9, judge_text="", tokens=1),
        JudgeResult(critiquer_id=None, target_id=0, is_successful=True, confidence=0.9, judge_text="", tokens=1),
        JudgeResult(critiquer_id=None, target_id=1, is_successful=True, confidence=0.9, judge_text="", tokens=1),
    ])

    candidates = [{"id": i} for i in range(3)]
    assert sel.select_winner_id(candidates) == 2
    assert sel.is_converged() is False


def test_elo_winner_has_highest_rating():
    from src.methods.ear.equilibrium.elo import EloSelector
    from src.methods.ear.types import JudgeResult

    sel = EloSelector()
    sel.initialize([0, 1])

    # Candidate 0 successfully critiques candidate 1 (panel-style: critiquer_id=0)
    sel.update([
        JudgeResult(critiquer_id=0, target_id=1, is_successful=True, confidence=0.9, judge_text="", tokens=1),
    ])

    candidates = [{"id": 0}, {"id": 1}]
    assert sel.select_winner_id(candidates) == 0
    scores = sel.get_scores()
    assert scores[0] > scores[1]
    assert sel.is_converged() is False


def test_elo_external_critic_only_penalizes_target():
    from src.methods.ear.equilibrium.elo import EloSelector
    from src.methods.ear.types import JudgeResult

    sel = EloSelector()
    sel.initialize([0, 1])

    initial_scores = sel.get_scores().copy()

    # External critic (critiquer_id=None) successfully critiques candidate 0
    sel.update([
        JudgeResult(critiquer_id=None, target_id=0, is_successful=True, confidence=0.9, judge_text="", tokens=1),
    ])

    scores_after = sel.get_scores()
    assert scores_after[0] < initial_scores[0]   # target penalized
    assert scores_after[1] == initial_scores[1]  # untouched candidate unchanged


def test_nash_converges_when_no_revisions():
    from src.methods.ear.equilibrium.nash import NashSelector
    from src.methods.ear.types import JudgeResult, RevisionResult

    sel = NashSelector()
    sel.initialize([0, 1, 2])

    sel.update([
        JudgeResult(critiquer_id=None, target_id=0, is_successful=True, confidence=0.9, judge_text="", tokens=1),
    ])

    # No revisions → converged
    sel.notify_revisions([])
    assert sel.is_converged() is True


def test_nash_not_converged_when_answer_changed():
    from src.methods.ear.equilibrium.nash import NashSelector
    from src.methods.ear.types import JudgeResult, RevisionResult

    sel = NashSelector()
    sel.initialize([0, 1])

    sel.update([])
    sel.notify_revisions([
        RevisionResult(
            candidate_id=0, old_answer="42", new_answer="37",
            revised_raw="", reasoning_text="", answer_text="",
            tokens=10, answer_changed=True,
        )
    ])
    assert sel.is_converged() is False


# ---------------------------------------------------------------------------
# Full EAR method — result structure
# ---------------------------------------------------------------------------


def _make_ear(model, judge_model, critique_strategy_name, selector_name, k=2, rounds=1, allow_revision=True):
    from src.common.parsing.ear_judge import JudgeParser
    from src.methods.EAR import EAR
    from src.methods.ear.judge import Judge
    from src.registry import CRITIQUE_STRATEGIES, EQUILIBRIUM_SELECTORS

    import src.common.prompts.ear_prompts  # noqa: F401

    critique_strategy = CRITIQUE_STRATEGIES.get(critique_strategy_name)(
        critique_prompt=FakePrompt()
    )
    equilibrium_selector = EQUILIBRIUM_SELECTORS.get(selector_name)()
    judge = Judge(model=judge_model, judge_prompt=FakePrompt(), judge_parser=JudgeParser())

    return EAR(
        model=model,
        prompt_template=FakePrompt(),
        parser=FakeParser(),
        critique_strategy=critique_strategy,
        equilibrium_selector=equilibrium_selector,
        judge=judge,
        revision_prompt=FakePrompt(),
        k=k,
        rounds=rounds,
        allow_revision=allow_revision,
    )


def test_ear_result_structure_answer_level_elo():
    ear = _make_ear(
        model=FakeGenerationModel("Step 1. The answer is 42"),
        judge_model=FakeJudgeModel(),
        critique_strategy_name="answer_level",
        selector_name="elo",
        k=2,
        rounds=2,
    )
    instance = {"id": "ex-1", "question": "1+1", "answer": "2", "dataset": "gsm8k", "split": "test"}
    result = ear.run(instance, max_new_tokens=64)

    # Top-level fields
    assert "extracted_answer" in result
    assert "parse_success" in result
    assert "initial_candidates" in result
    assert "rounds" in result
    assert "final_scores" in result
    assert "selected_candidate_id" in result
    assert "run_config" in result
    assert "metrics" in result

    assert len(result["initial_candidates"]) == 2
    assert len(result["rounds"]) == 2

    # Round structure
    for rnd in result["rounds"]:
        assert "round_num" in rnd
        assert "critiques" in rnd
        assert "judge_results" in rnd
        assert "revisions" in rnd
        assert "scores_after_round" in rnd

    # Metrics
    m = result["metrics"]
    assert m["total_model_calls"] > 0
    assert m["total_critiques"] > 0
    assert m["successful_critiques"] >= 0

    # run_config
    rc = result["run_config"]
    assert rc["k"] == 2
    assert rc["rounds"] == 2
    assert rc["critique_strategy"] == "AnswerLevelCritique"
    assert rc["equilibrium_selector"] == "EloSelector"


def test_ear_no_revision_produces_no_revisions():
    ear = _make_ear(
        model=FakeGenerationModel("Step 1. The answer is 42"),
        judge_model=FakeJudgeModel(),
        critique_strategy_name="answer_level",
        selector_name="survival",
        k=2,
        rounds=1,
        allow_revision=False,
    )
    instance = {"id": "ex-1", "question": "1+1", "answer": "2"}
    result = ear.run(instance, max_new_tokens=64)

    total_revisions = sum(len(r["revisions"]) for r in result["rounds"])
    assert total_revisions == 0


def test_nash_converges_early_with_no_revisions():
    ear = _make_ear(
        model=FakeGenerationModel("Step 1. The answer is 42"),
        judge_model=FakeInvalidJudgeModel(),  # no successful critiques → no revisions
        critique_strategy_name="reasoning_level",
        selector_name="nash",
        k=2,
        rounds=5,
        allow_revision=True,
    )
    instance = {"id": "ex-1", "question": "1+1", "answer": "2"}
    result = ear.run(instance, max_new_tokens=64)

    # With no successful critiques, no revisions happen → Nash converges after round 1
    assert result["metrics"]["rounds_until_convergence"] == 1
    assert len(result["rounds"]) == 1


# ---------------------------------------------------------------------------
# Full EAR sweep harness
# ---------------------------------------------------------------------------


def test_ear_sweep_writes_results(tmp_path):
    from src.experiments.gsm8k_ear_sweep import EarRunConfig, run_gsm8k_ear_sweep
    from src.common.parsing.ear_judge import JudgeParser

    import src.common.prompts.ear_prompts  # noqa: F401

    instances = [
        {"id": "ex-1", "question": "1+1", "answer": "2"},
        {"id": "ex-2", "question": "2+2", "answer": "4"},
    ]

    def model_factory():
        return FakeGenerationModel("Reasoning. The answer is 42")

    ear_configs = [
        EarRunConfig(k=2, rounds=1, critique_strategy="answer_level",
                     equilibrium_selector="elo", allow_revision=False, max_new_tokens=64),
        EarRunConfig(k=2, rounds=1, critique_strategy="answer_level",
                     equilibrium_selector="survival", allow_revision=False, max_new_tokens=64),
    ]

    from src.registry import PROMPTS
    critique_prompts = {"answer_level": PROMPTS.get("ear_answer_level_critique")()}
    judge_prompt = PROMPTS.get("ear_judge")()
    revision_prompt = PROMPTS.get("ear_revision")()
    judge_parser = JudgeParser()

    summary = run_gsm8k_ear_sweep(
        model_factory=model_factory,
        instances=instances,
        prompt_template=FakePrompt(),
        parser=FakeParser(),
        critique_prompts=critique_prompts,
        judge_prompt=judge_prompt,
        judge_parser=judge_parser,
        revision_prompt=revision_prompt,
        ear_configs=ear_configs,
        experiment_name="gsm8k_ear_test",
        base_dir=str(tmp_path),
        progress=False,
    )

    assert summary["num_configs"] == 2
    assert len(summary["runs"]) == 2

    for run in summary["runs"]:
        assert Path(run["raw_path"]).exists()
        assert run["num_examples"] == 2
        lines = Path(run["raw_path"]).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        record = json.loads(lines[0])
        assert "rounds" in record
        assert "final_scores" in record
        assert "initial_candidates" in record
        assert "metrics" in record
