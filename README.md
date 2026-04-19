# Selecting the Hardest-to-Refute Answer: Equilibrium-Based Aggregation for LLM Reasoning

This project studies inference-time aggregation methods for reasoning LLMs.

The central idea is to compare standard chain-of-thought baselines against an equilibrium-based aggregation method that selects answers by robustness to adversarial refutation rather than majority agreement alone.

## Current Status

Implemented:

- GSM8K dataset adapter
- GSM8K zero-shot CoT prompt and answer parser
- `SingleCoT` reasoning method with budget sweep
- `SelfConsistency` reasoning method (k samples + majority vote with numeric-aware canonicalization)
- `EAR` (Equilibrium Aggregation Reasoning) with pluggable critique strategies and equilibrium selectors
- Config-driven unified experiment runner (`benchmarks/run_experiment.py`)
- Result logging to `results/raw` and `results/summaries`, plot generation to `results/plots`
- Side-by-side method comparison script (`benchmarks/compare_methods.py`)
- Analysis scripts: scale curve, Pareto frontier, ablation heatmap, efficiency table
- Sanity-check script with per-model accuracy floor validation (`benchmarks/sanity_check.py`)
- 31 deterministic tests covering all components

### Experiment Results (GSM8K, qwen3-0.6B, 100 examples)

All runs use Qwen3-0.6B in non-thinking mode (`/no_think`) on an NVIDIA RTX A6000.

**SingleCoT** (greedy, budget sweep):

| Budget (tokens) | Accuracy | Avg tokens used | Parse rate |
|----------------|----------|-----------------|------------|
| 512 | 57.0% | 408 | 100% |
| 1000 | 69.7% | 523 | 100% |

**Self-Consistency** (temperature=0.7, budget=512 per candidate):

| k | Accuracy | Avg total tokens |
|---|----------|-----------------|
| 3 | 55.0% | 1248 |
| 5 | ~59% (partial) | — |

EAR experiments are scheduled to run next across all four Qwen3 model sizes (0.6B, 1.7B, 4B, 8B).

Planned next:

- Additional datasets: LogiQA, ARC-Easy, ARC-Challenge
- Richer calibration and robustness metrics (ECE, confident error rate)
- Larger-scale sweeps across model sizes (4b → 32b)

## Research Goal

The project goal is to evaluate whether equilibrium-based aggregation can improve reasoning reliability over standard inference-time baselines.

Methods are benchmarked under a shared evaluation setup and compared on accuracy, robustness, calibration, and compute tradeoffs.

## Methods

### `SingleCoT`

One reasoning path per question. Sweeps over `max_new_tokens` budget. This is the baseline.

### `SelfConsistency`

k independently sampled reasoning paths per question. Final answer selected by majority vote. Sweeps over k.

### `EAR`

Equilibrium Aggregation Reasoning. Generates k candidate reasoning paths, then runs iterative critique rounds. Each critique is evaluated by a separate judge model call. The final answer is selected by an equilibrium selector that scores candidates based on how well they withstand critique.

**Critique strategies** (pluggable via `CRITIQUE_STRATEGIES` registry):

- `answer_level` — external critic evaluates each candidate's final answer independently (k calls/round)
- `reasoning_level` — external critic evaluates each candidate's full reasoning chain (k calls/round)
- `panel` — cross-candidate critique: each candidate sees all others' solutions and critiques every other candidate (k×(k-1) calls/round)

**Equilibrium selectors** (pluggable via `EQUILIBRIUM_SELECTORS` registry):

- `elo` — Elo rating system; successful critique transfers rating from target to critiquer (panel) or penalizes target against a reference (external critics); highest Elo wins
- `survival` — winner is the candidate with fewest successful critiques against it
- `nash` — fixed-point convergence: rounds continue until no candidate revises its answer; winner is the most stable candidate at convergence

**Revision** is optional. When enabled, a candidate that receives a successful critique generates a revised answer before the next round.

## Datasets

| Dataset | Status | Description |
|---|---|---|
| `GSM8K` | Implemented | Grade-school math word problems, free-form numeric answers |
| `LogiQA` | Planned | Logical reasoning multiple-choice |
| `ARC-Easy` | Planned | AI2 ARC science questions, easy split |
| `ARC-Challenge` | Planned | AI2 ARC science questions, hard split |

## Repository Layout

```text
.
├── benchmarks/
│   ├── run_experiment.py            # unified config-driven runner (main entrypoint)
│   ├── compare_methods.py           # side-by-side method comparison
│   ├── run_gsm8k_single_cot_sweep.py
│   └── analyze_results.py
├── configs/
│   ├── gsm8k_single_cot.yaml
│   ├── gsm8k_self_consistency.yaml
│   ├── gsm8k_ear.yaml               # full EAR grid sweep
│   ├── compare_ear.yaml             # single EAR config for comparison
│   └── compare_sc.yaml             # matching SC config for comparison
├── results/
│   ├── raw/                         # per-instance JSONL, one file per run
│   ├── summaries/                   # aggregate metrics JSON per run
│   └── plots/                       # PNG plots and plot manifests
├── src/
│   ├── common/
│   │   ├── datasets/                # dataset adapters (GSM8K)
│   │   ├── eval/                    # metrics, voting utilities
│   │   ├── models/                  # HuggingFace model wrappers
│   │   ├── parsing/                 # answer parsers (GSM8K, EAR judge)
│   │   ├── prompts/                 # prompt templates (CoT, EAR critique/judge/revision)
│   │   └── utils/                   # storage, experiment tracking
│   ├── experiments/
│   │   ├── gsm8k_single_cot_sweep.py
│   │   ├── gsm8k_self_consistency_sweep.py
│   │   └── gsm8k_ear_sweep.py
│   ├── methods/
│   │   ├── Base.py
│   │   ├── SingleCoT.py
│   │   ├── SelfConsistency.py
│   │   ├── EAR.py
│   │   └── ear/
│   │       ├── types.py             # CritiqueResult, JudgeResult, RevisionResult
│   │       ├── judge.py             # Judge wrapper
│   │       ├── critique/            # answer_level, reasoning_level, panel
│   │       └── equilibrium/        # elo, nash, survival
│   └── registry.py                  # MODELS, DATASETS, PROMPTS, PARSERS, METHODS,
│                                    # CRITIQUE_STRATEGIES, EQUILIBRIUM_SELECTORS
└── tests/
```

## Environment

This project uses:

- Python `3.13+`
- [`uv`](https://docs.astral.sh/uv/) for environment and command execution

Install dependencies with:

```bash
uv sync --dev
```

Run tests with:

```bash
uv run pytest -q
```

Install the local commit hooks with:

```bash
uv run pre-commit install
```

## Running Experiments

All experiments are run through the unified config-driven runner. Teams pick a config, optionally override the model, and run one command.

```bash
uv run python benchmarks/run_experiment.py --config configs/<config>.yaml
```

Override the model or dataset size without editing the config:

```bash
uv run python benchmarks/run_experiment.py \
  --config configs/gsm8k_self_consistency.yaml \
  --model qwen3-0.6b \
  --num_examples 100
```

Any list-valued parameter under `method:` in the YAML automatically triggers a sweep over those values. The runner builds the full Cartesian product for EAR.

### SingleCoT budget sweep

```bash
uv run python benchmarks/run_experiment.py --config configs/gsm8k_single_cot.yaml
```

Or using the legacy script directly:

```bash
uv run python benchmarks/run_gsm8k_single_cot_sweep.py --budgets 64,128,256,512 --num_examples 50
```

### Self-Consistency k sweep

```bash
uv run python benchmarks/run_experiment.py --config configs/gsm8k_self_consistency.yaml
```

### EAR full grid sweep

```bash
uv run python benchmarks/run_experiment.py --config configs/gsm8k_ear.yaml
```

The default `gsm8k_ear.yaml` sweeps 3 critique strategies × 3 selectors × 2 revision settings × 2 k values × 2 round values = 72 configurations. Reduce the lists in the config for faster iteration.

### Comparing two methods

After running both experiments, pass the raw JSONL paths to the comparison script:

```bash
uv run python benchmarks/compare_methods.py \
  --ear results/raw/<ear_run>.jsonl \
  --sc  results/raw/<sc_run>.jsonl \
  --output_dir results/plots
```

## Registered Models

| Registry key | Model |
|---|---|
| `gemma-3-1b-it` | `google/gemma-3-1b-it` (gated) |
| `qwen3-0.6b` | `Qwen/Qwen3-0.6B` |

To add a new model, register it in `src/common/models/huggingface.py`:

```python
@MODELS.register("my-model")
def load_my_model(**kwargs) -> HuggingFaceModelWrapper:
    return HuggingFaceModelWrapper("org/model-name", **kwargs)
```

Then reference `my-model` in any config file or pass `--model my-model` on the CLI.

## Results Layout

The harness writes all outputs into `results/` automatically.

- `results/raw/` — one JSONL per run; each line is one instance with full trace
- `results/summaries/` — one aggregate JSON per run and one for the full sweep
- `results/plots/` — PNG plots and a plot manifest JSON

### SingleCoT / SC per-instance fields

- question, model generation, extracted answer, ground truth, correctness flag
- parse success flag, token usage metrics, run configuration

### EAR per-instance fields

- question, ground truth, run configuration
- `initial_candidates` — k candidates before any critique
- `rounds` — per-round list of critiques, judge verdicts, revisions, and scores
- `final_scores` — selector scores for each candidate
- `selected_candidate_id`, `extracted_answer`, `parse_success`
- `metrics` — total tokens broken down by generation / critique / judge / revision, model calls, rounds until convergence

## Evaluation Plan

The project compares methods on:

- standard benchmark accuracy
- refutation robustness under increasing critique budgets
- confident error rate
- calibration metrics such as ECE
- compute vs performance tradeoffs (accuracy per token, accuracy per model call)

Accuracy-oriented GSM8K evaluation is in place. Robustness and calibration analysis will be added as the benchmark suite expands.

## Design Contracts

These interfaces should stay stable unless the team agrees on a change.

### Dataset adapters

Dataset loaders emit normalized instances with:

- `id`, `dataset`, `split`, `question`, `answer`
- optional dataset-specific fields such as `raw_answer` or answer choices

### Reasoning methods

Methods return structured dictionaries, not just a final answer, so that downstream analysis has access to:

- intermediate traces and generations
- critique and refutation history (EAR)
- token and cost metadata

### Model adapters

Model wrappers expose:

- `generate(prompt, **kwargs) -> str`
- `count_tokens(text) -> int`

### Critique strategies

Implement `BaseCritiqueStrategy` from `src/methods/ear/critique/base.py` and register with `@CRITIQUE_STRATEGIES.register("key")`.

### Equilibrium selectors

Implement `BaseEquilibriumSelector` from `src/methods/ear/equilibrium/base.py` and register with `@EQUILIBRIUM_SELECTORS.register("key")`.

## Testing Philosophy

Tests validate harness behavior without downloading or running external models. All components use fake models, parsers, and prompts in tests.

Current coverage (31 tests):

- prompt formatting and answer extraction
- registry behavior and duplicate prevention
- majority vote with numeric canonicalization
- `SingleCoT` and `SelfConsistency` result structure and token metrics
- EAR judge parser (VALID/INVALID extraction)
- all three critique strategies (output count and critiquer_id semantics)
- all three equilibrium selectors (score logic and convergence)
- end-to-end EAR run structure and per-instance JSONL fields
- full budget sweep and k sweep with fake models

## Contribution Guidelines

When contributing:

- use `uv` for commands and tests
- keep experiment outputs structured and machine-readable
- prefer extending the shared harness over adding one-off scripts
- add or update tests when changing parsing, evaluation, or method contracts
- avoid coupling new method implementations directly to a single benchmark script
- register new models in `src/common/models/huggingface.py` and reference by registry key in configs

Recommended workflow:

1. sync the environment with `uv sync --dev`
2. make your change in `src/`
3. add or update tests in `tests/`
4. run `uv run pre-commit run --all-files`
5. run `uv run pytest -q`
6. run a small benchmark slice (`--num_examples 20`) before a larger experiment

PR review checklist:

- code follows the shared dataset/method/model contracts
- tests cover the new or changed behavior
- `pre-commit` and `pytest` pass locally
- documentation is updated when contributor-facing behavior changes
- changes do not silently break existing experiment outputs or evaluation paths

## PR Workflow

Pull requests are expected to pass repository checks before merge.

The CI workflow runs on each pull request and on pushes to `main`:

- `uv sync --dev`
- `uv run pre-commit run --all-files`
- `uv run pytest -q`

## Known Gaps

- LogiQA and ARC dataset adapters are not implemented yet
- calibration metrics (ECE, confident error rate) are not yet computed
- large-scale model size sweeps (4b → 32b) have not been run yet
- the real model experiment path is separate from deterministic tests, so passing tests does not guarantee model availability on a new machine

## TODO

### Scope

Central claim: *EAR outperforms self-consistency at matched token budgets on math and logic reasoning, and the gap grows with model scale.*

Models: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B (same family, clean scale curve)
Benchmarks: GSM8K, LogiQA
EAR: zero-shot only — EAR-RL and DPO flywheel are future work

### Infrastructure

- [ ] Register `qwen3-1.7b` in `src/common/models/huggingface.py`
- [ ] Register `qwen3-4b` in `src/common/models/huggingface.py`
- [ ] Register `qwen3-8b` in `src/common/models/huggingface.py`
- [ ] Add LogiQA dataset adapter in `src/common/datasets/logiqa.py`
- [ ] Add MCQ letter extractor (A–D) in `src/common/parsing/`
- [ ] Add MCQ reasoning prompt in `src/common/prompts/`
- [ ] Add LogiQA configs: `configs/logiqa_single_cot.yaml`, `configs/logiqa_self_consistency.yaml`, `configs/logiqa_ear.yaml`

### Experiments

- [ ] Run full GSM8K sweep (SingleCoT, SC k∈{3,5,8}, EAR all 9 strategy×selector combos k∈{3,5} R∈{1,3}) on Qwen3-0.6B
- [ ] Run full GSM8K sweep on Qwen3-1.7B
- [ ] Run full GSM8K sweep on Qwen3-4B
- [ ] Run full GSM8K sweep on Qwen3-8B
- [ ] Run full LogiQA sweep on Qwen3-0.6B
- [ ] Run full LogiQA sweep on Qwen3-1.7B
- [ ] Run full LogiQA sweep on Qwen3-4B
- [ ] Run full LogiQA sweep on Qwen3-8B

### Analysis

- [ ] Pareto frontier plot: accuracy vs total tokens per instance, one point per (method, config), both benchmarks
- [ ] Scale curve: accuracy vs model size for best EAR config vs best SC config, both benchmarks
- [ ] Ablation heatmap: strategy × selector accuracy grid at k=5, R=3 (GSM8K + LogiQA)
- [ ] Token efficiency table: accuracy, tokens/instance, calls/instance, acc/kToken for all methods
- [ ] Critique success rate breakdown by strategy and model size

### Paper

- [ ] Update benchmark table to GSM8K + LogiQA only (remove planned datasets)
- [ ] Update model table to Qwen3-0.6B / 1.7B / 4B / 8B
- [ ] Trim contributions list to EAR zero-shot only; move EAR-RL and DPO to future work
- [ ] Fill in result tables once experiments complete
- [ ] Insert generated plots into paper
- [ ] Write analysis section based on actual results
- [ ] Write conclusion
