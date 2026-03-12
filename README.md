# Selecting the Hardest-to-Refute Answer: Equilibrium-Based Aggregation for LLM Reasoning

This project studies inference-time aggregation methods for reasoning LLMs.

The central idea is to compare standard chain-of-thought baselines against an equilibrium-based aggregation method that selects answers by robustness to adversarial refutation rather than majority agreement alone.

The current repository includes the first implementation milestone for this project: a GSM8K `SingleCoT` budget-sweep harness with persisted results, plots, and tests. That milestone is infrastructure for the broader research project, not the project definition itself.

## Current Status

Implemented so far:

- GSM8K dataset adapter
- GSM8K zero-shot CoT prompt
- GSM8K answer parser
- `SingleCoT` reasoning method
- budget-sweep experiment runner for `max_new_tokens`
- result logging to `results/raw` and `results/summaries`
- plot generation to `results/plots`
- deterministic test harness using fake models

Planned next:

- `Self-Consistency / Best-of-N`
- `EAR`
- additional datasets such as LogiQA and ARC-style QA benchmarks
- richer calibration and robustness analysis

## Research Goal

The project goal is to evaluate whether equilibrium-based aggregation can improve reasoning reliability over standard inference-time baselines.

We will benchmark multiple reasoning methods under a shared evaluation setup and compare them on accuracy, robustness, calibration, and compute tradeoffs.

## Methods To Benchmark

The benchmark suite is planned around these algorithms:

- `SingleCoT`
  One reasoning path per question. This is the baseline.

- `Self-Consistency / Best-of-N`
  Multiple independently sampled reasoning paths per question, with the final answer chosen by majority vote.

- `EAR`
  Equilibrium Aggregation Reasoning. Multiple candidate reasoning paths iteratively critique and challenge one another, and the selected answer is the one that is hardest to refute under a bounded critique budget.

## Datasets

The intended benchmark datasets are:

- `GSM8K`
  Grade-school math word problems with free-form numeric answers.

- `LogiQA`
  Logical reasoning multiple-choice reading comprehension.

- `ARC-Easy`
  Easier split of AI2 ARC science multiple-choice questions.

- `ARC-Challenge`
  Harder split of AI2 ARC science multiple-choice questions.

Current implementation status:

- implemented now: `GSM8K`
- planned next: `LogiQA`, `ARC-Easy`, `ARC-Challenge`

## Repository Layout

```text
.
├── benchmarks/
│   ├── run_gsm8k_cot.py
│   ├── run_gsm8k_single_cot_sweep.py
│   └── analyze_results.py
├── results/
│   ├── raw/
│   ├── summaries/
│   └── plots/
├── src/
│   ├── common/
│   │   ├── datasets/
│   │   ├── eval/
│   │   ├── models/
│   │   ├── parsing/
│   │   ├── prompts/
│   │   └── utils/
│   ├── experiments/
│   ├── methods/
│   └── registry.py
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

## Current Implemented Experiment

The current implemented experiment is:

- dataset: `GSM8K`
- model: `google/gemma-3-1b-it`
- method: `SingleCoT`
- sweep variable: `max_new_tokens`

## Running the GSM8K SingleCoT Sweep

The main experimental entrypoint is:

```bash
uv run python benchmarks/run_gsm8k_single_cot_sweep.py
```

Example with a custom budget sweep and a smaller evaluation slice:

```bash
uv run python benchmarks/run_gsm8k_single_cot_sweep.py \
  --budgets 8,16,32,64,128,256,512,1000 \
  --num_examples 50
```

Supported arguments:

- `--budgets`: comma-separated `max_new_tokens` values
- `--num_examples`: number of examples to evaluate, `0` means full split
- `--split`: dataset split, default is `test`
- `--base_dir`: output directory, default is `results`
- `--temperature`: generation temperature, default is `0.0`

## Results Layout

The harness writes outputs into `results/` automatically.

- `results/raw/`: one JSONL file per budget run with per-example outputs
- `results/summaries/`: one summary JSON per run and one aggregate summary for the sweep
- `results/plots/`: generated PNG plots and a plot manifest JSON

For each budget run, the raw JSONL includes:

- question text
- model generation
- extracted answer
- ground truth
- correctness flag
- parse success flag
- token usage metrics
- run configuration such as `max_new_tokens`

The aggregate sweep currently generates:

- accuracy vs token budget
- average reasoning tokens vs token budget
- parse success rate vs token budget

## Evaluation Plan

The project is intended to compare methods using:

- standard benchmark accuracy
- refutation robustness under increasing critique budgets
- confident error rate
- calibration metrics such as ECE
- compute vs performance tradeoffs

The current codebase only implements part of this plan. Accuracy-oriented GSM8K evaluation is in place first; the more advanced robustness and calibration analysis will be added as the benchmark suite expands.

## Design Contracts

These interfaces should stay stable unless the team agrees on a change.

### Dataset adapters

Dataset loaders should emit normalized instances with:

- `id`
- `dataset`
- `split`
- `question`
- `answer`
- optional dataset-specific fields such as `raw_answer` or answer choices

### Reasoning methods

Methods should return structured outputs rather than just a final answer. This is important because later methods such as self-consistency and EAR will need:

- intermediate traces
- critique/refutation history
- confidence-like signals
- token and cost metadata

### Model adapters

Model wrappers should expose:

- `generate(prompt, **kwargs)`
- `count_tokens(text)`

The current budget sweep uses `max_new_tokens` overrides per run.

## Testing Philosophy

Tests are designed to validate harness behavior without downloading or running external models.

Current test coverage includes:

- prompt formatting
- answer extraction
- registry behavior
- `SingleCoT` result structure
- end-to-end GSM8K budget sweep with a fake model

This keeps CI fast and lets contributors work on logic before running expensive model experiments.

## Contribution Guidelines

When contributing:

- use `uv` for commands and tests
- keep experiment outputs structured and machine-readable
- prefer extending the shared harness over adding one-off scripts
- add or update tests when changing parsing, evaluation, or method contracts
- avoid coupling new method implementations directly to a single benchmark script

Before opening a pull request:

- make sure your branch is up to date with `main`
- keep PRs focused on one logical change
- include a short description of the problem, approach, and validation
- do not commit generated files from `results/` unless the team explicitly wants a curated artifact
- if you change public interfaces or result schemas, document that change in the README or PR description

Recommended workflow:

1. sync the environment with `uv sync --dev`
2. make your change in `src/`
3. add or update tests in `tests/`
4. run `uv run pre-commit run --all-files`
5. run `uv run pytest -q`
6. run a small benchmark slice before a larger experiment

PR review checklist:

- code follows the shared dataset/method/model contracts
- tests cover the new behavior or changed behavior
- `pre-commit` and `pytest` pass locally
- documentation is updated when contributor-facing behavior changes
- changes do not silently break existing experiment outputs or evaluation paths

## PR Workflow

Pull requests are expected to pass the repository checks before merge.

The CI workflow in [.github/workflows/pr-checks.yml](/home/shashwat/Desktop/LLM-Reasoning/.github/workflows/pr-checks.yml) runs on each pull request and on pushes to `main`. It performs:

- `uv sync --dev`
- `uv run pre-commit run --all-files`
- `uv run pytest -q`

This gives the team one consistent gate for formatting, linting, basic repository hygiene, and deterministic test coverage.

## Known Gaps

- `Self-Consistency / Best-of-N` is not implemented yet
- `EAR` is not implemented yet
- current benchmark coverage is GSM8K-focused
- LogiQA and ARC dataset adapters are not implemented yet
- the real model experiment path is separate from deterministic tests, so passing tests does not guarantee model-download availability on a new machine

## Suggested Next Steps

- implement `Self-Consistency / Best-of-N` on top of the current `SingleCoT` contract
- add a reusable multiple-choice dataset/evaluator path for LogiQA and ARC-style QA
- add calibration and robustness metrics to the aggregate summaries
- standardize experiment configs so runs can be reproduced from a manifest rather than CLI flags alone
