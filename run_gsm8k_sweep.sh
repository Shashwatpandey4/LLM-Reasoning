#!/bin/bash
#SBATCH --job-name=gsm8k_sweep
#SBATCH --output=logs/gsm8k_sweep_%j.out
#SBATCH --error=logs/gsm8k_sweep_%j.err
#SBATCH --time=55:00:00
#SBATCH --partition=gpu            # TODO: confirm Sol GPU partition name
#SBATCH --gres=gpu:a100:1          # TODO: confirm GPU type available on Sol
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# ── Environment ──────────────────────────────────────────────────────────────
# Set HuggingFace cache to scratch so models persist across jobs
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME

# uv needs to be on PATH — adjust if Sol uses modules
export PATH="$HOME/.local/bin:$PATH"

# Project directory — update to wherever you cloned the repo on Sol
PROJECT_DIR="$HOME/LLM-Reasoning"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found"; exit 1; }

mkdir -p logs results/raw results/summaries results/plots

# ── Helper ───────────────────────────────────────────────────────────────────
run() {
    local desc="$1"; shift
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  START: $desc"
    echo "  $(date)"
    echo "════════════════════════════════════════════════════════"
    uv run python benchmarks/run_experiment.py "$@"
    local exit_code=$?
    echo "  DONE:  $desc  (exit $exit_code)  $(date)"
    return $exit_code
}

# ── qwen3-0.6b ───────────────────────────────────────────────────────────────
# SingleCoT already done from previous session — skip
run "0.6b | SC k=[3,5]" \
    --config configs/gsm8k_sc_fast.yaml --model qwen3-0.6b

run "0.6b | EAR (6 configs)" \
    --config configs/gsm8k_ear_fast.yaml --model qwen3-0.6b

# ── qwen3-1.7b ───────────────────────────────────────────────────────────────
run "1.7b | SingleCoT" \
    --config configs/gsm8k_single_cot_fast.yaml --model qwen3-1.7b

run "1.7b | SC k=[3,5]" \
    --config configs/gsm8k_sc_fast.yaml --model qwen3-1.7b

run "1.7b | EAR (6 configs)" \
    --config configs/gsm8k_ear_fast.yaml --model qwen3-1.7b

# ── qwen3-4b ─────────────────────────────────────────────────────────────────
run "4b | SingleCoT" \
    --config configs/gsm8k_single_cot_fast.yaml --model qwen3-4b

run "4b | SC k=[3,5]" \
    --config configs/gsm8k_sc_fast.yaml --model qwen3-4b

run "4b | EAR (6 configs)" \
    --config configs/gsm8k_ear_fast.yaml --model qwen3-4b

# ── qwen3-8b ─────────────────────────────────────────────────────────────────
run "8b | SingleCoT" \
    --config configs/gsm8k_single_cot_fast.yaml --model qwen3-8b

run "8b | SC k=[3,5]" \
    --config configs/gsm8k_sc_fast.yaml --model qwen3-8b

run "8b | EAR (6 configs)" \
    --config configs/gsm8k_ear_fast.yaml --model qwen3-8b

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ALL EXPERIMENTS COMPLETE  $(date)"
echo "════════════════════════════════════════════════════════"
