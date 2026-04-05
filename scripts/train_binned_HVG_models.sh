#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Binned-HVG ablation experiment
#
# Fixed: ResidualMLP, CIndex loss, seed 37.
# Sweep:
#   n_bins     : 2  8  16  32  51  64  128  256  512  1024
#   strategy   : linear  quantile                  (per-gene)
#                global_linear  global_quantile     (global; BulkRNABert=global_linear@64)
#                per_sample_linear  per_sample_quantile  (per-sample; scGPT=per_sample_quantile@51,
#                                                         CancerFoundation=per_sample_quantile@100)
#   splits     : 0  1  2  3  4
#
# GPU layout (8 GPUs, 2 per dataset, strategies split 3+3):
#   GPU 0 -> TCGA        strategies A
#   GPU 1 -> Hartwig     strategies A
#   GPU 2 -> CPTAC       strategies A
#   GPU 3 -> TARGET_AML  strategies A
#   GPU 4 -> TCGA        strategies B
#   GPU 5 -> Hartwig     strategies B
#   GPU 6 -> CPTAC       strategies B
#   GPU 7 -> TARGET_AML  strategies B
#
#   Strategies A: linear  global_linear  per_sample_linear
#   Strategies B: quantile  global_quantile  per_sample_quantile
#
# Concurrency per GPU (MAX_PARALLEL): 8
#
# Usage:
#   # Launch all datasets across GPUs 0–7
#   ./scripts/train_binned_HVG_models.sh
#
#   # Launch a single GPU/dataset/strategy-subset manually
#   ./scripts/train_binned_HVG_models.sh <GPU_ID> <DATASET> <STRATEGY...>
#   e.g. ./scripts/train_binned_HVG_models.sh 0 TCGA linear global_linear per_sample_linear
# ============================================================

LOG_DIR="logs/hvg_binning"
mkdir -p "$LOG_DIR"

MAX_PARALLEL="${MAX_PARALLEL:-8}"
CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
HP_PATH="${HP_PATH:-/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json}"
TRAIN_SCRIPT="train.train_binned_HVG_model"

# Fixed choices
ARCH="ResidualMLP"
LOSS="CIndex"
SEED=37

N_BINS_LIST=(2 8 16 32 51 64 128 256 512 1024)
SPLITS=(0 1 2 3 4)

# Strategy subsets for the two GPUs per dataset
STRATEGIES_A=(linear global_linear per_sample_linear)
STRATEGIES_B=(quantile global_quantile per_sample_quantile)

# Best-performing n_HVGs per dataset from the main benchmarking
declare -A BEST_N_HVGS
BEST_N_HVGS[TCGA]=10000
BEST_N_HVGS[Hartwig]=5000
BEST_N_HVGS[CPTAC]=1250
BEST_N_HVGS[TARGET_AML]=1250

# ============================================================
# run_dataset <GPU_ID> <DATASET> <STRATEGY...>
#   Runs all (n_bins x splits) jobs for one dataset/GPU/strategy
#   subset, keeping at most MAX_PARALLEL models simultaneously.
# ============================================================
run_dataset() {
    local GPU_ID="$1"
    local DATASET="$2"
    shift 2
    local STRATEGIES=("$@")
    local N_HVGS="${BEST_N_HVGS[$DATASET]}"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    local OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs/${DATASET}/survival/PFS/binned_HVG_ablation"
    local running=0

    echo "============================================================"
    echo "GPU $GPU_ID | Dataset: $DATASET | Seed: $SEED | n_HVGs: $N_HVGS"
    echo "Strategies: ${STRATEGIES[*]}"
    echo "OUT_BASE: $OUT_BASE"
    echo "MAX_PARALLEL: $MAX_PARALLEL"
    echo "============================================================"

    for STRATEGY in "${STRATEGIES[@]}"; do
      for N_BINS in "${N_BINS_LIST[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do

          EMBEDDING_TAG="binned_${STRATEGY}_${N_BINS}_nHVGs_${N_HVGS}"
          PREFIX="${ARCH}_${LOSS}_rnaKey_${EMBEDDING_TAG}_seed_${SEED}_split_${SPLIT}_"

          # Skip if a matching output directory already exists
          if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
              echo "SKIP (exists): ${PREFIX}*"
              continue
          fi

          LOG_FILE="${LOG_DIR}/${DATASET}_${ARCH}_${LOSS}_${EMBEDDING_TAG}_seed_${SEED}_split_${SPLIT}.log"
          echo "Running: $DATASET | strategy=$STRATEGY bins=$N_BINS | seed=$SEED split=$SPLIT (GPU=$GPU_ID)"
          echo "  → log: $LOG_FILE"

          python -m "$TRAIN_SCRIPT" \
              --config_path  "$CONFIG_PATH" \
              --dataset      "$DATASET" \
              --n_HVGs       "$N_HVGS" \
              --n_bins       "$N_BINS" \
              --bin_strategy "$STRATEGY" \
              --arch         "$ARCH" \
              --loss_name    "$LOSS" \
              --seed         "$SEED" \
              --split_idx    "$SPLIT" \
              --hp_path      "$HP_PATH" \
              > "$LOG_FILE" 2>&1 &

          running=$((running + 1))

          if [[ $running -ge $MAX_PARALLEL ]]; then
              wait -n 2>/dev/null || true
              running=$((running - 1))
          fi

        done   # SPLIT
      done     # N_BINS
    done       # STRATEGY

    # Drain remaining background jobs
    wait
    echo "All jobs completed for $DATASET (GPU $GPU_ID, strategies: ${STRATEGIES[*]})"
}

# ============================================================
# Entry point
# ============================================================
if [[ $# -ge 3 ]]; then
    # Manual mode: ./script.sh <GPU_ID> <DATASET> <STRATEGY...>
    GPU_ID="$1"; DATASET="$2"; shift 2
    run_dataset "$GPU_ID" "$DATASET" "$@"
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [<GPU_ID> <DATASET> <STRATEGY...>]"
    echo "  No args         : launch all datasets across GPUs 0–7"
    echo "  GPU/dataset/strat: run one GPU manually"
    exit 1
fi

# Strategies A (linear variants) on GPUs 0–3
run_dataset 0 TCGA       "${STRATEGIES_A[@]}" &
run_dataset 1 Hartwig    "${STRATEGIES_A[@]}" &
run_dataset 2 CPTAC      "${STRATEGIES_A[@]}" &
run_dataset 3 TARGET_AML "${STRATEGIES_A[@]}" &

# Strategies B (quantile variants) on GPUs 4–7
run_dataset 4 TCGA       "${STRATEGIES_B[@]}" &
run_dataset 5 Hartwig    "${STRATEGIES_B[@]}" &
run_dataset 6 CPTAC      "${STRATEGIES_B[@]}" &
run_dataset 7 TARGET_AML "${STRATEGIES_B[@]}" &

wait
echo "All datasets completed."