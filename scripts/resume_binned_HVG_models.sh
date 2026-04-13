#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Resume binned-HVG ablation — targeted restart for remaining runs.
#
# GPU layout (optimised for remaining work after partial training):
#   GPU 0 -> TCGA       strategies A (linear, global_linear, per_sample_linear)
#   GPU 1 -> TCGA       strategies A continued (global_linear hi-bins + per_sample_linear)
#   GPU 2 -> TCGA       strategies B (quantile, global_quantile, per_sample_quantile)
#   GPU 3 -> TCGA       strategies B continued (global_quantile hi-bins + per_sample_quantile)
#   GPU 4 -> TARGET_AML strategies A
#   GPU 5 -> TARGET_AML strategies B
#   GPU 6 -> TARGET_AML strategies B continued + Hartwig all strategies
#   GPU 7 -> CPTAC all strategies
#
# Skip logic: any run with an existing output folder is skipped automatically.
# ============================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="logs/hvg_binning"
mkdir -p "$LOG_DIR"

MAX_PARALLEL="${MAX_PARALLEL:-8}"
CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
HP_PATH="${HP_PATH:-${REPO_ROOT}/hp_results/all_aggregated_hps.json}"
TRAIN_SCRIPT="train.train_binned_HVG_model"

ARCH="ResidualMLP"
LOSS="CIndex"
SEED=37

N_BINS_LIST=(2 8 16 32 51 64 128 256 512 1024)
SPLITS=(0 1 2 3 4)

declare -A BEST_N_HVGS
BEST_N_HVGS[TCGA]=10000
BEST_N_HVGS[Hartwig]=5000
BEST_N_HVGS[CPTAC]=1250
BEST_N_HVGS[TARGET_AML]=1250

# ============================================================
# run_dataset <GPU_ID> <DATASET> <STRATEGY...>
# ============================================================
run_dataset() {
    local GPU_ID="$1"
    local DATASET="$2"
    shift 2
    local STRATEGIES=("$@")
    local N_HVGS="${BEST_N_HVGS[$DATASET]}"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    local OUT_BASE="${REPO_ROOT}/model_outputs/${DATASET}/survival/PFS/binned_HVG_ablation"
    local running=0

    echo "============================================================"
    echo "GPU $GPU_ID | Dataset: $DATASET | n_HVGs: $N_HVGS"
    echo "Strategies: ${STRATEGIES[*]}"
    echo "MAX_PARALLEL: $MAX_PARALLEL"
    echo "============================================================"

    for STRATEGY in "${STRATEGIES[@]}"; do
      for N_BINS in "${N_BINS_LIST[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do

          EMBEDDING_TAG="binned_${STRATEGY}_${N_BINS}_nHVGs_${N_HVGS}"
          PREFIX="${ARCH}_${LOSS}_rnaKey_${EMBEDDING_TAG}_seed_${SEED}_split_${SPLIT}_"

          if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
              continue  # already done
          fi

          LOG_FILE="${LOG_DIR}/${DATASET}_${ARCH}_${LOSS}_${EMBEDDING_TAG}_seed_${SEED}_split_${SPLIT}.log"

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

        done
      done
    done

    wait
    echo "Done: $DATASET (GPU $GPU_ID, strategies: ${STRATEGIES[*]})"
}

# ── TCGA: 4 GPUs (most remaining work, largest dataset) ──────────────────────
run_dataset 0 TCGA linear global_linear                         &
run_dataset 1 TCGA per_sample_linear                            &
run_dataset 2 TCGA quantile global_quantile                     &
run_dataset 3 TCGA per_sample_quantile                          &

# ── TARGET_AML: 2 GPUs ───────────────────────────────────────────────────────
run_dataset 4 TARGET_AML linear global_linear per_sample_linear &
run_dataset 5 TARGET_AML quantile global_quantile per_sample_quantile &

# ── Hartwig: 1 GPU (mostly per_sample remaining, small dataset) ──────────────
run_dataset 6 Hartwig linear quantile global_linear global_quantile per_sample_linear per_sample_quantile &

# ── CPTAC: 1 GPU (mostly per_sample remaining, small dataset) ────────────────
run_dataset 7 CPTAC linear quantile global_linear global_quantile per_sample_linear per_sample_quantile &

wait
echo "All remaining runs completed."
