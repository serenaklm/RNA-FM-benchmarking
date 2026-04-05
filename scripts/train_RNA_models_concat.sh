#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Incremental-value experiment: FM + HVG concatenated models.
#
# Usage:
#   ./scripts/train_RNA_models_concat.sh
#
# Runs ALL experiments across 4 GPUs in parallel:
#   GPU 0 -> TCGA       (HVG=10000)
#   GPU 1 -> Hartwig    (HVG=5000)
#   GPU 2 -> CPTAC      (HVG=1250)
#   GPU 3 -> TARGET_AML (HVG=1250)
#
# Within each GPU: loops over
#   2 archs (Linear, ResidualMLP) x
#   2 losses (Cox, CIndex) x
#   6 FMs x 2 seeds x 5 splits  =  120 runs per GPU
#
# Logs written to: logs/concat_{DATASET}.log
# Outputs written to: concat_model_outputs/{DATASET}/survival/PFS/
# ============================================================

ARCHS=(Linear ResidualMLP)
LOSS_NAMES=(Cox CIndex)
SEEDS=(37 42)
SPLITS=(0 1 2 3 4)
FM_KEYS=(BulkFormer BulkRNABert COMPASS scGPT STATE CancerFoundation)

TRAIN_SCRIPT="train.train_RNA_model_concat"
CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
HP_PATH="/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json"

mkdir -p logs

# ── Per-dataset worker function ───────────────────────────────────────────────
run_dataset() {
    local DATASET="$1"
    local GPU_ID="$2"
    local N_HVGS="$3"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    local OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/concat_model_outputs/${DATASET}/survival/PFS"
    local MAX_PARALLEL=8
    local job_count=0

    echo "[GPU $GPU_ID | $DATASET] Starting. N_HVGS=$N_HVGS  MAX_PARALLEL=$MAX_PARALLEL"

    for ARCH in "${ARCHS[@]}"; do
      for LOSS_NAME in "${LOSS_NAMES[@]}"; do
        local MODEL_TAG="${ARCH}_${LOSS_NAME}"
        for FM_KEY in "${FM_KEYS[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            for SPLIT_IDX in "${SPLITS[@]}"; do

              local PREFIX="${MODEL_TAG}_concat_${FM_KEY}_HVG${N_HVGS}_seed_${SEED}_split_${SPLIT_IDX}_"

              if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
                echo "[GPU $GPU_ID | $DATASET] SKIP (already exists): ${PREFIX}*"
                continue
              fi

              echo "[GPU $GPU_ID | $DATASET] Launching: arch=$ARCH loss=$LOSS_NAME fm=$FM_KEY seed=$SEED split=$SPLIT_IDX"

              python -m "$TRAIN_SCRIPT" \
                --config_path      "$CONFIG_PATH" \
                --dataset          "$DATASET" \
                --hp_path          "$HP_PATH" \
                --arch             "$ARCH" \
                --loss_name        "$LOSS_NAME" \
                --seed             "$SEED" \
                --split_idx        "$SPLIT_IDX" \
                --fm_embedding_key "$FM_KEY" \
                --n_HVGs           "$N_HVGS" &

              job_count=$((job_count + 1))

              # Wait for the current batch of 8 before launching more
              if (( job_count % MAX_PARALLEL == 0 )); then
                echo "[GPU $GPU_ID | $DATASET] Waiting for batch of $MAX_PARALLEL to finish..."
                wait
              fi

            done
          done
        done
      done
    done

    # Wait for any remaining jobs in the last (incomplete) batch
    wait
    echo "[GPU $GPU_ID | $DATASET] Done."
}

# ── Launch one worker per GPU in parallel ────────────────────────────────────
echo "Launching all 4 dataset workers..."
echo "Logs: logs/concat_{DATASET}.log"
echo

run_dataset TCGA       0 10000 > logs/concat_TCGA.log       2>&1 &
run_dataset Hartwig    1  5000 > logs/concat_Hartwig.log    2>&1 &
run_dataset CPTAC      2  1250 > logs/concat_CPTAC.log      2>&1 &
run_dataset TARGET_AML 3  1250 > logs/concat_TARGET_AML.log 2>&1 &

wait
echo "All workers finished."
