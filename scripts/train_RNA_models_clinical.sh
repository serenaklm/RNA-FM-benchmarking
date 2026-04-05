#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Clinical + FM concatenation experiment.
#
# Trains [FM_embedding | clinical_features] models using the
# same Linear / ResidualMLP + Cox / CIndex setup as the main
# benchmarking, across all 4 datasets and 7 embedding keys.
#
# GPU assignment (8 GPUs, split by dataset × arch):
#   GPU 0 -> TCGA       + Linear
#   GPU 1 -> TCGA       + ResidualMLP
#   GPU 2 -> Hartwig    + Linear
#   GPU 3 -> Hartwig    + ResidualMLP
#   GPU 4 -> CPTAC      + Linear
#   GPU 5 -> CPTAC      + ResidualMLP
#   GPU 6 -> TARGET_AML + Linear
#   GPU 7 -> TARGET_AML + ResidualMLP
#
# Per GPU: 2 losses × 7 FM keys × 2 seeds × 5 splits = 140 runs
# Up to MAX_PARALLEL=8 jobs run simultaneously per GPU.
#
# Logs  → logs/clinical_concat_{DATASET}_{ARCH}.log
# Outputs → clinical_concat_model_outputs/{DATASET}/survival/PFS/
# ============================================================

LOSS_NAMES=(Cox CIndex)
SEEDS=(37 42)
SPLITS=(0 1 2 3 4)
FM_KEYS=(BulkFormer BulkRNABert COMPASS scGPT STATE CancerFoundation RNA_seq)

TRAIN_SCRIPT="train.train_RNA_model_clinical"
CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
HP_PATH="/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json"
CLINICAL_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking"

mkdir -p logs

# ── Per-GPU worker ────────────────────────────────────────────────────────────
run_worker() {
    local DATASET="$1"
    local GPU_ID="$2"
    local ARCH="$3"

    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    local OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/clinical_concat_model_outputs/${DATASET}/survival/PFS"
    local MAX_PARALLEL=8
    local job_count=0

    echo "[GPU $GPU_ID | $DATASET | $ARCH] Starting."

    for LOSS_NAME in "${LOSS_NAMES[@]}"; do
      local MODEL_TAG="${ARCH}_${LOSS_NAME}"
      for FM_KEY in "${FM_KEYS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
          for SPLIT_IDX in "${SPLITS[@]}"; do

            local PREFIX="${MODEL_TAG}_clinical_concat_${FM_KEY}_seed_${SEED}_split_${SPLIT_IDX}_"

            if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
              echo "[GPU $GPU_ID | $DATASET | $ARCH] SKIP (exists): ${PREFIX}*"
              continue
            fi

            echo "[GPU $GPU_ID | $DATASET | $ARCH] Launching: loss=$LOSS_NAME fm=$FM_KEY seed=$SEED split=$SPLIT_IDX"

            python -m "$TRAIN_SCRIPT" \
              --config_path       "$CONFIG_PATH" \
              --dataset           "$DATASET" \
              --hp_path           "$HP_PATH" \
              --arch              "$ARCH" \
              --loss_name         "$LOSS_NAME" \
              --seed              "$SEED" \
              --split_idx         "$SPLIT_IDX" \
              --rna_embedding_key "$FM_KEY" \
              --clinical_base     "$CLINICAL_BASE" &

            job_count=$((job_count + 1))

            if (( job_count % MAX_PARALLEL == 0 )); then
              echo "[GPU $GPU_ID | $DATASET | $ARCH] Waiting for batch of $MAX_PARALLEL..."
              wait
            fi

          done
        done
      done
    done

    wait
    echo "[GPU $GPU_ID | $DATASET | $ARCH] Done."
}

# ── Launch 8 workers (one per GPU) ───────────────────────────────────────────
echo "Launching 8 workers (dataset × arch)..."
echo "Logs: logs/clinical_concat_{DATASET}_{ARCH}.log"
echo

run_worker TCGA       0 Linear       > logs/clinical_concat_TCGA_Linear.log       2>&1 &
run_worker TCGA       1 ResidualMLP  > logs/clinical_concat_TCGA_ResidualMLP.log  2>&1 &
run_worker Hartwig    2 Linear       > logs/clinical_concat_Hartwig_Linear.log     2>&1 &
run_worker Hartwig    3 ResidualMLP  > logs/clinical_concat_Hartwig_ResidualMLP.log 2>&1 &
run_worker CPTAC      4 Linear       > logs/clinical_concat_CPTAC_Linear.log       2>&1 &
run_worker CPTAC      5 ResidualMLP  > logs/clinical_concat_CPTAC_ResidualMLP.log  2>&1 &
run_worker TARGET_AML 6 Linear       > logs/clinical_concat_TARGET_AML_Linear.log  2>&1 &
run_worker TARGET_AML 7 ResidualMLP  > logs/clinical_concat_TARGET_AML_ResidualMLP.log 2>&1 &

wait
echo "All workers finished."
