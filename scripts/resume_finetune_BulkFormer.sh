#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Resume crashed BulkFormer LoRA fine-tuning runs.
#
# For each split, finds the latest .ckpt in the existing output
# directory and resumes training from it.
#
# Usage:
#   ./scripts/resume_finetune_BulkFormer.sh <DATASET>
#
# Example:
#   ./scripts/resume_finetune_BulkFormer.sh TCGA
# ============================================================

DATASET="${1:?Usage: $0 <DATASET>}"

FM_NAME="BulkFormer"
LOSS_NAME="${LOSS_NAME:-CIndex}"
SEED="${SEED:-37}"
SPLITS=(0 1 2 3 4)
GPUS=(0 1 2 3 4)

CONFIG_PATH="${CONFIG_PATH:-./configs/finetune.yaml}"
TRAIN_SCRIPT="train.train_finetune"

PYTHON="/data/rbg/users/ashyamal/miniconda3/envs/bulkformer/bin/python"

OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs_finetune/${DATASET}/survival/PFS"

LOG_DIR="/data/rbg/users/ashyamal/RNA-FM-benchmarking/logs/finetune/${DATASET}/${FM_NAME}"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "DATASET=$DATASET  FM=$FM_NAME  LOSS=$LOSS_NAME  SEED=$SEED"
echo "============================================================"

for i in "${!SPLITS[@]}"; do
  SPLIT_IDX="${SPLITS[$i]}"
  GPU_ID="${GPUS[$i]}"

  # Find the existing output directory for this split
  PREFIX="finetune_${FM_NAME}_${LOSS_NAME}_LoRA_seed_${SEED}_split_${SPLIT_IDX}_"
  EXISTING_DIR=$(compgen -G "${OUT_BASE}/${PREFIX}*" 2>/dev/null | head -1 || true)

  if [[ -z "$EXISTING_DIR" ]]; then
    echo "SKIP split $SPLIT_IDX: no existing dir found (run train_finetune_BulkFormer.sh first)"
    continue
  fi

  # Find the checkpoint inside that directory
  CKPT=$(ls "${EXISTING_DIR}"/*.ckpt 2>/dev/null | head -1 || true)

  if [[ -z "$CKPT" ]]; then
    echo "SKIP split $SPLIT_IDX: no .ckpt found in $EXISTING_DIR"
    continue
  fi

  echo
  echo "Resuming: split=$SPLIT_IDX  GPU=$GPU_ID"
  echo "  dir:  $EXISTING_DIR"
  echo "  ckpt: $(basename "$CKPT")"

  LOG_FILE="${LOG_DIR}/split_${SPLIT_IDX}_seed_${SEED}_resume.log"
  echo "  log:  $LOG_FILE"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m "$TRAIN_SCRIPT" \
    --config_path "$CONFIG_PATH" \
    --dataset "$DATASET" \
    --fm_name "$FM_NAME" \
    --loss_name "$LOSS_NAME" \
    --seed "$SEED" \
    --split_idx "$SPLIT_IDX" \
    --ckpt_path "$CKPT" \
    > "$LOG_FILE" 2>&1 &
done

wait

echo
echo "Done: all splits resumed for $FM_NAME on $DATASET."
