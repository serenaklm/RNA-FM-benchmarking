#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Train CancerFoundation LoRA fine-tuning across all splits.
# Runs one split per GPU.
#
# Usage:
#   ./scripts/train_finetune_CancerFoundation.sh <DATASET>
#
# Example:
#   ./scripts/train_finetune_CancerFoundation.sh TCGA
# ============================================================

DATASET="${1:?Usage: $0 <DATASET>}"

FM_NAME="CancerFoundation"
LOSS_NAME="${LOSS_NAME:-CIndex}"
SEED="${SEED:-37}"
SPLITS=(0 1 2 3 4)
GPUS=(1 3 4 5 6)

CONFIG_PATH="${CONFIG_PATH:-./configs/finetune.yaml}"
TRAIN_SCRIPT="train.train_finetune"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

OUT_BASE="${REPO_ROOT}/model_outputs_finetune/${DATASET}/survival/PFS"

LOG_DIR="${REPO_ROOT}/logs/finetune/${DATASET}/${FM_NAME}"
mkdir -p "$LOG_DIR"

case "$DATASET" in
  TCGA|CPTAC|CALGB|Hartwig|ICGC|TARGET_AML) ;;
  *)
    echo "Unknown DATASET: $DATASET. Must be one of TCGA|CPTAC|CALGB|Hartwig|ICGC|TARGET_AML"
    exit 1
    ;;
esac

echo "============================================================"
echo "DATASET=$DATASET"
echo "FM_NAME=$FM_NAME"
echo "LOSS_NAME=$LOSS_NAME"
echo "SEED=$SEED"
echo "SPLITS=${SPLITS[*]}"
echo "GPUS=${GPUS[*]}"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON=$PYTHON"
echo "============================================================"

for i in "${!SPLITS[@]}"; do
  SPLIT_IDX="${SPLITS[$i]}"
  GPU_ID="${GPUS[$i]}"

  PREFIX="finetune_${FM_NAME}_${LOSS_NAME}_LoRA_seed_${SEED}_split_${SPLIT_IDX}_"

  if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
    echo "SKIP (already exists): ${PREFIX}*"
    continue
  fi

  echo
  echo "Running: dataset=$DATASET fm=$FM_NAME loss=$LOSS_NAME split=$SPLIT_IDX seed=$SEED (GPU=$GPU_ID)"

  LOG_FILE="${LOG_DIR}/split_${SPLIT_IDX}_seed_${SEED}.log"
  echo "Logging to: $LOG_FILE"

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m "$TRAIN_SCRIPT" \
    --config_path "$CONFIG_PATH" \
    --dataset "$DATASET" \
    --fm_name "$FM_NAME" \
    --loss_name "$LOSS_NAME" \
    --seed "$SEED" \
    --split_idx "$SPLIT_IDX" \
    > "$LOG_FILE" 2>&1 &
done

wait

echo
echo "Done: all splits for $FM_NAME on $DATASET."
