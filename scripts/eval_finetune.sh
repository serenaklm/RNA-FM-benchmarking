#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Evaluate fine-tuned FM survival models for a given FM + dataset.
# Each FM requires its own conda environment.
#
# Usage:
#   ./scripts/eval_finetune.sh <FM_NAME> <DATASET>
#
# Examples:
#   ./scripts/eval_finetune.sh BulkFormer TCGA
#   ./scripts/eval_finetune.sh CancerFoundation TCGA
#   ./scripts/eval_finetune.sh scGPT TCGA
#   ./scripts/eval_finetune.sh BulkRNABert TCGA
#   ./scripts/eval_finetune.sh COMPASS TCGA
#   ./scripts/eval_finetune.sh STATE TCGA
# ============================================================

FM_NAME="${1:?Usage: $0 <FM_NAME> <DATASET>}"
DATASET="${2:?Usage: $0 <FM_NAME> <DATASET>}"
ENDPOINT="${ENDPOINT:-PFS}"
GPU="${GPU:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SCRIPT="test.eval_finetune"
BASE_DIR="${REPO_ROOT}"

# ---- select python binary for this FM ----
# Override PYTHON env var if you need a specific interpreter (e.g. PYTHON=/path/to/env/bin/python)
PYTHON="${PYTHON:-python}"
case "$FM_NAME" in
  BulkFormer|CancerFoundation|scGPT|BulkRNABert|COMPASS|STATE)
    ;;
  *)
    echo "Unknown FM_NAME: $FM_NAME"
    echo "Supported: BulkFormer, CancerFoundation, scGPT, BulkRNABert, COMPASS, STATE"
    exit 1
    ;;
esac

LOG_DIR="${BASE_DIR}/logs/eval_finetune/${DATASET}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${FM_NAME}.log"

echo "============================================================"
echo "FM_NAME=$FM_NAME"
echo "DATASET=$DATASET"
echo "ENDPOINT=$ENDPOINT"
echo "GPU=$GPU"
echo "PYTHON=$PYTHON"
echo "Logging to: $LOG_FILE"
echo "============================================================"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m "$EVAL_SCRIPT" \
  --dataset "$DATASET" \
  --endpoint "$ENDPOINT" \
  --fm_name "$FM_NAME" \
  --verbose \
  2>&1 | tee "$LOG_FILE"
