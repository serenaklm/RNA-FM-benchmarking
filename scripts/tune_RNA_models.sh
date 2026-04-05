#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/tune_RNA_models.sh <RNA_EMBEDDING_KEY> <ARCH> <LOSS_NAME>
# Example:
#   ./scripts/tune_RNA_models.sh BulkFormer ResidualMLP Cox

KEY="$1"
ARCH="$2"
LOSS_NAME="$3"

SEED="${SEED:-37}"
N_TRIALS="${N_TRIALS:-50}"
CONFIG_PATH="${CONFIG_PATH:-./configs/TCGA_prognostic_all_models.yaml}"
GPUS=(0 1)
N_HVG="${N_HVG:-17490}"

mkdir -p ./hp_results

PIDS=()

for SPLIT_IDX in 0 1 2 3 4; do
  OUT_PATH="./hp_results/${KEY}_${ARCH}_${LOSS_NAME}_split${SPLIT_IDX}.json"

  if [[ -f "$OUT_PATH" ]]; then
    echo "SKIP (exists): $OUT_PATH"
    continue
  fi

  GPU=${GPUS[$((SPLIT_IDX % ${#GPUS[@]}))]}
  echo "Launching: key=$KEY split=$SPLIT_IDX arch=$ARCH loss=$LOSS_NAME on GPU=$GPU"

  CUDA_VISIBLE_DEVICES=$GPU python -m train.tune \
    --config_path "$CONFIG_PATH" \
    --rna_embedding_key "$KEY" \
    --arch "$ARCH" \
    --loss_name "$LOSS_NAME" \
    --n_HVGs "$N_HVG" \
    --n_trials "$N_TRIALS" \
    --split_idx "$SPLIT_IDX" \
    --seed "$SEED" \
    > "./hp_results/log_${KEY}_${ARCH}_${LOSS_NAME}_split${SPLIT_IDX}.txt" 2>&1 &

  PIDS+=($!)
done

echo "Waiting for ${#PIDS[@]} jobs..."
wait
echo "Done. Now run aggregate_hps.py to combine split results."