#!/usr/bin/env bash
# =============================================================================
# scripts/tune_UCE_ICI.sh
#
# Hyperparameter tuning for ICI response prediction, UCE only.
# Mirrors scripts/tune_ICI_models.sh (tunes on IMVigor210 within-cohort
# splits), scoped to just the UCE embedding instead of all 11.
#
# Grid: 2 archs x 5 splits = 10 jobs.
#
# Usage:
#   ./scripts/tune_UCE_ICI.sh              # N_TRIALS=50 (default)
#   N_TRIALS=30 ./scripts/tune_UCE_ICI.sh  # override trial count
#   NO_WANDB=1  ./scripts/tune_UCE_ICI.sh  # disable WandB logging
#   GPUS="0 1" ./scripts/tune_UCE_ICI.sh   # override GPU list (default: 0 1 2 3)
#
# After this finishes, aggregate with:
#   python -m train.agg_best_hps_ICI
# (train_UCE_ICI.sh does this automatically before training.)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/tune_ICI"
mkdir -p "$LOG_DIR"

N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-37}"
read -ra GPUS <<< "${GPUS:-0 1 2 3}"

WANDB_FLAG=""
if [[ -n "${NO_WANDB:-}" ]]; then
  WANDB_FLAG="--no_wandb"
fi

ARCHS=("Linear" "ResidualMLP")
SPLITS=(0 1 2 3 4)

echo "============================================================"
echo "UCE ICI HP Tuning"
echo "  N_TRIALS : $N_TRIALS"
echo "  GPUs     : ${GPUS[*]}"
echo "  Log dir  : $LOG_DIR"
echo "============================================================"

i=0
pids=()

for ARCH in "${ARCHS[@]}"; do
  for SPLIT in "${SPLITS[@]}"; do
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i + 1))
    LOG="${LOG_DIR}/UCE_${ARCH}_split${SPLIT}.log"

    echo "Launching UCE $ARCH split=$SPLIT on GPU $GPU (log: $LOG)"
    CUDA_VISIBLE_DEVICES="$GPU" nohup python -m train.tune_ICI \
      --rna_embedding_key UCE \
      --n_HVGs            0 \
      --arch              "$ARCH" \
      --split_idx         "$SPLIT" \
      --n_trials          "$N_TRIALS" \
      --seed              "$SEED" \
      ${WANDB_FLAG} \
      > "$LOG" 2>&1 &
    pids+=($!)
    disown
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "UCE ICI tuning complete."
echo "Next: python -m train.agg_best_hps_ICI"
echo "Then: ./scripts/train_UCE_ICI.sh"
echo "============================================================"
