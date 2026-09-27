#!/usr/bin/env bash
# =============================================================================
# scripts/train_UCE_PFS.sh
#
# Run PFS (survival) benchmarking training for UCE only.
# Assumes hyperparameter tuning for UCE has already finished (see
# scripts/tune_RNA_models.sh UCE <ARCH> <LOSS>, run for all 4 arch/loss combos).
#
# What this does:
#   1. Re-aggregates hp_results/all_aggregated_hps.json (train.agg_best_hps),
#      picking up UCE's tuned HPs alongside every other FM's.
#   2. Runs train_RNA_models.sh with EXPERIMENT=FM_SC across all 4 PFS
#      datasets x 2 archs x 2 losses x 2 seeds (the same grid scGPT/STATE/
#      CancerFoundation were run on). FM_SC also includes scGPT/STATE/
#      CancerFoundation, but train_RNA_models.sh's own skip-check makes those
#      no-ops since they're already done -- only UCE actually trains.
#
# Usage:
#   ./scripts/train_UCE_PFS.sh                # uses GPUs 0 1 2 3
#   GPUS="0 1" ./scripts/train_UCE_PFS.sh      # override GPU list
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/uce_pfs"
mkdir -p "$LOG_DIR"

read -ra GPUS <<< "${GPUS:-0 1 2 3}"

DATASETS=(TCGA CPTAC Hartwig TARGET_AML)
ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)
SEEDS=(37 42)

echo "============================================================"
echo "Step 1/2: Aggregating tuned HPs (train.agg_best_hps)"
echo "============================================================"
python -m train.agg_best_hps

echo ""
echo "============================================================"
echo "Step 2/2: PFS training for UCE"
echo "  Datasets : ${DATASETS[*]}"
echo "  Archs    : ${ARCHS[*]}"
echo "  Losses   : ${LOSSES[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  GPUs     : ${GPUS[*]}"
echo "  Log dir  : $LOG_DIR"
echo "============================================================"

i=0
pids=()

for DATASET in "${DATASETS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    for LOSS in "${LOSSES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        GPU=${GPUS[$((i % ${#GPUS[@]}))]}
        i=$((i + 1))
        LOGFILE="${LOG_DIR}/${DATASET}_${ARCH}_${LOSS}_seed${SEED}.log"
        echo "Launching $DATASET $ARCH $LOSS seed=$SEED on GPU $GPU (log: $LOGFILE)"
        SEED="$SEED" nohup ./scripts/train_RNA_models.sh "$GPU" "$DATASET" survival FM_SC "$ARCH" "$LOSS" \
          > "$LOGFILE" 2>&1 &
        disown
        pids+=($!)
      done
    done
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "All UCE PFS training runs complete. Logs in: $LOG_DIR"
echo "============================================================"
