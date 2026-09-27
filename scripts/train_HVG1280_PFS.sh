#!/usr/bin/env bash
# =============================================================================
# scripts/train_HVG1280_PFS.sh
#
# Trains the RNA_seq HVG=1280 baseline for PFS survival (matches UCE's
# 1280-dim embedding, for the dimension-matched FM-vs-HVG comparison).
#
# N_HVGS_LIST_4 in train_RNA_models.sh now includes 1280 alongside the
# existing (44, 256, 512, 640, 2058) values used for the other FMs. Since
# EXPERIMENT=HVG4 loops over the whole list, this will re-check all 6 values
# per (dataset, arch, loss, seed) -- but train_RNA_models.sh's own skip-check
# means the 5 already-trained sizes are no-ops; only 1280 actually trains.
#
# No HP re-aggregation needed: RNA_seq HVG baselines of any size all reuse
# the "RNA_seq_1250" tuned HPs (see get_effective_key in train_RNA_model.py).
#
# Grid: 4 datasets x 2 archs x 2 losses x 2 seeds = 32 top-level jobs
# (each internally loops all 6 HVG sizes x 5 splits; only 1280 x 5 = 5 of
# those actually run).
#
# Usage:
#   ./scripts/train_HVG1280_PFS.sh              # uses GPUs 0 1 2 3
#   GPUS="4 5 6 7" ./scripts/train_HVG1280_PFS.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/hvg1280_pfs"
mkdir -p "$LOG_DIR"

read -ra GPUS <<< "${GPUS:-0 1 2 3}"

DATASETS=(TCGA CPTAC Hartwig TARGET_AML)
ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)
SEEDS=(37 42)

echo "============================================================"
echo "PFS training: RNA_seq HVG=1280 baseline (matches UCE dim)"
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
        SEED="$SEED" nohup ./scripts/train_RNA_models.sh "$GPU" "$DATASET" survival HVG4 "$ARCH" "$LOSS" \
          > "$LOGFILE" 2>&1 &
        pids+=($!)
        disown
      done
    done
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "HVG=1280 PFS training complete. Logs in: $LOG_DIR"
echo "============================================================"
