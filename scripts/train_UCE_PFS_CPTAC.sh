#!/usr/bin/env bash
# =============================================================================
# scripts/train_UCE_PFS_CPTAC.sh
#
# Retrain UCE PFS models for CPTAC only.
#
# Background: the first pass of data_preprocessing/add_UCE_to_RNA_embeddings.py
# mismatched CPTAC's UCE embeddings (its consolidated pkl is keyed by case_id,
# but FM embedding files are keyed by the RNA-seq sample_id -- now fixed via
# the record's own "sample_id" field). TCGA, Hartwig, and TARGET_AML were
# unaffected and don't need to be rerun.
#
# Prerequisite (run once, before this script):
#   python data_preprocessing/add_UCE_to_RNA_embeddings.py --only CPTAC
#
# Grid: 2 archs x 2 losses x 2 seeds = 8 jobs.
# No HP re-aggregation needed -- tuning was TCGA-only and is unaffected.
#
# Usage:
#   ./scripts/train_UCE_PFS_CPTAC.sh              # uses GPUs 0 1 2 3
#   GPUS="4 5" ./scripts/train_UCE_PFS_CPTAC.sh    # override GPU list
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/uce_pfs"
mkdir -p "$LOG_DIR"

read -ra GPUS <<< "${GPUS:-0 1 2 3}"

ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)
SEEDS=(37 42)

echo "============================================================"
echo "UCE PFS retraining: CPTAC only"
echo "  Archs  : ${ARCHS[*]}"
echo "  Losses : ${LOSSES[*]}"
echo "  Seeds  : ${SEEDS[*]}"
echo "  GPUs   : ${GPUS[*]}"
echo "  Log dir: $LOG_DIR"
echo "============================================================"

i=0
pids=()

for ARCH in "${ARCHS[@]}"; do
  for LOSS in "${LOSSES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      GPU=${GPUS[$((i % ${#GPUS[@]}))]}
      i=$((i + 1))
      LOGFILE="${LOG_DIR}/CPTAC_${ARCH}_${LOSS}_seed${SEED}.log"
      echo "Launching CPTAC $ARCH $LOSS seed=$SEED on GPU $GPU (log: $LOGFILE)"
      SEED="$SEED" nohup ./scripts/train_RNA_models.sh "$GPU" CPTAC survival FM_SC "$ARCH" "$LOSS" \
        > "$LOGFILE" 2>&1 &
      disown
      pids+=($!)
    done
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "CPTAC UCE PFS training complete. Logs in: $LOG_DIR"
echo "============================================================"
