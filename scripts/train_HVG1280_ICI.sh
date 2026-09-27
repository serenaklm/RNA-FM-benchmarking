#!/usr/bin/env bash
# =============================================================================
# scripts/train_HVG1280_ICI.sh
#
# Trains the RNA_seq HVG=1280 baseline for ICI response prediction (matches
# UCE's 1280-dim embedding).
#
# N_HVGS_LIST_4 in train_ICI_models.sh now includes 1280 alongside the
# existing (44, 256, 512, 640, 2058) values. EXPERIMENT=HVG4 re-checks all 6
# sizes per (paradigm, arch), but train_ICI_models.sh's own skip-check means
# the 5 already-trained sizes are no-ops; only 1280 actually trains.
#
# No HP re-aggregation needed: RNA_seq HVG baselines of any size reuse the
# "RNA_seq_1250" tuned HPs (see train_ICI_model.py's hp_key logic).
#
# Grid: 3 paradigms x 2 archs = 6 top-level jobs (each internally loops all
# 6 HVG sizes x both seeds x cohorts/splits; only the 1280 slice actually runs).
#
# Usage:
#   ./scripts/train_HVG1280_ICI.sh              # uses GPUs 0 1 2 3 4 5
#   GPUS="2 3 4 5" ./scripts/train_HVG1280_ICI.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/hvg1280_ici"
mkdir -p "$LOG_DIR"

read -ra GPUS <<< "${GPUS:-0 1 2 3 4 5}"

PARADIGMS=(within_cohort loco transfer)
ARCHS=(Linear ResidualMLP)

export HP_PATH="${REPO_ROOT}/hp_results_ICI/all_aggregated_hps_ICI.json"

echo "============================================================"
echo "ICI training: RNA_seq HVG=1280 baseline (matches UCE dim)"
echo "  Paradigms : ${PARADIGMS[*]}"
echo "  Archs     : ${ARCHS[*]}"
echo "  GPUs      : ${GPUS[*]}"
echo "  HP_PATH   : $HP_PATH"
echo "  Log dir   : $LOG_DIR"
echo "============================================================"

i=0
pids=()

for PARADIGM in "${PARADIGMS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i + 1))
    LOGFILE="${LOG_DIR}/${PARADIGM}_${ARCH}.log"
    echo "Launching $PARADIGM $ARCH on GPU $GPU (log: $LOGFILE)"
    nohup ./scripts/train_ICI_models.sh "$GPU" "$PARADIGM" HVG4 "$ARCH" \
      > "$LOGFILE" 2>&1 &
    pids+=($!)
    disown
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "HVG=1280 ICI training complete. Logs in: $LOG_DIR"
echo "============================================================"
