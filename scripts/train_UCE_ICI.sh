#!/usr/bin/env bash
# =============================================================================
# scripts/train_UCE_ICI.sh
#
# Run ICI response-prediction benchmarking training for UCE only.
# Assumes hyperparameter tuning for UCE has already finished
# (scripts/tune_UCE_ICI.sh).
#
# What this does:
#   1. Re-aggregates hp_results_ICI/all_aggregated_hps_ICI.json
#      (train.agg_best_hps_ICI), picking up UCE's tuned HPs alongside every
#      other FM's.
#   2. Runs train_ICI_models.sh with EXPERIMENT=FM_SC across all 3 paradigms
#      (within_cohort, loco, transfer) x 2 archs (Linear, ResidualMLP). Each
#      call handles both seeds (37, 42) and all cohorts/splits internally
#      (train_ICI_models.sh defaults to both seeds when $SEED is unset).
#      FM_SC also includes scGPT/STATE/CancerFoundation, but
#      train_ICI_models.sh's own skip-check makes those no-ops since they're
#      already done -- only UCE actually trains.
#
# Usage:
#   ./scripts/train_UCE_ICI.sh                # uses GPUs 0 1 2 3 4 5
#   GPUS="0 1 2" ./scripts/train_UCE_ICI.sh    # override GPU list
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/uce_ici"
mkdir -p "$LOG_DIR"

read -ra GPUS <<< "${GPUS:-0 1 2 3 4 5}"

PARADIGMS=(within_cohort loco transfer)
ARCHS=(Linear ResidualMLP)

export HP_PATH="${REPO_ROOT}/hp_results_ICI/all_aggregated_hps_ICI.json"

echo "============================================================"
echo "Step 1/2: Aggregating tuned ICI HPs (train.agg_best_hps_ICI)"
echo "============================================================"
python -m train.agg_best_hps_ICI

echo ""
echo "============================================================"
echo "Step 2/2: ICI training for UCE"
echo "  Paradigms : ${PARADIGMS[*]}"
echo "  Archs     : ${ARCHS[*]}"
echo "  GPUs      : ${GPUS[*]}"
echo "  HP_PATH   : $HP_PATH"
echo "  Log dir   : $LOG_DIR"
echo "============================================================"
echo "Note: transfer runs 20 cohort pairs internally per call and will take"
echo "considerably longer than within_cohort/loco (5 cohorts each)."
echo "============================================================"

i=0
pids=()

for PARADIGM in "${PARADIGMS[@]}"; do
  for ARCH in "${ARCHS[@]}"; do
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    i=$((i + 1))
    LOGFILE="${LOG_DIR}/${PARADIGM}_${ARCH}.log"
    echo "Launching $PARADIGM $ARCH on GPU $GPU (log: $LOGFILE)"
    nohup ./scripts/train_ICI_models.sh "$GPU" "$PARADIGM" FM_SC "$ARCH" \
      > "$LOGFILE" 2>&1 &
    disown
    pids+=($!)
  done
done

echo ""
echo "Launched ${#pids[@]} jobs. Waiting for completion..."
wait "${pids[@]}"

echo ""
echo "============================================================"
echo "All UCE ICI training runs complete. Logs in: $LOG_DIR"
echo "============================================================"
