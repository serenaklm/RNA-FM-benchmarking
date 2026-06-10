#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# scripts/run_within_4gpu.sh
#
# Within-cohort sweep across 4 GPUs, 2 experiments per GPU.
# Parallelism: 2 exps × 2 archs × 2 seeds = 8 concurrent jobs per GPU.
#
# GPU allocation (2 experiments per GPU):
#   GPU 4: HVG1   + HVG2
#   GPU 5: HVG3   + BASELINE
#   GPU 6: GS     + FM_BULK
#   GPU 7: FM_SC  + PCA
#
# Peak concurrency: 4 GPUs × 8 jobs = 32 concurrent training runs
# Total models: 2 cohorts × 25 emb × 5 splits × 2 seeds × 2 archs = 1,000
# (IMVigor210 + IMMotion150 only — Hugo/Riaz/Rose too small for within-cohort CV)
#
# Usage:
#   ./scripts/run_within_4gpu.sh
#
# Override config/HP paths:
#   CONFIG_PATH=... HP_PATH=... ./scripts/run_within_4gpu.sh
# =============================================================================

LOGDIR="./logs/ICI_within_cohort"
mkdir -p "$LOGDIR"

TRAIN_ICI="./scripts/train_ICI_models.sh"

export CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
export HP_PATH="${HP_PATH:-./hp_results_ICI/all_aggregated_hps_ICI.json}"
export COHORTS_FILTER="IMVigor210 IMMotion150"

ARCHS=(Linear ResidualMLP)
SEEDS=(37 42)

# ---------------------------------------------------------------------------
# run_gpu <GPU> <EXP1> [EXP2 ...]
#   Launches all (exp × arch × seed) combos in parallel on a single GPU.
# ---------------------------------------------------------------------------
run_gpu () {
  local GPU="$1"
  shift
  local EXPS=("$@")
  local PIDS=()

  echo "[GPU${GPU}] Launching within_cohort ${EXPS[*]} — all exp+arch+seed combos"
  for EXP in "${EXPS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        local LOGFILE="${LOGDIR}/gpu${GPU}_within_${EXP}_${ARCH}_seed${SEED}.log"
        echo "[GPU${GPU}] -> ${EXP} ${ARCH} seed=${SEED}"
        SEED="${SEED}" "$TRAIN_ICI" "$GPU" within_cohort "$EXP" "$ARCH" \
          > "$LOGFILE" 2>&1 &
        PIDS+=($!)
      done
    done
  done

  wait "${PIDS[@]}"
  echo "[GPU${GPU}] ${EXPS[*]} DONE"
}

echo "Launching within-cohort benchmarking across 4 GPUs (8 parallel jobs/GPU)..."
echo "Logs → ${LOGDIR}/"
echo ""

run_gpu 4 HVG1    HVG2    &
run_gpu 5 HVG3    BASELINE &
run_gpu 6 GS      FM_BULK  &
run_gpu 7 FM_SC   PCA      &

wait
echo ""
echo "============================================================"
echo " All within-cohort runs finished. Logs in: ${LOGDIR}/"
echo "============================================================"
