#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# scripts/run_within_8gpu.sh
#
# Within-cohort sweep across all 8 GPUs.
# Identical structure to run_loco_8gpu.sh — 1 experiment per GPU,
# 2 archs × 2 seeds all in parallel = 4 concurrent jobs per GPU.
#
# GPU allocation (1 experiment per GPU):
#   GPU 0: HVG1
#   GPU 1: HVG2
#   GPU 2: HVG3
#   GPU 3: BASELINE
#   GPU 4: GS
#   GPU 5: FM_BULK
#   GPU 6: FM_SC
#   GPU 7: PCA
#
# Peak concurrency: 8 GPUs × 4 jobs = 32 concurrent training runs
# Total models: 2 cohorts × 25 emb × 5 splits × 2 seeds × 2 archs = 1,000
# (IMVigor210 + IMMotion150 only — Hugo/Riaz/Rose too small for within-cohort CV)
#
# Usage:
#   ./scripts/run_within_8gpu.sh
#
# Override config/HP paths:
#   CONFIG_PATH=... HP_PATH=... ./scripts/run_within_8gpu.sh
# =============================================================================

LOGDIR="./logs/ICI_within_cohort"
mkdir -p "$LOGDIR"

TRAIN_ICI="./scripts/train_ICI_models.sh"

export CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
export HP_PATH="${HP_PATH:-./hp_results_ICI/all_aggregated_hps_ICI.json}"
export COHORTS_FILTER="IMVigor210 IMMotion150"

ARCHS=(Linear ResidualMLP)
SEEDS=(37 42)

run_gpu () {
  local GPU="$1"
  local EXP="$2"
  local PIDS=()

  echo "[GPU${GPU}] Launching within_cohort ${EXP} — all arch+seed combos"
  for ARCH in "${ARCHS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      local LOGFILE="${LOGDIR}/gpu${GPU}_within_${EXP}_${ARCH}_seed${SEED}.log"
      echo "[GPU${GPU}] -> ${EXP} ${ARCH} seed=${SEED}"
      SEED="${SEED}" "$TRAIN_ICI" "$GPU" within_cohort "$EXP" "$ARCH" \
        > "$LOGFILE" 2>&1 &
      PIDS+=($!)
    done
  done

  wait "${PIDS[@]}"
  echo "[GPU${GPU}] ${EXP} DONE"
}

echo "Launching within-cohort benchmarking across 8 GPUs (4 parallel jobs/GPU)..."
echo "Logs → ${LOGDIR}/"
echo ""

run_gpu 0 HVG1     &
run_gpu 1 HVG2     &
run_gpu 2 HVG3     &
run_gpu 3 BASELINE &
run_gpu 4 GS       &
run_gpu 5 FM_BULK  &
run_gpu 6 FM_SC    &
run_gpu 7 PCA      &

wait
echo ""
echo "============================================================"
echo " All within-cohort runs finished. Logs in: ${LOGDIR}/"
echo "============================================================"
