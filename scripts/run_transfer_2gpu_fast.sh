#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# scripts/run_transfer_2gpu_fast.sh
#
# Transfer sweep across 2 GPUs, split by train cohort for ~2x speedup.
#
# GPU allocation (all experiments, one train cohort each):
#   GPU 4: all experiments — train=IMVigor210 (4 pairs: IMVigor210→Hugo/Riaz/Rose/IMMotion150)
#   GPU 5: all experiments — train=IMMotion150 (4 pairs: IMMotion150→Hugo/Riaz/Rose/IMVigor210)
#
# Speedup vs run_transfer_4gpu.sh:
#   Each GPU only loops over 4 cohort pairs instead of 8 → ~2x wall-time reduction.
#
# Peak concurrency: 2 GPUs × 32 jobs = 64 concurrent training runs
#   (8 exps × 2 archs × 2 seeds per GPU)
# Total models: 8 pairs × 25 emb × 5 splits × 2 seeds × 2 archs = 4,000
#
# Usage:
#   ./scripts/run_transfer_2gpu_fast.sh
#
# Override GPU IDs:
#   GPU_A=0 GPU_B=1 ./scripts/run_transfer_2gpu_fast.sh
# =============================================================================

LOGDIR="./logs/ICI_transfer"
mkdir -p "$LOGDIR"

TRAIN_ICI="./scripts/train_ICI_models.sh"

export CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
export HP_PATH="${HP_PATH:-./hp_results_ICI/all_aggregated_hps_ICI.json}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

ARCHS=(Linear ResidualMLP)
SEEDS=(37 42)

ALL_EXPS=(HVG1 HVG2 HVG3 BASELINE GS FM_BULK FM_SC PCA)

# ---------------------------------------------------------------------------
# run_gpu <GPU> <TRAIN_COHORT> <EXP1> [EXP2 ...]
#   Launches all (exp × arch × seed) combos in parallel on a single GPU,
#   restricted to the given train cohort.
# ---------------------------------------------------------------------------
run_gpu () {
  local GPU="$1"
  local TRAIN_COHORT="$2"
  shift 2
  local EXPS=("$@")
  local PIDS=()

  echo "[GPU${GPU}] Launching transfer train=${TRAIN_COHORT} ${EXPS[*]} — all exp+arch+seed combos"
  for EXP in "${EXPS[@]}"; do
    for ARCH in "${ARCHS[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        local LOGFILE="${LOGDIR}/gpu${GPU}_transfer_${EXP}_${ARCH}_seed${SEED}_train${TRAIN_COHORT}.log"
        echo "[GPU${GPU}] -> ${EXP} ${ARCH} seed=${SEED} train=${TRAIN_COHORT}"
        SEED="${SEED}" TRAIN_COHORTS_FILTER="${TRAIN_COHORT}" "$TRAIN_ICI" "$GPU" transfer "$EXP" "$ARCH" \
          > "$LOGFILE" 2>&1 &
        PIDS+=($!)
      done
    done
  done

  wait "${PIDS[@]}"
  echo "[GPU${GPU}] train=${TRAIN_COHORT} ${EXPS[*]} DONE"
}

echo "Launching transfer benchmarking across 2 GPUs (32 parallel jobs/GPU)..."
echo "  GPU${GPU_A}: train=IMVigor210"
echo "  GPU${GPU_B}: train=IMMotion150"
echo "Logs → ${LOGDIR}/"
echo ""

run_gpu "$GPU_A" "IMVigor210" "${ALL_EXPS[@]}" &
run_gpu "$GPU_B" "IMMotion150" "${ALL_EXPS[@]}" &

wait
echo ""
echo "============================================================"
echo " All transfer runs finished. Logs in: ${LOGDIR}/"
echo "============================================================"
