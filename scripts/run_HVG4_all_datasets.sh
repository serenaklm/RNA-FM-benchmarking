#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_HVG4_all_datasets.sh
#
# Runs HVG4 (44, 256, 512, 640, 2058 HVGs — FM-matched dims)
# across TCGA, CPTAC, Hartwig, and TARGET_AML.
#
# One GPU per dataset; each GPU runs all 4 arch/loss combos
# (Linear/ResidualMLP x Cox/CIndex) in parallel for seed 37,
# then repeats for seed 42.
#
# GPU assignments (override via env vars):
#   GPU_TCGA        (default 4)
#   GPU_CPTAC       (default 5)
#   GPU_HARTWIG     (default 6)
#   GPU_TARGET_AML  (default 7)
# ============================================================

TASK="survival"
EXP="HVG4"

GPU_TCGA="${GPU_TCGA:-4}"
GPU_CPTAC="${GPU_CPTAC:-5}"
GPU_HARTWIG="${GPU_HARTWIG:-6}"
GPU_TARGET_AML="${GPU_TARGET_AML:-7}"

ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)

LOGDIR="./logs/hvg4_all_datasets"
mkdir -p "${LOGDIR}"

# Runs HVG4 for one dataset on one GPU, both seeds in parallel
run_dataset () {
  local GPU="$1" DATASET="$2"
  local PIDS=()
  echo "[GPU${GPU} | ${DATASET}] Launching HVG4 (all 8 jobs)..."

  for SEED in 37 42; do
    for ARCH in "${ARCHS[@]}"; do
      for LOSS in "${LOSSES[@]}"; do
        local LOGFILE="${LOGDIR}/gpu${GPU}_${DATASET}_HVG4_${ARCH}_${LOSS}_seed${SEED}.log"
        echo "[GPU${GPU}] -> ${DATASET} ${ARCH} ${LOSS} seed=${SEED}"
        SEED="${SEED}" nohup ./scripts/train_RNA_models.sh \
          "$GPU" "$DATASET" "$TASK" "$EXP" "$ARCH" "$LOSS" \
          > "$LOGFILE" 2>&1 &
        PIDS+=($!)
      done
    done
  done

  echo "[GPU${GPU} | ${DATASET}] Waiting for ${#PIDS[@]} jobs..."
  wait "${PIDS[@]}"
  echo "[GPU${GPU} | ${DATASET}] done."
}

run_dataset "$GPU_TCGA"       TCGA       &
run_dataset "$GPU_CPTAC"      CPTAC      &
run_dataset "$GPU_HARTWIG"    Hartwig    &
run_dataset "$GPU_TARGET_AML" TARGET_AML &

wait
echo "All HVG4 sweeps finished. Logs in: ${LOGDIR}"
