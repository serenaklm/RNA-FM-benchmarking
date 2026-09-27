#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_all_single_dataset.sh
#
# Runs all 32 PFS embeddings for one dataset (both archs, losses, seeds).
#
# GPU 0: HVG1 + HVG2 + HVG3
# GPU 1: HVG4 (FM-matched sizes, incl. 1280 for UCE) + BASELINE
# GPU 2: MAX + FM_SC (scGPT, STATE, CancerFoundation, UCE) + PCA
# GPU 3: SSGSEA + FM_BULK
#
# Usage: ./scripts/run_all_single_dataset.sh <DATASET>
#
# Up to 12 concurrent processes per GPU (experiments x 2 archs x 2 losses)
# Seed 37 first, then seed 42 once GPU is done.
# ============================================================

DATASET=$1
TASK="survival"

ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)

LOGDIR="./logs/${1}_runs"
mkdir -p "${LOGDIR}"

run_gpu () {
  local GPU="$1"
  shift
  local EXPERIMENTS=("$@")

  for SEED in 37 42; do
    local PIDS=()
    echo "[GPU${GPU}] Launching seed=${SEED}: ${EXPERIMENTS[*]}"

    for EXP in "${EXPERIMENTS[@]}"; do
      for ARCH in "${ARCHS[@]}"; do
        for LOSS in "${LOSSES[@]}"; do
          local LOGFILE="${LOGDIR}/gpu${GPU}_${DATASET}_${EXP}_${ARCH}_${LOSS}_seed${SEED}.log"
          echo "[GPU${GPU}] -> ${EXP} ${ARCH} ${LOSS} seed=${SEED}"
          SEED="${SEED}" nohup ./scripts/train_RNA_models.sh \
            "$GPU" "$DATASET" "$TASK" "$EXP" "$ARCH" "$LOSS" \
            > "$LOGFILE" 2>&1 &
          PIDS+=($!)
        done
      done
    done

    echo "[GPU${GPU}] Waiting for seed=${SEED} (${#PIDS[@]} jobs)..."
    wait "${PIDS[@]}"
    echo "[GPU${GPU}] seed=${SEED} done."
  done
}

run_gpu 0 HVG1 HVG2 HVG3       &
run_gpu 1 HVG4 BASELINE    &
run_gpu 2 MAX  FM_SC PCA       &
run_gpu 3 SSGSEA FM_BULK   &

wait
echo "All ${1} runs finished. Logs in: ${LOGDIR}"