#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# run_all_TCGA.sh
#
# Each GPU runs its 6 experiments for seed 37, then immediately
# starts seed 42 once done — all 4 GPUs in parallel.
# ============================================================

DATASET="TCGA"
TASK="survival"

GPU_LINEAR_COX="${GPU_LINEAR_COX:-4}"
GPU_LINEAR_CINDEX="${GPU_LINEAR_CINDEX:-5}"
GPU_RESIDUALMLP_COX="${GPU_RESIDUALMLP_COX:-6}"
GPU_RESIDUALMLP_CINDEX="${GPU_RESIDUALMLP_CINDEX:-7}"

EXPERIMENTS=(HVG1 HVG2 HVG3 BASELINE MAX SSGSEA)

LOGDIR="./logs/all_tcga_runs"
mkdir -p "${LOGDIR}"

# Runs all experiments for a given arch/loss/gpu across both seeds sequentially
run_gpu () {
  local GPU="$1" ARCH="$2" LOSS="$3"

  for SEED in 37 42; do
    local PIDS=()
    echo "[GPU${GPU} | ${ARCH} ${LOSS}] Launching seed=${SEED}..."

    for EXP in "${EXPERIMENTS[@]}"; do
      local LOGFILE="${LOGDIR}/gpu${GPU}_${EXP}_${ARCH}_${LOSS}_seed${SEED}.log"
      echo "[GPU${GPU}] -> ${EXP} seed=${SEED}"
      SEED="${SEED}" nohup ./scripts/train_RNA_models.sh \
        "$GPU" "$DATASET" "$TASK" "$EXP" "$ARCH" "$LOSS" \
        > "$LOGFILE" 2>&1 &
      PIDS+=($!)
    done

    wait "${PIDS[@]}"
    echo "[GPU${GPU} | ${ARCH} ${LOSS}] seed=${SEED} done."
  done
}

# Launch all 4 GPUs in parallel, each handling both seeds sequentially
run_gpu "$GPU_LINEAR_COX"         Linear      Cox      &
run_gpu "$GPU_LINEAR_CINDEX"      Linear      CIndex   &
run_gpu "$GPU_RESIDUALMLP_COX"    ResidualMLP Cox      &
run_gpu "$GPU_RESIDUALMLP_CINDEX" ResidualMLP CIndex   &

wait
echo "All TCGA sweeps finished. Logs in: ${LOGDIR}"