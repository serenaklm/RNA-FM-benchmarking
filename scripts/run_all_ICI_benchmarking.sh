#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# scripts/run_all_ICI_benchmarking.sh
#
# Top-level 8-GPU sweep for all three ICI evaluation paradigms.
# Dispatches to scripts/train_ICI_models.sh for the heavy lifting.
#
# GPU allocation:
#   GPU 0: within_cohort  + Linear
#   GPU 1: within_cohort  + ResidualMLP
#   GPU 2: loco           + Linear
#   GPU 3: loco           + ResidualMLP
#   GPU 4: transfer       + Linear      [HVG1 + HVG2 + HVG3 + BASELINE]
#   GPU 5: transfer       + Linear      [GS + FM_BULK + FM_SC + PCA]
#   GPU 6: transfer       + ResidualMLP [HVG1 + HVG2 + HVG3 + BASELINE]
#   GPU 7: transfer       + ResidualMLP [GS + FM_BULK + FM_SC + PCA]
#
# Transfer is split across two GPUs per arch because it has 20 cohort pairs.
#
# Total models: ~15,000
#   within_cohort: 5 cohorts × 25 emb × 5 splits × 2 seeds × 2 archs = 2,500
#   loco:          5 held-out × 25 emb × 5 splits × 2 seeds × 2 archs = 2,500
#   transfer:      20 pairs   × 25 emb × 5 splits × 2 seeds × 2 archs = 10,000
#
# Usage:
#   ./scripts/run_all_ICI_benchmarking.sh
#
# Override config/HP paths:
#   CONFIG_PATH=... HP_PATH=... ./scripts/run_all_ICI_benchmarking.sh
# =============================================================================

LOGDIR="./logs/ICI_benchmarking"
mkdir -p "$LOGDIR"

TRAIN_ICI="./scripts/train_ICI_models.sh"

export CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"
export HP_PATH="${HP_PATH:-Not provided}"

# ---------------------------------------------------------------------------
# within_cohort — runs all cohorts, all 25 embeddings, for one arch on one GPU
# ---------------------------------------------------------------------------
run_within () {
  local GPU="$1"
  local ARCH="$2"
  for EXP in HVG1 HVG2 HVG3 BASELINE GS FM_BULK FM_SC PCA; do
    local LOGFILE="${LOGDIR}/gpu${GPU}_within_cohort_${EXP}_${ARCH}.log"
    echo "[GPU${GPU}] within_cohort  exp=${EXP}  arch=${ARCH}"
    "$TRAIN_ICI" "$GPU" within_cohort "$EXP" "$ARCH" \
      > "$LOGFILE" 2>&1
    echo "[GPU${GPU}] within_cohort  exp=${EXP}  arch=${ARCH}  DONE"
  done
}

# ---------------------------------------------------------------------------
# loco — all held-out cohorts, all 25 embeddings, for one arch on one GPU
# ---------------------------------------------------------------------------
run_loco () {
  local GPU="$1"
  local ARCH="$2"
  for EXP in HVG1 HVG2 HVG3 BASELINE GS FM_BULK FM_SC PCA; do
    local LOGFILE="${LOGDIR}/gpu${GPU}_loco_${EXP}_${ARCH}.log"
    echo "[GPU${GPU}] loco  exp=${EXP}  arch=${ARCH}"
    "$TRAIN_ICI" "$GPU" loco "$EXP" "$ARCH" \
      > "$LOGFILE" 2>&1
    echo "[GPU${GPU}] loco  exp=${EXP}  arch=${ARCH}  DONE"
  done
}

# ---------------------------------------------------------------------------
# transfer (RNA_seq) — HVG1 HVG2 HVG3 BASELINE run sequentially on one GPU
# ---------------------------------------------------------------------------
run_transfer_hvg () {
  local GPU="$1"
  local ARCH="$2"
  for EXP in HVG1 HVG2 HVG3 BASELINE; do
    local LOGFILE="${LOGDIR}/gpu${GPU}_transfer_${EXP}_${ARCH}.log"
    echo "[GPU${GPU}] transfer  exp=${EXP}  arch=${ARCH}"
    "$TRAIN_ICI" "$GPU" transfer "$EXP" "$ARCH" \
      > "$LOGFILE" 2>&1
    echo "[GPU${GPU}] transfer  exp=${EXP}  arch=${ARCH}  DONE"
  done
}

# ---------------------------------------------------------------------------
# transfer (GS/FM/PCA) — GS FM_BULK FM_SC PCA on one GPU
# ---------------------------------------------------------------------------
run_transfer_fm () {
  local GPU="$1"
  local ARCH="$2"
  for EXP in GS FM_BULK FM_SC PCA; do
    local LOGFILE="${LOGDIR}/gpu${GPU}_transfer_${EXP}_${ARCH}.log"
    echo "[GPU${GPU}] transfer  exp=${EXP}  arch=${ARCH}"
    "$TRAIN_ICI" "$GPU" transfer "$EXP" "$ARCH" \
      > "$LOGFILE" 2>&1
    echo "[GPU${GPU}] transfer  exp=${EXP}  arch=${ARCH}  DONE"
  done
}

# ---------------------------------------------------------------------------
# Launch all 8 GPU processes in parallel
# ---------------------------------------------------------------------------
echo "Launching ICI benchmarking across 8 GPUs..."
echo "Logs → ${LOGDIR}/"
echo ""

run_within  0 Linear        &
run_within  1 ResidualMLP   &

run_loco    2 Linear        &
run_loco    3 ResidualMLP   &

run_transfer_hvg 4 Linear      &
run_transfer_fm  5 Linear      &

run_transfer_hvg 6 ResidualMLP &
run_transfer_fm  7 ResidualMLP &

wait
echo ""
echo "All ICI benchmarking runs finished. Logs in: ${LOGDIR}"
