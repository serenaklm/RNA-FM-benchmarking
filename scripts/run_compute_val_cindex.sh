#!/bin/bash
# Runs compute_val_cindex.py for all 4 paper datasets, one per GPU, in parallel.
# Usage: bash scripts/run_compute_val_cindex.sh
#   GPUS="4 5 6 7" bash scripts/run_compute_val_cindex.sh   # override GPU list

LOG_DIR="logs/val_cindex"
mkdir -p $LOG_DIR

read -ra GPUS <<< "${GPUS:-0 1 2 3}"

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPUS[0]} python -m val.compute_val_cindex --dataset TCGA       > $LOG_DIR/TCGA.log       2>&1 &
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPUS[1]} python -m val.compute_val_cindex --dataset Hartwig    > $LOG_DIR/Hartwig.log    2>&1 &
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPUS[2]} python -m val.compute_val_cindex --dataset CPTAC      > $LOG_DIR/CPTAC.log      2>&1 &
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${GPUS[3]} python -m val.compute_val_cindex --dataset TARGET_AML > $LOG_DIR/TARGET_AML.log 2>&1 &

wait
echo "All datasets complete."
