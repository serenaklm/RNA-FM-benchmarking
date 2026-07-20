#!/bin/bash
# Runs compute_val_cindex.py for all 4 paper datasets, one per GPU (0-3) in parallel.
# Usage: bash scripts/run_compute_val_cindex.sh

LOG_DIR="logs/val_cindex"
mkdir -p $LOG_DIR

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 python -m val.compute_val_cindex --dataset Hartwig     > $LOG_DIR/Hartwig.log    2>&1 &

wait
echo "All datasets complete."
