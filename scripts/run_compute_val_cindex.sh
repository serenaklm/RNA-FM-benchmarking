#!/bin/bash
# Runs compute_val_cindex.py for all 4 paper datasets, one per GPU (0-3) in parallel.
# Usage: bash scripts/run_compute_val_cindex.sh

LOG_DIR="logs/val_cindex"
mkdir -p $LOG_DIR

CUDA_VISIBLE_DEVICES=4 python -m val.compute_val_cindex --dataset TCGA       > $LOG_DIR/TCGA.log       2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m val.compute_val_cindex --dataset Hartwig     > $LOG_DIR/Hartwig.log    2>&1 &
CUDA_VISIBLE_DEVICES=6 python -m val.compute_val_cindex --dataset CPTAC       > $LOG_DIR/CPTAC.log      2>&1 &
CUDA_VISIBLE_DEVICES=7 python -m val.compute_val_cindex --dataset TARGET_AML  > $LOG_DIR/TARGET_AML.log 2>&1 &

wait
echo "All datasets complete."
