#!/bin/bash

GPU=5
LOG_DIR="logs/eval"
mkdir -p $LOG_DIR

# TCGA Hartwig CPTAC TARGET_AML
for dataset in TCGA Hartwig CPTAC TARGET_AML; do
    CUDA_VISIBLE_DEVICES=$GPU python -m test.eval_all_RNA_models_concat \
        --dataset $dataset --endpoint PFS > $LOG_DIR/${dataset}_concat.log 2>&1 &
done

wait
echo "All done"
