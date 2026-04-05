#!/bin/bash

GPU=5
LOG_DIR="logs/eval"
mkdir -p $LOG_DIR

for dataset in TCGA Hartwig CPTAC TARGET_AML; do
    CUDA_VISIBLE_DEVICES=$GPU python -m test.eval_all_RNA_models \
        --dataset $dataset --task survival --endpoint PFS > $LOG_DIR/${dataset}.log 2>&1 &
done

wait
echo "All done"