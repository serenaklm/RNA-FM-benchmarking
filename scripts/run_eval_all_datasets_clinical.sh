#!/bin/bash

GPU=2
LOG_DIR="logs/eval"
mkdir -p $LOG_DIR

for dataset in TCGA Hartwig CPTAC TARGET_AML; do
    CUDA_VISIBLE_DEVICES=$GPU python -m test.eval_all_RNA_models_clinical \
        --dataset $dataset --endpoint PFS > $LOG_DIR/${dataset}_clinical_concat.log 2>&1 &
done

wait
echo "All done"
