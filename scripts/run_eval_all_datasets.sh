#!/bin/bash

GPU="${GPU:-6}"
LOG_DIR="logs/eval"
mkdir -p $LOG_DIR

for dataset in TCGA Hartwig CPTAC TARGET_AML; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$GPU python -m test.eval_all_RNA_models \
        --dataset $dataset --task survival --endpoint PFS > $LOG_DIR/${dataset}.log 2>&1 &
done

wait
echo "All done"