#!/usr/bin/env bash
# ============================================================
# Evaluate all binned-HVG ablation runs across 4 datasets.
# Runs each dataset in parallel on a separate GPU.
#
# Usage:
#   # Evaluate all datasets (GPUs 0–3):
#   ./scripts/run_eval_binned_HVG.sh
#
#   # Evaluate a single dataset on a specific GPU:
#   ./scripts/run_eval_binned_HVG.sh <GPU_ID> <DATASET>
#   e.g.  ./scripts/run_eval_binned_HVG.sh 0 TCGA
# ============================================================

LOG_DIR="logs/eval_binned_HVG"
mkdir -p "$LOG_DIR"

run_eval() {
    local GPU_ID="$1"
    local DATASET="$2"
    local LOG_FILE="${LOG_DIR}/${DATASET}.log"

    echo "Evaluating $DATASET on GPU $GPU_ID  →  $LOG_FILE"
    CUDA_VISIBLE_DEVICES="$GPU_ID" python -m test.eval_binned_HVG_models \
        --dataset "$DATASET" \
        > "$LOG_FILE" 2>&1
    echo "$DATASET done"
}

if [[ $# -eq 2 ]]; then
    run_eval "$1" "$2"
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "Usage: $0 [<GPU_ID> <DATASET>]"
    exit 1
fi

run_eval 1 TCGA       &
run_eval 1 Hartwig    &
run_eval 1 CPTAC      &
run_eval 1 TARGET_AML &

wait
echo "All evaluations completed."
