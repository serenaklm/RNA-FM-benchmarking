#!/usr/bin/env bash
set -euo pipefail

DATASET=""
GPU=""

LOGROOT="./logs/embeddings"

usage() {
  echo "Usage: $0 --dataset <DATASET> --gpu <GPU>"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" || -z "$GPU" ]] && usage

DATA_LOG_DIR="${LOGROOT}/${DATASET}"
mkdir -p "$DATA_LOG_DIR"

echo "===================================="
echo "Running embeddings for dataset: $DATASET"
echo "GPU: $GPU"
echo "Logs: $DATA_LOG_DIR"
echo "===================================="

run_step () {
    NAME="$1"
    CMD="$2"

    LOGFILE="${DATA_LOG_DIR}/${NAME}.log"
    OKFILE="${DATA_LOG_DIR}/${NAME}.ok"

    if [[ -f "$OKFILE" ]]; then
        echo "Skipping $NAME (already completed)"
        return
    fi

    echo ""
    echo "===== Running $NAME ====="
    echo "Log: $LOGFILE"

    (
      set -x
      eval "$CMD"
    ) 2>&1 | tee "$LOGFILE"

    touch "$OKFILE"

    echo "===== Finished $NAME ====="
}

########################################
# GPU EMBEDDINGS FIRST
########################################

run_step "scGPT" \
"bash scripts/embeddings/run_scGPT.sh \
  --dataset $DATASET \
  --gpu $GPU"

run_step "COMPASS" \
"bash scripts/embeddings/run_COMPASS.sh \
  --dataset $DATASET \
  --gpu $GPU"

run_step "BulkFormer" \
"bash scripts/embeddings/run_BulkFormer.sh \
  --dataset $DATASET \
  --gpu $GPU"

run_step "CancerFoundation" \
"bash scripts/embeddings/run_CancerFoundation.sh \
  --dataset $DATASET \
  --gpu $GPU"

########################################
# CPU EMBEDDINGS
########################################

run_step "ssGSEA" \
"bash scripts/embeddings/run_ssGSEA.sh \
  --dataset $DATASET"

run_step "STATE" \
"bash scripts/embeddings/run_STATE.sh \
  --dataset $DATASET"

# Run BulkRNABert on CPU due to memory constraints 
# (on CPU it is very slow, approximately 150 samples per hour)
#run_step "BulkRNABert" \
#"bash scripts/embeddings/run_BulkRNABert.sh \
#  --dataset $DATASET"

run_step "PCA" \
"cd data_preprocessing && python compute_PCA.py --data_set $DATASET"

echo ""
echo "===================================="
echo "ALL EMBEDDINGS COMPLETED SUCCESSFULLY"
echo "===================================="