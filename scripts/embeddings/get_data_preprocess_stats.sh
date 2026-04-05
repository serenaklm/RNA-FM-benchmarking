#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
GPU=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

usage() {
  echo "Usage: $0 --dataset <DATASET> [--gpu <GPU>]"
  echo "Note: --gpu is accepted but not used — no GPU is required with --just_preprocess_data"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --gpu)     GPU="$2";     shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" ]] && usage

LOGROOT="${REPO_ROOT}/logs/preprocess_stats"
DATA_LOG_DIR="${LOGROOT}/${DATASET}"
mkdir -p "$DATA_LOG_DIR"

echo "===================================="
echo "Collecting preprocessing stats for: $DATASET"
echo "Logs: $DATA_LOG_DIR"
echo "===================================="

# ---------- BulkRNABert (CPU, no conda switch) ----------
echo ""
echo "===== BulkRNABert ====="
cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"
python BulkRNABert.py --data_set "$DATASET" --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/BulkRNABert.log"

# ---------- BulkFormer (bulkformer env, no GPU needed) ----------
echo ""
echo "===== BulkFormer ====="
cd "${REPO_ROOT}/../BulkFormer/BulkFormer"
conda activate bulkformer
python get_BulkFormer_embeddings.py --data_set "$DATASET" --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/BulkFormer.log"
conda deactivate

# ---------- COMPASS (compass env, no GPU needed) ----------
echo ""
echo "===== COMPASS ====="
cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"
conda activate compass
python COMPASS.py --data_set "$DATASET" --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/COMPASS.log"
conda deactivate

# ---------- scGPT (scgpt_env, no GPU needed) ----------
echo ""
echo "===== scGPT ====="
cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"
conda activate scgpt_env
python scGPT.py --data_set "$DATASET" --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/scGPT.log"
conda deactivate

# ---------- STATE (CPU, process_hvg only) ----------
echo ""
echo "===== STATE ====="
cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"
python STATE.py --data_set "$DATASET" --action process_hvg --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/STATE.log"

# ---------- CancerFoundation (cancerfoundation env, no GPU needed) ----------
echo ""
echo "===== CancerFoundation ====="
cd "${REPO_ROOT}/../CancerFoundation/tutorial"
conda activate cancerfoundation
python CancerFoundation.py --data_set "$DATASET" --just_preprocess_data \
  2>&1 | tee "${DATA_LOG_DIR}/CancerFoundation.log"
conda deactivate

cd "$REPO_ROOT"

echo ""
echo "===================================="
echo "ALL PREPROCESSING STATS COLLECTED"
echo "Stats JSON files saved under each FM's embedding folder."
echo "Logs saved to: $DATA_LOG_DIR"
echo "===================================="
