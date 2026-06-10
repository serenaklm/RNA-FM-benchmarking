#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
DATA_DIR=""
GPU=""
STATE_MODEL="/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/STATE/SE-600M"

usage() { echo "Usage: $0 --dataset <DATASET> [--gpu <GPU>] [--data_dir <DIR>]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --gpu)      GPU="$2";      shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" ]] && usage

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

# Resolve input/output paths for the `state emb transform` CLI call
if [[ -n "$DATA_DIR" ]]; then
  STATE_INPUT="${DATA_DIR}/RNA/${DATASET}_bulk_RNA_processed_log1p_HVG.h5ad"
  STATE_OUTPUT="${DATA_DIR}/RNA/FM_embeddings/STATE/${DATASET}_STATE_emb.h5ad"
else
  STATE_INPUT="/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/${DATASET}/RNA/${DATASET}_bulk_RNA_processed_log1p_HVG.h5ad"
  STATE_OUTPUT="/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/${DATASET}/RNA/FM_embeddings/STATE/${DATASET}_STATE_emb.h5ad"
fi

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]] && EXTRA_ARGS+=(--data_dir "$DATA_DIR")

mkdir -p "$(dirname "$STATE_OUTPUT")"

cd data_preprocessing/get_FM_embeddings

python STATE.py --data_set "$DATASET" --action process_hvg "${EXTRA_ARGS[@]}"

conda activate state

state emb transform \
 --model-folder "$STATE_MODEL" \
 --input  "$STATE_INPUT" \
 --output "$STATE_OUTPUT"

conda deactivate

python STATE.py --data_set "$DATASET" --action extract_embeddings "${EXTRA_ARGS[@]}"