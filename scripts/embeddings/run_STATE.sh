#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
STATE_MODEL="/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/STATE/SE-600M"

usage() { echo "Usage: $0 --dataset <DATASET>"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" ]] && usage

cd data_preprocessing/get_FM_embeddings

python STATE.py --data_set "$DATASET" --action process_hvg

conda activate state

state emb transform \
 --model-folder "$STATE_MODEL" \
 --input "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/${DATASET}/RNA/${DATASET}_bulk_RNA_processed_log1p_HVG.h5ad" \
 --output "/data/rbg/users/ashyamal/RNA-FM-benchmarking/data/${DATASET}/RNA/FM_embeddings/STATE/${DATASET}_STATE_emb.h5ad"

conda deactivate

python STATE.py --data_set "$DATASET" --action extract_embeddings