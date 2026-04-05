#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
GPU=""

usage() { echo "Usage: $0 --dataset <DATASET> --gpu <GPU>"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" || -z "$GPU" ]] && usage

cd ../BulkFormer/BulkFormer

conda activate bulkformer
CUDA_VISIBLE_DEVICES=$GPU python get_BulkFormer_embeddings.py --data_set "$DATASET"
conda deactivate