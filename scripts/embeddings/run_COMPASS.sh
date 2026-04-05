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

cd data_preprocessing/get_FM_embeddings

conda activate compass
CUDA_VISIBLE_DEVICES=$GPU python COMPASS.py --data_set "$DATASET"
conda deactivate