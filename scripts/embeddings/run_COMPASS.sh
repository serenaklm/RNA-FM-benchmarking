#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
GPU=""
DATA_DIR=""
CANCER_CODE=""

usage() { echo "Usage: $0 --dataset <DATASET> --gpu <GPU> [--data_dir <DIR>] [--cancer_code <INT>]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)     DATASET="$2";     shift 2;;
    --gpu)         GPU="$2";         shift 2;;
    --data_dir)    DATA_DIR="$2";    shift 2;;
    --cancer_code) CANCER_CODE="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" || -z "$GPU" ]] && usage

cd data_preprocessing/get_FM_embeddings

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]]    && EXTRA_ARGS+=(--data_dir "$DATA_DIR")
[[ -n "$CANCER_CODE" ]] && EXTRA_ARGS+=(--cancer_code "$CANCER_CODE")

conda activate compass
CUDA_VISIBLE_DEVICES=$GPU python COMPASS.py --data_set "$DATASET" "${EXTRA_ARGS[@]}"
conda deactivate