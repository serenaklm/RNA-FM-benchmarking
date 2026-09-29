#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

DATASET=""
GPU=""
DATA_DIR=""

usage() { echo "Usage: $0 --dataset <DATASET> --gpu <GPU> [--data_dir <DIR>]"; exit 1; }

PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2;;
    --gpu)      GPU="$2";      shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    *) PASSTHRU+=("$1"); shift;;  # other flags (e.g. --sample_ids FILE) go to the python script
  esac
done

[[ -z "$DATASET" || -z "$GPU" ]] && usage

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]] && EXTRA_ARGS+=(--data_dir "$DATA_DIR")

# CancerFoundation.py (versioned in this repo) runs from the CancerFoundation checkout itself
cd data_preprocessing/get_FM_embeddings

conda activate cancerfoundation
CUDA_VISIBLE_DEVICES=$GPU python CancerFoundation.py --data_set "$DATASET" "${EXTRA_ARGS[@]}" "${PASSTHRU[@]}"
conda deactivate