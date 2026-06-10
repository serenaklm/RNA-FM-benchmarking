#!/usr/bin/env bash
set -euo pipefail

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh

DATASET=""
GPU=""
DATA_DIR=""

usage() { echo "Usage: $0 --dataset <DATASET> --gpu <GPU> [--data_dir <DIR>]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2;;
    --gpu)      GPU="$2";      shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" || -z "$GPU" ]] && usage

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]] && EXTRA_ARGS+=(--data_dir "$DATA_DIR")

cd /data/rbg/users/ashyamal/CancerFoundation/tutorial

conda activate cancerfoundation
CUDA_VISIBLE_DEVICES=$GPU python CancerFoundation.py --data_set "$DATASET" "${EXTRA_ARGS[@]}"
conda deactivate