#!/usr/bin/env bash
set -euo pipefail

DATASET=""
GPU=""
DATA_DIR=""

usage() { echo "Usage: $0 --dataset <DATASET> [--gpu <GPU>] [--data_dir <DIR>]"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2;;
    --gpu)      GPU="$2";      shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" ]] && usage

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]] && EXTRA_ARGS+=(--data_dir "$DATA_DIR")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"

source /data/rbg/users/ashyamal/miniconda3/etc/profile.d/conda.sh
conda activate rna_modeling

python BulkRNABert.py --data_set "$DATASET" "${EXTRA_ARGS[@]}"