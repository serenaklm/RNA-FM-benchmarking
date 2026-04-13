#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

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

BULKFORMER_DIR=$(python -c "import yaml; print(yaml.safe_load(open('${REPO_ROOT}/configs/data_paths.yaml'))['model_repos']['BulkFormer'])")
cd "$BULKFORMER_DIR"

conda activate bulkformer
CUDA_VISIBLE_DEVICES=$GPU python get_BulkFormer_embeddings.py --data_set "$DATASET"
conda deactivate