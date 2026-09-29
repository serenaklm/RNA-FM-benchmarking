#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

DATASET=""
DATA_DIR=""
GPU=""
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_MODEL="${STATE_MODEL:-${REPO_ROOT}/data/STATE/SE-600M}"

usage() { echo "Usage: $0 --dataset <DATASET> [--gpu <GPU>] [--data_dir <DIR>]"; exit 1; }

PASSTHRU=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)  DATASET="$2";  shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --gpu)      GPU="$2";      shift 2;;
    *) PASSTHRU+=("$1"); shift;;  # other flags (e.g. --sample_ids FILE) go to the python script
  esac
done

[[ -z "$DATASET" ]] && usage

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

# Resolve input/output paths for the `state emb transform` CLI call
if [[ -n "$DATA_DIR" ]]; then
  STATE_INPUT="${DATA_DIR}/RNA/${DATASET}_bulk_RNA_processed_log1p_HVG.h5ad"
  STATE_OUTPUT="${DATA_DIR}/RNA/FM_embeddings/STATE/${DATASET}_STATE_emb.h5ad"
else
  STATE_INPUT="${REPO_ROOT}/data/${DATASET}/RNA/${DATASET}_bulk_RNA_processed_log1p_HVG.h5ad"
  STATE_OUTPUT="${REPO_ROOT}/data/${DATASET}/RNA/FM_embeddings/STATE/${DATASET}_STATE_emb.h5ad"
fi

EXTRA_ARGS=()
[[ -n "$DATA_DIR" ]] && EXTRA_ARGS+=(--data_dir "$DATA_DIR")

mkdir -p "$(dirname "$STATE_OUTPUT")"

cd "${REPO_ROOT}/data_preprocessing/get_FM_embeddings"

# HVG selection needs scanpy; run it in rna_modeling rather than whatever python is active
conda activate rna_modeling
python STATE.py --data_set "$DATASET" --action process_hvg "${EXTRA_ARGS[@]}" "${PASSTHRU[@]}"
conda deactivate

conda activate state

# Call the env's own `state`, not whichever `state` comes first on PATH (e.g. ~/.uv/bin)
"$CONDA_PREFIX/bin/state" emb transform \
 --model-folder "$STATE_MODEL" \
 --input  "$STATE_INPUT" \
 --output "$STATE_OUTPUT"

conda deactivate

conda activate rna_modeling
python STATE.py --data_set "$DATASET" --action extract_embeddings "${EXTRA_ARGS[@]}"
conda deactivate