#!/usr/bin/env bash
set -euo pipefail

DATASET=""

usage() { echo "Usage: $0 --dataset <DATASET>"; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    *) usage;;
  esac
done

[[ -z "$DATASET" ]] && usage

cd data_preprocessing/get_FM_embeddings
python BulkRNABert.py --data_set "$DATASET"