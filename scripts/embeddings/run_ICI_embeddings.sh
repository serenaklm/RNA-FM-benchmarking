#!/usr/bin/env bash
# =============================================================================
# run_ICI_embeddings.sh
#
# Run all embedding steps (FM + ssGSEA + PCA) for a single ICI cohort.
# Reads cohort metadata from configs/immunotherapy_cohorts.yaml.
#
# Usage:
#   ./scripts/embeddings/run_ICI_embeddings.sh --cohort Hugo --gpu 0
#   ./scripts/embeddings/run_ICI_embeddings.sh --cohort Riaz --gpu 1
#   ./scripts/embeddings/run_ICI_embeddings.sh --cohort Rose --gpu 2
#
# After all cohorts are processed, assemble the final pickles:
#   python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Hugo
#   python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Riaz
#   python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort Rose
#   python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge
#
# Notes:
#   - All FM scripts (including BulkFormer and CancerFoundation) accept --data_dir
#     so no symlinks are needed.
#   - The TCGA gene list is used for PCA and ssGSEA (--gene_name_path override).
# =============================================================================
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

COHORT=""
GPU=""

usage() {
  echo "Usage: $0 --cohort <COHORT> --gpu <GPU>"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cohort) COHORT="$2"; shift 2;;
    --gpu)    GPU="$2";    shift 2;;
    *) usage;;
  esac
done

[[ -z "$COHORT" || -z "$GPU" ]] && usage

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data/immunotherapy_cohorts/${COHORT}"
LOGROOT="${REPO_ROOT}/logs/embeddings/ICI/${COHORT}"
TCGA_GENE_LIST="${REPO_ROOT}/data/TCGA/TCGA_genes_ordered_by_variance.pkl"
CONFIG="${REPO_ROOT}/configs/immunotherapy_cohorts.yaml"

# Read cancer code for this cohort from the YAML config
CANCER_CODE=$(python3 -c "
import yaml, sys
cfg = yaml.safe_load(open('${CONFIG}'))
code = cfg['cohorts'].get('${COHORT}', {}).get('tcga_cancer_code')
if code is None:
    print(f'ERROR: tcga_cancer_code not found for cohort ${COHORT}', file=sys.stderr)
    sys.exit(1)
print(code)
")

mkdir -p "$LOGROOT"

echo "===================================="
echo "ICI Cohort  : $COHORT"
echo "GPU         : $GPU"
echo "Data dir    : $DATA_DIR"
echo "Cancer code : $CANCER_CODE"
echo "Logs        : $LOGROOT"
echo "===================================="

# ── Step runner ────────────────────────────────────────────────────────────────
run_step() {
  NAME="$1"
  CMD="$2"
  LOGFILE="${LOGROOT}/${NAME}.log"
  OKFILE="${LOGROOT}/${NAME}.ok"

  if [[ -f "$OKFILE" ]]; then
    echo "Skipping $NAME (already completed)"
    return
  fi

  echo ""
  echo "===== Running $NAME ====="
  ( set -x; eval "$CMD" ) 2>&1 | tee "$LOGFILE"
  touch "$OKFILE"
  echo "===== Finished $NAME ====="
}

# ── GPU FM embeddings ──────────────────────────────────────────────────────────

run_step "scGPT" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_scGPT.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR}"

run_step "COMPASS" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_COMPASS.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR} \
    --cancer_code ${CANCER_CODE}"

run_step "BulkFormer" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_BulkFormer.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR}"

run_step "CancerFoundation" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_CancerFoundation.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR}"

run_step "UCE" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_UCE.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR}"

# ── CPU embeddings ─────────────────────────────────────────────────────────────

run_step "ssGSEA" \
  "cd ${REPO_ROOT}/data_preprocessing && \
   python get_ssGSEA_scores.py \
     --gene_set all \
     --data_set $COHORT \
     --data_dir ${DATA_DIR} \
     --gene_name_path ${TCGA_GENE_LIST} \
     --output_dir ${DATA_DIR}/ssGSEA_scores"

run_step "STATE" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_STATE.sh \
    --dataset $COHORT --gpu $GPU \
    --data_dir ${DATA_DIR}"

run_step "PCA" \
  "cd ${REPO_ROOT}/data_preprocessing && \
   python compute_PCA.py \
     --data_set $COHORT \
     --data_dir ${DATA_DIR} \
     --gene_name_path ${TCGA_GENE_LIST}"

run_step "BulkRNABert" \
  "bash ${REPO_ROOT}/scripts/embeddings/run_BulkRNABert.sh \
    --dataset $COHORT \
    --data_dir ${DATA_DIR}"

echo ""
echo "===================================="
echo "ALL ICI EMBEDDINGS DONE: $COHORT"
echo "===================================="
echo "Next steps:"
echo "  python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort $COHORT"
echo "  python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge  (after all cohorts)"
echo "  python data_preprocessing/get_ICI_data_splits.py                               (after merge)"
