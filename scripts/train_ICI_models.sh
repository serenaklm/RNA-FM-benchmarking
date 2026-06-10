#!/usr/bin/env bash
# =============================================================================
# scripts/train_ICI_models.sh
#
# Launch ICI response benchmarking training runs.
# Reads cohort names from configs/immunotherapy_cohorts.yaml automatically —
# adding a cohort to the YAML is sufficient to include it in all loops.
#
# Usage:
#   ./scripts/train_ICI_models.sh <GPU_ID> <PARADIGM> <EXPERIMENT> <ARCH>
#
# PARADIGM:   within_cohort | loco | transfer
# EXPERIMENT: HVG | HVG1 | HVG2 | HVG3 | GS | SSGSEA | MAX | FM_BULK | FM_SC | PCA | ALL | TEST
# ARCH:       Linear | MLP | ResidualMLP
#
# Examples:
#   ./scripts/train_ICI_models.sh 0 within_cohort HVG    ResidualMLP
#   ./scripts/train_ICI_models.sh 1 loco          FM_BULK Linear
#   ./scripts/train_ICI_models.sh 2 transfer      GS      MLP
# =============================================================================
set -euo pipefail

GPU_ID="$1"
PARADIGM="$2"
EXPERIMENT="$3"
ARCH="$4"

export CUDA_VISIBLE_DEVICES="$GPU_ID"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/configs/all_models.yaml}"
ICI_CONFIG="${REPO_ROOT}/configs/immunotherapy_cohorts.yaml"
ICI_PKL_DIR="${REPO_ROOT}/data/RNA_embeddings/ICI"
HP_PATH="${HP_PATH:-Not provided}"

SEED="${SEED:-}"   # if set by caller (e.g. run_loco_8gpu.sh), use it; else run both
if [[ -n "$SEED" ]]; then
  SEEDS=("$SEED")
else
  SEEDS=(37 42)
fi
SPLITS=(0 1 2 3 4)

# ── Read cohort names from YAML ────────────────────────────────────────────────
# Requires python + pyyaml (same env used for training)
COHORTS=($(python -c "
import yaml, sys
with open('${ICI_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(' '.join(cfg['cohorts'].keys()))
"))

# Optional filter: COHORTS_FILTER="IMVigor210 IMMotion150" restricts the loop
if [[ -n "${COHORTS_FILTER:-}" ]]; then
    IFS=' ' read -ra COHORTS <<< "$COHORTS_FILTER"
    echo "Cohorts filtered to: ${COHORTS[*]}"
else
    echo "Cohorts from config: ${COHORTS[*]}"
fi

# ── Experiment key lists (same as train_RNA_models.sh) ────────────────────────
HVG_KEYS=(RNA_seq)
SSGSEA_KEYS=(ssGSEA_AURORA ssGSEA_immuno ssGSEA_hallmark ssGSEA_combined)
MAX_KEYS=(max_AURORA max_immuno max_hallmark max_combined)
GS_KEYS=("${MAX_KEYS[@]}" "${SSGSEA_KEYS[@]}")
FM_BULK_KEYS=(BulkFormer BulkRNABert COMPASS)
FM_SC_KEYS=(scGPT STATE CancerFoundation)
PCA_KEYS=(PCA_50 PCA_100)

N_HVGS_LIST_DEFAULT=("None")
N_HVGS_LIST_HVG=(18 50 132 500 729 1250 5000 10000)
N_HVGS_LIST_1=(500 729 1250)
N_HVGS_LIST_2=(18 50 5000)
N_HVGS_LIST_3=(132 10000)
TEST_LIST_HVG=(50)

KEYS=()
N_HVGS_LIST=("${N_HVGS_LIST_DEFAULT[@]}")

case "$EXPERIMENT" in
  HVG)    KEYS=("${HVG_KEYS[@]}");                      N_HVGS_LIST=("${N_HVGS_LIST_HVG[@]}");;
  HVG1)   KEYS=("${HVG_KEYS[@]}");                      N_HVGS_LIST=("${N_HVGS_LIST_1[@]}");;
  HVG2)   KEYS=("${HVG_KEYS[@]}");                      N_HVGS_LIST=("${N_HVGS_LIST_2[@]}");;
  HVG3)   KEYS=("${HVG_KEYS[@]}");                      N_HVGS_LIST=("${N_HVGS_LIST_3[@]}");;
  GS)     KEYS=("${GS_KEYS[@]}");;
  SSGSEA) KEYS=("${SSGSEA_KEYS[@]}");;
  MAX)    KEYS=("${MAX_KEYS[@]}");;
  FM_BULK) KEYS=("${FM_BULK_KEYS[@]}");;
  FM_SC)  KEYS=("${FM_SC_KEYS[@]}");;
  PCA)    KEYS=("${PCA_KEYS[@]}");;
  BASELINE) KEYS=("${HVG_KEYS[@]}");                     N_HVGS_LIST=("None");;
  TEST)   KEYS=("${HVG_KEYS[@]}");                      N_HVGS_LIST=("${TEST_LIST_HVG[@]}");;
  ALL)    KEYS=("${HVG_KEYS[@]}" "${GS_KEYS[@]}" "${FM_BULK_KEYS[@]}" "${FM_SC_KEYS[@]}" "${PCA_KEYS[@]}");;
  *)
    echo "Unknown EXPERIMENT: $EXPERIMENT"
    echo "Valid: HVG|HVG1|HVG2|HVG3|BASELINE|GS|SSGSEA|MAX|FM_BULK|FM_SC|PCA|ALL|TEST"
    exit 1
    ;;
esac

TRAIN_SCRIPT="train.train_ICI_model"
MODEL_TAG="${ARCH}_BCE"

echo "============================================================"
echo "GPU=$GPU_ID  PARADIGM=$PARADIGM  EXPERIMENT=$EXPERIMENT  ARCH=$ARCH"
echo "SEEDS=${SEEDS[*]}  SPLITS=${SPLITS[*]}"
echo "KEYS=${KEYS[*]}"
echo "N_HVGS_LIST=${N_HVGS_LIST[*]}"
echo "============================================================"

# ── Build (dataset, split_folder, cohort_tag) tuples for this paradigm ─────────

run_paradigm() {
  local DATASET="$1"
  local SPLIT_FOLDER="$2"
  local COHORT_TAG="$3"

  local OUT_BASE="${REPO_ROOT}/model_outputs/ICI/${PARADIGM}/${COHORT_TAG}"

  for KEY in "${KEYS[@]}"; do
    for N_HVG in "${N_HVGS_LIST[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        for SPLIT_IDX in "${SPLITS[@]}"; do

          RUN_N_HVG="$N_HVG"
          if [[ "$KEY" != "RNA_seq" ]]; then
            RUN_N_HVG="None"
          fi

          PREFIX="${MODEL_TAG}_rnaKey_${KEY}_nHVGs_${RUN_N_HVG}_seed_${SEED}_split_${SPLIT_IDX}_"
          if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
            echo "SKIP (exists): ${PREFIX}*"
            continue
          fi

          echo ""
          echo "Running: paradigm=$PARADIGM tag=$COHORT_TAG key=$KEY n_hvgs=$RUN_N_HVG seed=$SEED split=$SPLIT_IDX"

          CMD=(python -m "$TRAIN_SCRIPT"
            --config_path "$CONFIG_PATH"
            --dataset     "$DATASET"
            --paradigm    "$PARADIGM"
            --cohort_tag  "$COHORT_TAG"
            --split_folder "$SPLIT_FOLDER"
            --arch        "$ARCH"
            --seed        "$SEED"
            --split_idx   "$SPLIT_IDX"
            --rna_embedding_key "$KEY"
            --hp_path     "$HP_PATH"
          )
          if [[ "$RUN_N_HVG" != "None" ]]; then
            CMD+=(--n_HVGs "$RUN_N_HVG")
          fi

          echo "${CMD[@]}"
          if ! "${CMD[@]}"; then
            echo "FAILED (exit $?): ${PREFIX} — continuing"
          fi

        done
      done
      # If KEY != RNA_seq, don't repeat for every N_HVG
      if [[ "$KEY" != "RNA_seq" ]]; then break; fi
    done
  done
}

# ── Dispatch by paradigm ───────────────────────────────────────────────────────

case "$PARADIGM" in

  within_cohort)
    for COHORT in "${COHORTS[@]}"; do
      DATASET="ICI_merged"
      SPLIT_FOLDER="${REPO_ROOT}/splits/ICI/within_cohort/${COHORT}"
      COHORT_TAG="within_${COHORT}"
      echo ""
      echo ">>> within_cohort: $COHORT"
      run_paradigm "$DATASET" "$SPLIT_FOLDER" "$COHORT_TAG"
    done
    ;;

  loco)
    for HELD_OUT in "${COHORTS[@]}"; do
      DATASET="ICI_merged"
      SPLIT_FOLDER="${REPO_ROOT}/splits/ICI/loco/held_out_${HELD_OUT}"
      COHORT_TAG="loco_held_out_${HELD_OUT}"
      echo ""
      echo ">>> loco: held-out = $HELD_OUT"
      run_paradigm "$DATASET" "$SPLIT_FOLDER" "$COHORT_TAG"
    done
    ;;

  transfer)
    # Optional filter: TRAIN_COHORTS_FILTER="IMVigor210 IMMotion150" restricts training cohorts
    if [[ -n "${TRAIN_COHORTS_FILTER:-}" ]]; then
        IFS=' ' read -ra TRAIN_COHORTS <<< "$TRAIN_COHORTS_FILTER"
        echo "Transfer training cohorts filtered to: ${TRAIN_COHORTS[*]}"
    else
        TRAIN_COHORTS=("${COHORTS[@]}")
        echo "Transfer training cohorts: ${TRAIN_COHORTS[*]}"
    fi
    for TRAIN_COHORT in "${TRAIN_COHORTS[@]}"; do
      for TEST_COHORT in "${COHORTS[@]}"; do
        [[ "$TRAIN_COHORT" == "$TEST_COHORT" ]] && continue
        DATASET="ICI_merged"
        SPLIT_FOLDER="${REPO_ROOT}/splits/ICI/transfer/${TRAIN_COHORT}_to_${TEST_COHORT}"
        COHORT_TAG="transfer_${TRAIN_COHORT}_to_${TEST_COHORT}"
        echo ""
        echo ">>> transfer: $TRAIN_COHORT → $TEST_COHORT"
        run_paradigm "$DATASET" "$SPLIT_FOLDER" "$COHORT_TAG"
      done
    done
    ;;

  *)
    echo "Unknown PARADIGM: $PARADIGM  (must be within_cohort|loco|transfer)"
    exit 1
    ;;
esac

echo ""
echo "============================================================"
echo "All training runs complete."
echo "============================================================"
