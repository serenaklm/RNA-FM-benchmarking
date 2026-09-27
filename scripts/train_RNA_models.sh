#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Usage:
#   ./scripts/train_RNA_models.sh <GPU_ID> <DATASET> <TASK> <EXPERIMENT> <ARCH> <LOSS_NAME>
#
# DATASET:   TCGA | CPTAC | Hartwig | TARGET_AML
# TASK:      survival | response
# EXPERIMENT:
#   HVG        — 8 HVG sizes (18, 50, 132, 500, 729, 1250, 5000, 10000)
#   HVG1/2/3   — subsets of the HVG sizes (for parallel execution)
#   HVG4       — FM-matched HVG sizes (44, 256, 512, 640, 1280, 2058)
#   GS         — all gene-set embeddings (4 collections × max/ssGSEA)
#   SSGSEA | MAX | HALLMARK — gene-set subsets
#   FM_BULK    — BulkFormer, BulkRNABert, COMPASS
#   FM_SC      — scGPT, STATE, CancerFoundation, UCE
#   PCA        — PCA_50, PCA_100 (fit on each training split)
#   BASELINE   — raw expression (all genes)
#   ALL | TEST
# ARCH:      Linear | MLP | ResidualMLP
# LOSS_NAME: Cox | CIndex   (ignored for TASK=response, but still required by this script)
#
# Examples:
#   ./scripts/train_RNA_models.sh 0 TCGA   survival HVG ResidualMLP CIndex
#   ./scripts/train_RNA_models.sh 0 CPTAC  survival HVG Linear      Cox
#   ./scripts/train_RNA_models.sh 0 TARGET_AML survival FM_SC ResidualMLP Cox
#   ./scripts/train_RNA_models.sh 0 Hartwig survival FM_BULK Linear Cox
#   ./scripts/train_RNA_models.sh 0 Hartwig response HVG ResidualMLP Cox
# ============================================================

GPU_ID="$1"
DATASET="$2"      # TCGA | CPTAC | Hartwig | TARGET_AML
TASK="$3"         # survival | response
EXPERIMENT="$4"
ARCH="$5"
LOSS_NAME="$6"    # still passed; python can ignore for response

export CUDA_VISIBLE_DEVICES="$GPU_ID"

SEED="${SEED:-37}"
WANDB_PROJECT_SUFFIX="${WANDB_PROJECT_SUFFIX:-}"
SPLITS=(0 1 2 3 4)

N_HVGS_LIST_DEFAULT=("None")
N_HVGS_LIST_HVG=(18 50 132 500 729 1250 5000 10000)
TEST_LIST_HVG=(50)

N_HVGS_LIST_1=(500 729 1250)
N_HVGS_LIST_2=(18 50 5000)
N_HVGS_LIST_3=(132 10000)
N_HVGS_LIST_4=(44 256 512 640 1280 2058)   # matches FM embedding dimensionalities

# Python module path (your new script)
TRAIN_SCRIPT="train.train_RNA_model"

# Single shared config (override with env CONFIG_PATH)
CONFIG_PATH="${CONFIG_PATH:-./configs/all_models.yaml}"

# Tuned HPs path (override with env HP_PATH)
HP_PATH="/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json"

# -----------------------
# Validate inputs
# -----------------------
case "$DATASET" in
  TCGA|CPTAC|CALGB|Hartwig|ICGC|TARGET_AML) ;;
  *)
    echo "Unknown DATASET: $DATASET (must be TCGA|CPTAC|CALGB|Hartwig|ICGC|TARGET_AML)"
    exit 1
    ;;
esac

case "$TASK" in
  survival|response) ;;
  *)
    echo "Unknown TASK: $TASK (must be survival|response)"
    exit 1
    ;;
esac

# Response currently only supported for Hartwig
if [[ "$TASK" == "response" && "$DATASET" != "Hartwig" ]]; then
  echo "TASK=response only supported for DATASET=Hartwig"
  exit 1
fi

# -----------------------
# Output base for skip-check
# -----------------------
# (Task-specific; survival includes endpoint)
if [[ "$TASK" == "survival" ]]; then
  OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs/${DATASET}/survival/PFS"
else
  OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs/${DATASET}/response"
fi

# -----------------------
# Experiment-specific keys
# -----------------------
HVG_KEYS=(RNA_seq)

SSGSEA_KEYS=(
  ssGSEA_AURORA ssGSEA_immuno ssGSEA_hallmark ssGSEA_combined
)

MAX_KEYS=(
  max_AURORA max_immuno max_hallmark max_combined
)

GS_KEYS=(
  max_AURORA max_immuno max_hallmark max_combined
  ssGSEA_AURORA ssGSEA_immuno ssGSEA_hallmark ssGSEA_combined
)

HALLMARK_KEYS=(
  max_hallmark ssGSEA_hallmark
)

FM_BULK_KEYS=(BulkFormer BulkRNABert COMPASS)
FM_SC_KEYS=(scGPT STATE CancerFoundation UCE)

PCA_KEYS=(PCA_50 PCA_100)

KEYS=()
N_HVGS_LIST=("${N_HVGS_LIST_DEFAULT[@]}")

case "$EXPERIMENT" in
  HVG)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${N_HVGS_LIST_HVG[@]}")
    ;;
  HVG1)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${N_HVGS_LIST_1[@]}")
    ;;
  HVG2)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${N_HVGS_LIST_2[@]}")
    ;;
  HVG3)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${N_HVGS_LIST_3[@]}")
    ;;
  HVG4)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${N_HVGS_LIST_4[@]}")
    ;;
  GS)
    KEYS=("${GS_KEYS[@]}")
    ;;
  SSGSEA)
    KEYS=("${SSGSEA_KEYS[@]}")
    ;;
  MAX)
    KEYS=("${MAX_KEYS[@]}")
    ;;
  HALLMARK)
    KEYS=("${HALLMARK_KEYS[@]}")
    ;;
  FM_BULK)
    KEYS=("${FM_BULK_KEYS[@]}")
    ;;
  FM_SC)
    KEYS=("${FM_SC_KEYS[@]}")
    ;;
  TEST)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("${TEST_LIST_HVG[@]}")
    ;;
  BASELINE)
    KEYS=("${HVG_KEYS[@]}")
    N_HVGS_LIST=("None")
    ;;
  PCA)
    KEYS=("${PCA_KEYS[@]}")
    ;;
  ALL)
    KEYS=("${HVG_KEYS[@]}" "${GS_KEYS[@]}" "${FM_BULK_KEYS[@]}" "${FM_SC_KEYS[@]}")
    ;;
  *)
    echo "Unknown EXPERIMENT: $EXPERIMENT (must be HVG|HVG1|HVG2|HVG3|HVG4|GS|SSGSEA|MAX|HALLMARK|FM_BULK|FM_SC|TEST|ALL)"
    exit 1
    ;;
esac

# Model tag for naming/skip-check
if [[ "$TASK" == "survival" ]]; then
  MODEL_TAG="${ARCH}_${LOSS_NAME}"
else
  MODEL_TAG="${ARCH}_BCE"
fi

echo "============================================================"
echo "GPU_ID=$GPU_ID"
echo "DATASET=$DATASET"
echo "TASK=$TASK"
echo "EXPERIMENT=$EXPERIMENT"
echo "ARCH=$ARCH"
echo "LOSS_NAME=$LOSS_NAME"
echo "MODEL_TAG=$MODEL_TAG"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "HP_PATH=$HP_PATH"
echo "SEED=$SEED"
echo "SPLITS=${SPLITS[*]}"
echo "KEYS=${KEYS[*]}"
echo "N_HVGS_LIST=${N_HVGS_LIST[*]}"
echo "OUT_BASE=$OUT_BASE"
echo "============================================================"

# -----------------------
# Loop over keys/splits/(optional) HVGs
# -----------------------
for KEY in "${KEYS[@]}"; do
  for N_HVG in "${N_HVGS_LIST[@]}"; do
    for SPLIT_IDX in "${SPLITS[@]}"; do

      # Only use the HVG sweep when KEY is RNA_seq; otherwise force None
      RUN_N_HVG="$N_HVG"
      if [[ "$KEY" != "RNA_seq" ]]; then
        RUN_N_HVG="None"
      fi

      PREFIX="${MODEL_TAG}_rnaKey_${KEY}_nHVGs_${RUN_N_HVG}_seed_${SEED}_split_${SPLIT_IDX}_"

      if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null; then
        echo "SKIP (already exists): ${PREFIX}*"
        continue
      fi

      echo
      echo "Running: dataset=$DATASET task=$TASK experiment=$EXPERIMENT split=$SPLIT_IDX key=$KEY n_hvgs=$RUN_N_HVG seed=$SEED (GPU=$GPU_ID) model=$MODEL_TAG"

      CMD=(python -m "$TRAIN_SCRIPT"
        --config_path "$CONFIG_PATH"
        --dataset "$DATASET"
        --task "$TASK"
        --hp_path "$HP_PATH"
        --arch "$ARCH"
        --loss_name "$LOSS_NAME"
        --seed "$SEED"
        --split_idx "$SPLIT_IDX"
        --rna_embedding_key "$KEY"
        --wandb_project_suffix "$WANDB_PROJECT_SUFFIX"
      )

      if [[ "$RUN_N_HVG" != "None" ]]; then
        CMD+=(--n_HVGs "$RUN_N_HVG")
      fi

      echo "${CMD[@]}"
      "${CMD[@]}"
    done

    # If KEY != RNA_seq don't repeat for every N_HVG
    if [[ "$KEY" != "RNA_seq" ]]; then
      break
    fi
  done
done