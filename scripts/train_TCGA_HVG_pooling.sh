#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# train_TCGA_HVG_pooling.sh
#
# Trains TCGA survival models on HVG-pooled FM embeddings.
#
# GPU 6: BulkRNABert_HVG (1250, 5000, 10000)
# GPU 7: BulkFormer_HVG  (1250, 2000, 5000, 10000)
#
# Per GPU: up to 8 concurrent jobs (2 keys x 2 archs x 2 losses)
# Seeds: 37 then 42 (sequential per GPU)
# ============================================================

DATASET="TCGA"
TASK="survival"
CONFIG_PATH="./configs/all_models_HVG_pooling.yaml"
HP_PATH="/data/rbg/users/ashyamal/RNA-FM-benchmarking/hp_results/all_aggregated_hps.json"
TRAIN_SCRIPT="train.train_RNA_model_on_pooled_HVG_embeds"

ARCHS=(Linear ResidualMLP)
LOSSES=(Cox CIndex)
SPLITS=(0 1 2 3 4)
SEEDS=(37 42)

LOGDIR="./logs/TCGA_HVG_pooling"
mkdir -p "$LOGDIR"

OUT_BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking/model_outputs/HVG_pooling/${DATASET}/survival/PFS"

# ---------------------------------------------------------------------------
# run_key GPU KEY
#   Runs all splits sequentially for one (GPU, KEY, ARCH, LOSS, SEED) combo.
#   Called as a background job.
# ---------------------------------------------------------------------------
run_key () {
  local GPU="$1"
  local KEY="$2"
  local ARCH="$3"
  local LOSS="$4"
  local SEED="$5"

  export CUDA_VISIBLE_DEVICES="$GPU"

  for SPLIT_IDX in "${SPLITS[@]}"; do
    PREFIX="${ARCH}_${LOSS}_rnaKey_${KEY}_nHVGs_None_seed_${SEED}_split_${SPLIT_IDX}_"
    if compgen -G "${OUT_BASE}/${PREFIX}*" > /dev/null 2>&1; then
      echo "[GPU${GPU}] SKIP: ${PREFIX}*"
      continue
    fi

    echo "[GPU${GPU}] Running: key=${KEY} arch=${ARCH} loss=${LOSS} seed=${SEED} split=${SPLIT_IDX}"
    python -m "$TRAIN_SCRIPT" \
      --config_path "$CONFIG_PATH" \
      --dataset "$DATASET" \
      --task "$TASK" \
      --hp_path "$HP_PATH" \
      --arch "$ARCH" \
      --loss_name "$LOSS" \
      --seed "$SEED" \
      --split_idx "$SPLIT_IDX" \
      --rna_embedding_key "$KEY"
  done
}

# ---------------------------------------------------------------------------
# run_gpu GPU KEY_BATCH...
#   For each seed, launches one background job per (key x arch x loss),
#   then waits before moving to next seed.
# ---------------------------------------------------------------------------
run_gpu () {
  local GPU="$1"
  shift
  local KEYS=("$@")

  for SEED in "${SEEDS[@]}"; do
    local PIDS=()
    echo "[GPU${GPU}] Launching seed=${SEED}: keys=(${KEYS[*]})"

    for KEY in "${KEYS[@]}"; do
      for ARCH in "${ARCHS[@]}"; do
        for LOSS in "${LOSSES[@]}"; do
          local LOGFILE="${LOGDIR}/gpu${GPU}_${KEY}_${ARCH}_${LOSS}_seed${SEED}.log"
          run_key "$GPU" "$KEY" "$ARCH" "$LOSS" "$SEED" \
            > "$LOGFILE" 2>&1 &
          PIDS+=($!)
        done
      done
    done

    echo "[GPU${GPU}] Waiting for seed=${SEED} (${#PIDS[@]} jobs)..."
    wait "${PIDS[@]}"
    echo "[GPU${GPU}] seed=${SEED} done."
  done
}

# ---------------------------------------------------------------------------
# GPU 0: BulkRNABert_HVG — batch into groups of 2 keys (8 jobs each)
# GPU 1: BulkFormer_HVG  — batch into groups of 2 keys (8 jobs each)
# ---------------------------------------------------------------------------

gpu6 () {
  run_gpu 6 BulkRNABert_HVG_1250 BulkRNABert_HVG_5000
  run_gpu 6 BulkRNABert_HVG_10000
}

gpu7 () {
  run_gpu 7 BulkFormer_HVG_1250 BulkFormer_HVG_2000
  run_gpu 7 BulkFormer_HVG_5000 BulkFormer_HVG_10000
}

gpu6 &
gpu7 &

wait
echo "All TCGA HVG-pooling runs finished. Logs in: ${LOGDIR}"
