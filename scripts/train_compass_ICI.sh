#!/bin/bash
# train_compass_ICI.sh
#
# Run COMPASS fine-tuning across all ICI evaluation paradigms.
# Loops over: paradigm × cohort_tag × mode × split (5 splits each)
# Runs up to N_GPUS jobs in parallel, one job per GPU (round-robin assignment).
#
# Usage:
#   bash scripts/train_compass_ICI.sh [--paradigm loco|within_cohort|transfer] [--mode LFT|PFT|FFT|NFT] [--n_gpus 8]
#
# Default: run everything across GPUs 0–7.

CONDA_ENV="compass"
REPO="/data/rbg/users/ashyamal/RNA-FM-benchmarking"
PYTHON="conda run -n ${CONDA_ENV} python"

SEED=42
MAX_EPOCHS=500
PATIENCE=10
BATCH_SIZE=32
N_GPUS=8
WANDB_PROJECT="compass-ICI"
USE_WANDB=true

# ── Parse optional overrides ──────────────────────────────────────────────────
FILTER_PARADIGM=""
FILTER_MODE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --paradigm)      FILTER_PARADIGM="$2"; shift 2 ;;
        --mode)          FILTER_MODE="$2";     shift 2 ;;
        --n_gpus)        N_GPUS="$2";          shift 2 ;;
        --wandb_project) WANDB_PROJECT="$2";   shift 2 ;;
        --no_wandb)      USE_WANDB=false;      shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Paradigm definitions ──────────────────────────────────────────────────────
declare -A PARADIGM_TAGS

PARADIGM_TAGS["loco"]="held_out_Hugo held_out_IMMotion150 held_out_IMVigor210 held_out_Riaz held_out_Rose"

PARADIGM_TAGS["within_cohort"]="Hugo IMMotion150 IMVigor210 Riaz Rose"

PARADIGM_TAGS["transfer"]="Hugo_to_IMMotion150 Hugo_to_IMVigor210 Hugo_to_Riaz Hugo_to_Rose \
    IMMotion150_to_Hugo IMMotion150_to_IMVigor210 IMMotion150_to_Riaz IMMotion150_to_Rose \
    IMVigor210_to_Hugo IMVigor210_to_IMMotion150 IMVigor210_to_Riaz IMVigor210_to_Rose \
    Riaz_to_Hugo Riaz_to_IMMotion150 Riaz_to_IMVigor210 Riaz_to_Rose \
    Rose_to_Hugo Rose_to_IMMotion150 Rose_to_IMVigor210 Rose_to_Riaz"

MODES="LFT PFT FFT"
SPLITS="0 1 2 3 4"

# ── Parallel job runner ───────────────────────────────────────────────────────
# Keeps exactly N_GPUS jobs running; each child gets its own CUDA_VISIBLE_DEVICES.

JOB_COUNT=0
GPU_IDX=0
declare -a PIDS=()

wait_for_slot() {
    # Block until fewer than N_GPUS jobs are running
    while (( ${#PIDS[@]} >= N_GPUS )); do
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
        (( ${#PIDS[@]} >= N_GPUS )) && sleep 2
    done
}

# ── Main loop ─────────────────────────────────────────────────────────────────
cd "${REPO}"

for PARADIGM in loco within_cohort transfer; do
    [[ -n "$FILTER_PARADIGM" && "$PARADIGM" != "$FILTER_PARADIGM" ]] && continue

    for COHORT_TAG in ${PARADIGM_TAGS[$PARADIGM]}; do
        for MODE in $MODES; do
            [[ -n "$FILTER_MODE" && "$MODE" != "$FILTER_MODE" ]] && continue

            for SPLIT in $SPLITS; do
                # Skip if a completed run (has predictions.csv) already exists
                EXISTING=$(find "${REPO}/model_outputs/ICI/compass/${PARADIGM}/${COHORT_TAG}" \
                    -mindepth 1 -maxdepth 1 -type d -name "COMPASS_${MODE}_*_split_${SPLIT}_*" \
                    2>/dev/null | xargs -I{} sh -c '[ -f "{}/predictions.csv" ] && echo {}' 2>/dev/null | head -1)
                if [[ -n "$EXISTING" ]]; then
                    echo "SKIP (done): paradigm=${PARADIGM}  tag=${COHORT_TAG}  mode=${MODE}  split=${SPLIT}"
                    continue
                fi

                wait_for_slot

                GPU=$(( GPU_IDX % N_GPUS ))
                GPU_IDX=$(( GPU_IDX + 1 ))

                echo "==> [GPU ${GPU}] paradigm=${PARADIGM}  tag=${COHORT_TAG}  mode=${MODE}  split=${SPLIT}"

                WANDB_FLAG=""
                $USE_WANDB && WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT}"

                CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON} -m train.train_compass_ICI \
                    --paradigm    "${PARADIGM}" \
                    --cohort_tag  "${COHORT_TAG}" \
                    --split       "${SPLIT}" \
                    --mode        "${MODE}" \
                    --seed        "${SEED}" \
                    --device      "cuda" \
                    --max_epochs  "${MAX_EPOCHS}" \
                    --patience    "${PATIENCE}" \
                    --batch_size  "${BATCH_SIZE}" \
                    ${WANDB_FLAG} \
                    > "${REPO}/logs/compass_${PARADIGM}_${COHORT_TAG}_${MODE}_split${SPLIT}.log" 2>&1 &

                PIDS+=($!)
            done
        done
    done
done

# Wait for all remaining jobs
echo "Waiting for remaining jobs..."
wait
echo "All COMPASS training runs complete."
