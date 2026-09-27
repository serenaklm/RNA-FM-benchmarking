#!/usr/bin/env bash
# =============================================================================
# scripts/tune_ICI_models.sh
#
# Hyperparameter tuning for ICI response prediction.
#
# Tunes on IMVigor210 within-cohort splits (the largest ICI cohort),
# mirroring the PFS approach of tuning on TCGA.
#
# Embeddings tuned (12 total — identical to PFS benchmarking):
#   7 Foundation models : BulkFormer, BulkRNABert, COMPASS, scGPT, STATE,
#                         CancerFoundation, UCE
#   1 HVG               : RNA_seq_1250 (HPs shared across all HVG counts)
#   2 Gene sets         : ssGSEA_combined + max_combined (one per pooling strategy;
#                         HPs shared within strategy across all GS collections)
#   1 PCA               : PCA_50 (HPs shared with PCA_100)
#   1 Raw expression    : RNA_seq (all genes)
#
# Grid:
#   12 embeddings × 2 archs × 5 splits = 120 jobs
#
# Hardware:
#   4 GPUs (IDs 0–3), up to MAX_PER_GPU=16 concurrent jobs per GPU.
#   30 jobs per GPU → batch 1 (16 jobs) + batch 2 (14 jobs) per GPU.
#   Estimated wall time: ~75 minutes (2 batches × ~37 min/batch at 50 trials × ~45s each).
#
# Outputs:
#   logs/tune_ICI/               — per-job stdout/stderr logs
#   hp_results_ICI/
#     tuning_history_ICI.csv     — all Optuna trial results
#     {key}_{arch}_split{N}.json — per-run best HP JSON
#
# After all jobs finish, run:
#   python -m train.agg_best_hps_ICI
# to produce hp_results_ICI/all_aggregated_hps_ICI.json.
#
# Usage:
#   ./scripts/tune_ICI_models.sh              # N_TRIALS=20 (default)
#   N_TRIALS=30 ./scripts/tune_ICI_models.sh  # override trial count
#   NO_WANDB=1  ./scripts/tune_ICI_models.sh  # disable WandB logging
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/logs/tune_ICI"
mkdir -p "$LOG_DIR"

N_TRIALS="${N_TRIALS:-50}"
SEED="${SEED:-37}"
MAX_PER_GPU=16

# Add --no_wandb flag if NO_WANDB env var is set
WANDB_FLAG=""
if [[ -n "${NO_WANDB:-}" ]]; then
  WANDB_FLAG="--no_wandb"
fi

echo "============================================================"
echo "ICI HP Tuning"
echo "  N_TRIALS    : $N_TRIALS"
echo "  GPUs        : 0 1 2 3  (max ${MAX_PER_GPU} concurrent per GPU)"
echo "  WandB       : ${NO_WANDB:+disabled}"
echo "  Log dir     : $LOG_DIR"
echo "  HP out dir  : ${REPO_ROOT}/hp_results_ICI/"
echo "============================================================"

# ── Embeddings: "key:n_hvgs"  (n_hvgs=0 → None / no HVG filter) ─────────────
# Matches PFS benchmarking exactly (agg_best_hps.py RNA_KEYS):
#   7 FMs  — each tuned independently (different embedding dims/scales)
#   1 HVG  — RNA_seq_1250; tuned HPs shared across all HVG counts
#   2 GS   — one per pooling strategy (ssGSEA + max); HPs shared within strategy
#   1 PCA  — PCA_50; tuned HPs shared with PCA_100
#   1 Raw  — RNA_seq (all genes)
# Total: 12 embeddings × 2 archs × 5 splits = 120 jobs
EMBEDDINGS=(
  "BulkFormer:0"
  "BulkRNABert:0"
  "COMPASS:0"
  "scGPT:0"
  "STATE:0"
  "CancerFoundation:0"
  "UCE:0"
  "RNA_seq:1250"        # HVG representative (same as PFS)
  "ssGSEA_combined:0"   # ssGSEA pooling representative (same as PFS)
  "max_combined:0"      # max pooling representative (same as PFS)
  "PCA_50:0"            # PCA representative (same as PFS)
  "RNA_seq:0"           # raw expression — all genes (same as PFS)
)

ARCHS=("Linear" "ResidualMLP")
SPLITS=(0 1 2 3 4)

# ── Build the full ordered list of jobs ───────────────────────────────────────
ALL_JOBS=()
for EMB in "${EMBEDDINGS[@]}"; do
  KEY="${EMB%%:*}"
  NHVG="${EMB##*:}"
  for ARCH in "${ARCHS[@]}"; do
    for SPLIT in "${SPLITS[@]}"; do
      ALL_JOBS+=("${KEY}:${NHVG}:${ARCH}:${SPLIT}")
    done
  done
done

N_JOBS="${#ALL_JOBS[@]}"
echo "Total jobs  : $N_JOBS"
echo ""

# ── Distribute jobs to GPUs round-robin ──────────────────────────────────────
declare -a GPU_JOBS_0=()
declare -a GPU_JOBS_1=()
declare -a GPU_JOBS_2=()
declare -a GPU_JOBS_3=()

for i in "${!ALL_JOBS[@]}"; do
  case $((i % 4)) in
    0) GPU_JOBS_0+=("${ALL_JOBS[$i]}") ;;
    1) GPU_JOBS_1+=("${ALL_JOBS[$i]}") ;;
    2) GPU_JOBS_2+=("${ALL_JOBS[$i]}") ;;
    3) GPU_JOBS_3+=("${ALL_JOBS[$i]}") ;;
  esac
done

# ── Function: run one GPU's jobs in batches of MAX_PER_GPU ───────────────────
run_gpu_jobs() {
  local GPU_ID="$1"
  shift
  local JOBS=("$@")
  local BATCH_PIDS=()
  local batch_count=0
  local total_launched=0

  echo "[GPU $GPU_ID] Starting ${#JOBS[@]} jobs (batches of $MAX_PER_GPU)"

  for JOB in "${JOBS[@]}"; do
    IFS=':' read -r KEY NHVG ARCH SPLIT <<< "$JOB"

    local LOG="${LOG_DIR}/gpu${GPU_ID}_${KEY}_nhvg${NHVG}_${ARCH}_split${SPLIT}.log"

    CUDA_VISIBLE_DEVICES="$GPU_ID" python -m train.tune_ICI \
      --rna_embedding_key "$KEY" \
      --n_HVGs            "$NHVG" \
      --arch              "$ARCH" \
      --split_idx         "$SPLIT" \
      --n_trials          "$N_TRIALS" \
      --seed              "$SEED" \
      ${WANDB_FLAG} \
      > "$LOG" 2>&1 &

    BATCH_PIDS+=($!)
    batch_count=$((batch_count + 1))
    total_launched=$((total_launched + 1))

    # Wait for the current batch to finish before launching more
    if [[ $batch_count -ge $MAX_PER_GPU ]]; then
      echo "[GPU $GPU_ID] Waiting for batch of $batch_count jobs (launched $total_launched / ${#JOBS[@]})..."
      wait "${BATCH_PIDS[@]}"
      BATCH_PIDS=()
      batch_count=0
      echo "[GPU $GPU_ID] Batch done."
    fi
  done

  # Wait for any remaining jobs in the last partial batch
  if [[ ${#BATCH_PIDS[@]} -gt 0 ]]; then
    echo "[GPU $GPU_ID] Waiting for final ${#BATCH_PIDS[@]} jobs..."
    wait "${BATCH_PIDS[@]}"
    echo "[GPU $GPU_ID] Done."
  fi
}

# ── Launch all 4 GPUs in parallel ────────────────────────────────────────────
run_gpu_jobs 0 "${GPU_JOBS_0[@]}" &
run_gpu_jobs 1 "${GPU_JOBS_1[@]}" &
run_gpu_jobs 2 "${GPU_JOBS_2[@]}" &
run_gpu_jobs 3 "${GPU_JOBS_3[@]}" &

wait

echo ""
echo "============================================================"
echo "All tuning jobs complete."
echo ""
echo "Next step — aggregate best HPs:"
echo "  python -m train.agg_best_hps_ICI"
echo ""
echo "Then retrain all ICI models with tuned HPs by passing"
echo "  --hp_path hp_results_ICI/all_aggregated_hps_ICI.json"
echo "to train_ICI_models.sh (or set HP_PATH env var)."
echo "============================================================"
