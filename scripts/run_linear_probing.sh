#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Runs linear probing on 4 GPUs in parallel (one dataset per GPU).
# Within each GPU, launches 8 processes in parallel, each handling
# a non-overlapping slice of the 30 embedding specs.
# Partial CSVs are merged into a single output file at the end.
#
# Usage:
#   ./scripts/run_linear_probing.sh
# ============================================================

BASE="/data/rbg/users/ashyamal/RNA-FM-benchmarking"
OUT_DIR="$BASE/benchmarking_results/linear_probe"
TMP_DIR="$OUT_DIR/tmp"
FINAL_CSV="$OUT_DIR/linear_probing_results.csv"
SCRIPT="mech.linear_probing"
LOG_DIR="logs/linear_probing"

mkdir -p "$OUT_DIR" "$TMP_DIR" "$LOG_DIR"

# 30 embedding specs split into 8 non-overlapping ranges [start, end)
# 4+4+4+4+4+4+3+3 = 30
EMB_STARTS=( 0  4  8 12 16 20 24 27)
EMB_ENDS=(   4  8 12 16 20 24 27 30)

run_dataset() {
    local DATASET="$1"
    local GPU_ID="$2"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"

    echo "[GPU $GPU_ID | $DATASET] Launching 8 embedding-range workers..."

    local job_count=0
    for i in $(seq 0 7); do
        local START="${EMB_STARTS[$i]}"
        local END="${EMB_ENDS[$i]}"
        local OUT_CSV="$TMP_DIR/linear_probing_${DATASET}_${START}_${END}.csv"
        local LOG="$LOG_DIR/${DATASET}_${START}_${END}.log"

        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m "$SCRIPT" \
            --dataset    "$DATASET" \
            --device     cuda:0 \
            --emb_start  "$START" \
            --emb_end    "$END" \
            --out_csv    "$OUT_CSV" \
            --overwrite \
            > "$LOG" 2>&1 &

        job_count=$((job_count + 1))
        if (( job_count % 8 == 0 )); then
            wait
        fi
    done
    wait
    echo "[GPU $GPU_ID | $DATASET] All workers done."
}

# Launch one dataset worker per GPU in parallel
run_dataset TCGA       4 > "$LOG_DIR/TCGA_main.log"       2>&1 &
run_dataset CPTAC      5 > "$LOG_DIR/CPTAC_main.log"      2>&1 &
run_dataset Hartwig    6 > "$LOG_DIR/Hartwig_main.log"    2>&1 &
run_dataset TARGET_AML 7 > "$LOG_DIR/TARGET_AML_main.log" 2>&1 &

wait
echo "All dataset workers finished. Merging CSVs..."

python3 -c "
import pandas as pd, glob, sys
pattern = sys.argv[1]
out     = sys.argv[2]
files   = sorted(glob.glob(pattern))
if not files:
    print('No partial CSVs found — nothing to merge.')
    sys.exit(1)
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df.to_csv(out, index=False)
print(f'Merged {len(files)} files ({len(df)} rows) -> {out}')
" "$TMP_DIR/linear_probing_*.csv" "$FINAL_CSV"

# Remove temp files, leaving only the merged result
rm -rf "$TMP_DIR"

echo "Done. Results at: $FINAL_CSV"
