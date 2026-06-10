#!/usr/bin/env bash
# =============================================================================
# run_all_ICI_embeddings.sh
#
# Launch embedding jobs for all 5 ICI cohorts in parallel, one per GPU (3..7).
# Each cohort runs in the background; this script waits for all to finish.
#
# Usage (from repo root):
#   bash scripts/embeddings/run_all_ICI_embeddings.sh
#
# After all jobs complete, assemble pickles and splits:
#   for COHORT in Hugo Riaz Rose IMMotion150 IMVigor210; do
#     python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort $COHORT
#   done
#   python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge
#   python data_preprocessing/get_ICI_data_splits.py
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${REPO_ROOT}/scripts/embeddings/run_ICI_embeddings.sh"
LOGROOT="${REPO_ROOT}/logs/embeddings/ICI"
mkdir -p "$LOGROOT"

declare -A COHORT_GPU=(
  [Hugo]=3
  [Riaz]=4
  [Rose]=5
  [IMMotion150]=6
  [IMVigor210]=7
)

pids=()
cohort_names=()

for COHORT in Hugo Riaz Rose IMMotion150 IMVigor210; do
  GPU="${COHORT_GPU[$COHORT]}"
  LOGFILE="${LOGROOT}/${COHORT}_launcher.log"
  echo "Launching $COHORT on GPU $GPU  (launcher log: $LOGFILE)"
  bash "$SCRIPT" --cohort "$COHORT" --gpu "$GPU" > "$LOGFILE" 2>&1 &
  pids+=($!)
  cohort_names+=("$COHORT")
done

echo ""
echo "All 5 cohorts launched. Waiting for completion..."
echo ""

all_ok=true
for i in "${!pids[@]}"; do
  COHORT="${cohort_names[$i]}"
  PID="${pids[$i]}"
  if wait "$PID"; then
    echo "  [OK]   $COHORT (pid $PID)"
  else
    echo "  [FAIL] $COHORT (pid $PID) — check ${LOGROOT}/${COHORT}_launcher.log"
    all_ok=false
  fi
done

echo ""
if $all_ok; then
  echo "All cohorts completed successfully."
  echo ""
  echo "Next steps:"
  echo "  for COHORT in Hugo Riaz Rose IMMotion150 IMVigor210; do"
  echo "    python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --cohort \$COHORT"
  echo "  done"
  echo "  python -m data_preprocessing.immunotherapy_cohorts.get_ICI_embeddings --merge"
  echo "  python data_preprocessing/get_ICI_data_splits.py"
else
  echo "One or more cohorts failed. Check logs in ${LOGROOT}/"
  exit 1
fi
