#!/usr/bin/env bash
# =============================================================================
# scripts/run_eval_ICI.sh
#
# Evaluate all ICI model outputs across all paradigms and cohort tags.
# Reads cohort names from configs/immunotherapy_cohorts.yaml.
#
# Produces one CSV per (paradigm, cohort_tag):
#   model_outputs/ICI/{paradigm}/{cohort_tag}/all_metrics.csv
#
# Usage:
#   ./scripts/run_eval_ICI.sh [GPU]
#   GPU defaults to 0.
#
# To run a single paradigm only, set PARADIGMS env var:
#   PARADIGMS="loco" ./scripts/run_eval_ICI.sh 1
# =============================================================================
set -euo pipefail

GPU="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ICI_CONFIG="${REPO_ROOT}/configs/immunotherapy_cohorts.yaml"
LOG_DIR="${REPO_ROOT}/logs/eval/ICI"
mkdir -p "$LOG_DIR"

# Read cohort names
COHORTS=($(python -c "
import yaml
with open('${ICI_CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(' '.join(cfg['cohorts'].keys()))
"))

echo "Cohorts: ${COHORTS[*]}"
echo "GPU: $GPU"

PARADIGMS_TO_RUN="${PARADIGMS:-within_cohort loco transfer}"

eval_tag() {
  local PARADIGM="$1"
  local COHORT_TAG="$2"
  local LOGFILE="${LOG_DIR}/${PARADIGM}__${COHORT_TAG}.log"

  echo "  Evaluating: paradigm=$PARADIGM  tag=$COHORT_TAG"
  python -m test.eval_ICI_models \
    --paradigm   "$PARADIGM" \
    --cohort_tag "$COHORT_TAG" \
    > "$LOGFILE" 2>&1 &
}

# Launch all eval jobs in background, then wait
PIDS=()

for PARADIGM in $PARADIGMS_TO_RUN; do
  case "$PARADIGM" in

    within_cohort)
      for COHORT in "${COHORTS[@]}"; do
        eval_tag "within_cohort" "within_${COHORT}"
      done
      ;;

    loco)
      for HELD_OUT in "${COHORTS[@]}"; do
        eval_tag "loco" "loco_held_out_${HELD_OUT}"
      done
      ;;

    transfer)
      for TRAIN_COHORT in "${COHORTS[@]}"; do
        for TEST_COHORT in "${COHORTS[@]}"; do
          [[ "$TRAIN_COHORT" == "$TEST_COHORT" ]] && continue
          eval_tag "transfer" "transfer_${TRAIN_COHORT}_to_${TEST_COHORT}"
        done
      done
      ;;

    *)
      echo "Unknown paradigm: $PARADIGM (skipping)"
      ;;
  esac
done

wait
echo ""
echo "All eval jobs complete. CSVs written to model_outputs/ICI/"
