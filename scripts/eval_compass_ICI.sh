#!/bin/bash
# eval_compass_ICI.sh
#
# Evaluate all completed COMPASS ICI runs and write all_metrics.csv per tag.
#
# Usage:
#   bash scripts/eval_compass_ICI.sh [--paradigm loco|within_cohort|transfer]

set -e

CONDA_ENV="compass"
REPO="/data/rbg/users/ashyamal/RNA-FM-benchmarking"
PYTHON="conda run -n ${CONDA_ENV} python"

FILTER_PARADIGM=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --paradigm) FILTER_PARADIGM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

declare -A PARADIGM_TAGS

PARADIGM_TAGS["loco"]="held_out_Hugo held_out_IMMotion150 held_out_IMVigor210 held_out_Riaz held_out_Rose"

PARADIGM_TAGS["within_cohort"]="Hugo IMMotion150 IMVigor210 Riaz Rose"

PARADIGM_TAGS["transfer"]="Hugo_to_IMMotion150 Hugo_to_IMVigor210 Hugo_to_Riaz Hugo_to_Rose \
    IMMotion150_to_Hugo IMMotion150_to_IMVigor210 IMMotion150_to_Riaz IMMotion150_to_Rose \
    IMVigor210_to_Hugo IMVigor210_to_IMMotion150 IMVigor210_to_Riaz IMVigor210_to_Rose \
    Riaz_to_Hugo Riaz_to_IMMotion150 Riaz_to_IMVigor210 Riaz_to_Rose \
    Rose_to_Hugo Rose_to_IMMotion150 Rose_to_IMVigor210 Rose_to_Riaz"

cd "${REPO}"

for PARADIGM in loco within_cohort transfer; do
    [[ -n "$FILTER_PARADIGM" && "$PARADIGM" != "$FILTER_PARADIGM" ]] && continue

    for COHORT_TAG in ${PARADIGM_TAGS[$PARADIGM]}; do
        echo "==> Evaluating paradigm=${PARADIGM}  tag=${COHORT_TAG}"
        ${PYTHON} -m test.eval_compass_ICI \
            --paradigm   "${PARADIGM}" \
            --cohort_tag "${COHORT_TAG}"
    done
done

echo "All COMPASS evaluations complete."
