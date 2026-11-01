#!/usr/bin/env bash
set -Eeuo pipefail

BATCH=$(realpath "${1:?Usage: bash analyse.sh batch_directory}")
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
source "$BATCH/campaign.conf"

if [[ -z ${BLOCK_SIZES:-} ]]; then
    echo "Set BLOCK_SIZES in $BATCH/campaign.conf after inspecting the blocking plateaus." >&2
    exit 1
fi
# Refuse incomplete campaigns before analyze_grid.sh replaces analysis outputs.
if [[ ! -f "$BATCH/runs.csv" ]] || ! awk -F, 'NR>1 {n++; if ($12!="OK") bad=1} END {exit (!n || bad)}' "$BATCH/runs.csv"; then
    echo "ERROR: runs.csv must report OK for every simulation." >&2
    exit 1
fi
# Resolve saved filenames locally (the batch may have been copied from another PC).
while IFS=, read -r name L beta seed therm sweeps every input data rest; do
    data="$BATCH/results/$(basename "$data")"
    if [[ ! -f "$data" && ! -f "$data.gz" ]]; then
        echo "ERROR: missing measurements for $name." >&2
        exit 1
    fi
done < <(tail -n +2 "$BATCH/runs.csv")
read -r -a blocks <<< "$BLOCK_SIZES"
ANALYSIS_JOBS=${ANALYSIS_JOBS:-8} bash analyze_grid.sh "$BATCH" "${blocks[@]}"

# Leave REFERENCE empty to perform only the block analysis and U(Rxi) plot.
if [[ -n ${REFERENCE:-} ]]; then
    MPLBACKEND=Agg python3 compare_universality.py \
        --reference "$REFERENCE" --perturbed "$BATCH/results" \
        --output-prefix "$BATCH/${OUTPUT_PREFIX:-comparison}"
fi
