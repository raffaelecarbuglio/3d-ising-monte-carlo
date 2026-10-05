#!/bin/bash

set -Eeuo pipefail
export LC_ALL=C

# Indipendente da MAX_JOBS delle simulazioni; 8 analisi per la macchina da 125 GiB.
ANALYSIS_JOBS="${ANALYSIS_JOBS:-8}"
if [[ ! "$ANALYSIS_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ANALYSIS_JOBS must be a positive integer." >&2
    exit 2
fi
# Ogni processo NumPy usa un solo thread.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Usage:
#
#   ./analyze_grid.sh RUN_DIR L:BLOCK_SIZE [L:BLOCK_SIZE ...]
#
# Examples:
#
#   ./analyze_grid.sh beta_runs_20260915_120000 8:32 16:32 24:64 32:64
#   ./analyze_grid.sh beta_runs_20260917_120000 48:64 64:64
#
# Works for both Wolff and Metropolis batches.

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RUN_DIR L:BLOCK_SIZE [L:BLOCK_SIZE ...]" >&2
    exit 2
fi

RUN_DIR="$1"
shift

RESULTS_DIR="$RUN_DIR/results"

SUMMARY_FILE="$RUN_DIR/scaling_summary.txt"
PLOT_FILE="$RUN_DIR/u_vs_rxi.png"
LOG_FILE="$RUN_DIR/analysis.log"


# ============================================================
# CHECKS
# ============================================================

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: directory '$RUN_DIR' does not exist." >&2
    exit 1
fi

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "ERROR: results directory '$RESULTS_DIR' does not exist." >&2
    exit 1
fi

if [[ ! -f "analyze.py" ]]; then
    echo "ERROR: analyze.py not found in the current directory." >&2
    exit 1
fi

if [[ ! -f "plot_u_vs_rxi.py" ]]; then
    echo "ERROR: plot_u_vs_rxi.py not found in the current directory." >&2
    exit 1
fi


# ============================================================
# BLOCK SIZES
# ============================================================

declare -A BLOCK

for specification in "$@"; do
    if [[ ! "$specification" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
        echo "ERROR: '$specification' must have the form L:BLOCK_SIZE with positive integers." >&2
        exit 1
    fi

    L="${BASH_REMATCH[1]}"
    block="${BASH_REMATCH[2]}"

    if [[ -n "${BLOCK[$L]+x}" ]]; then
        echo "ERROR: block size for L=$L was specified more than once." >&2
        exit 1
    fi

    BLOCK[$L]="$block"
done


# ============================================================
# PREPARE OUTPUT
# ============================================================

shopt -s nullglob

DATA_FILES=("$RESULTS_DIR"/L*_data.dat "$RESULTS_DIR"/L*_data.dat.gz)

shopt -u nullglob

if [[ ${#DATA_FILES[@]} -eq 0 ]]; then
    echo "ERROR: no data files found in '$RESULTS_DIR'." >&2
    exit 1
fi

echo "Found ${#DATA_FILES[@]} data files."
echo "Analysis directory: $RUN_DIR"
echo


# ============================================================
# ANALYSIS
# ============================================================

# Controlla tutti i file prima di avviare processi o cancellare risultati.
declare -A SEEN_DATA
for f in "${DATA_FILES[@]}"; do
    name=$(basename "$f")
    if [[ ! "$name" =~ ^L([0-9]+)_ ]]; then
        echo "ERROR: cannot determine L from '$name'." >&2
        exit 1
    fi
    L="${BASH_REMATCH[1]}"
    if [[ -z "${BLOCK[$L]+x}" ]]; then
        echo "ERROR: no block size defined for L=$L." >&2
        exit 1
    fi
    key="${f%.gz}"
    if [[ -n "${SEEN_DATA[$key]+x}" ]]; then
        echo "ERROR: both .dat and .dat.gz exist for '$key'." >&2
        exit 1
    fi
    SEEN_DATA[$key]=1
done

TMP_ANALYSIS=$(mktemp -d)
pids=()
indices=()
cleanup() {
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "$TMP_ANALYSIS"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

rm -f "$SUMMARY_FILE" "$LOG_FILE"
rm -f "$RESULTS_DIR"/*_blocks.txt
printf '# L beta Rxi err_Rxi U err_U\n' > "$SUMMARY_FILE"
successes=0

# Merge seriale: nessun processo scrive nei file comuni.
wait_batch() {
    local failed=0 i index
    for i in "${!pids[@]}"; do
        index="${indices[$i]}"
        if ! wait "${pids[$i]}"; then
            echo "ERROR: analysis failed for '${DATA_FILES[$index]}'. See: $LOG_FILE" >&2
            failed=1
        else
            awk '!/^#/' "$TMP_ANALYSIS/$index.summary" >> "$SUMMARY_FILE"
            successes=$((successes + 1))
        fi
        cat "$TMP_ANALYSIS/$index.log" >> "$LOG_FILE"
    done
    pids=()
    indices=()
    return "$failed"
}

echo "Concurrent analyses: $ANALYSIS_JOBS (one thread each)"
for index in "${!DATA_FILES[@]}"; do
    f="${DATA_FILES[$index]}"
    name=$(basename "$f")
    [[ "$name" =~ ^L([0-9]+)_ ]]
    L="${BASH_REMATCH[1]}"
    block="${BLOCK[$L]}"
    echo "Analyzing $name   L=$L   block_size=$block"
    printf '\n%s\nblock_size = %s\n' "$name" "$block" > "$TMP_ANALYSIS/$index.log"
    python3 analyze.py "$f" \
        --block-size "$block" \
        --summary-output "$TMP_ANALYSIS/$index.summary" \
        >> "$TMP_ANALYSIS/$index.log" 2>&1 &
    pids+=("$!")
    indices+=("$index")
    if [[ ${#pids[@]} -ge "$ANALYSIS_JOBS" ]]; then
        wait_batch
    fi
done
wait_batch

# ============================================================
# CHECK SUMMARY
# ============================================================

summary_points=$(grep -vc '^#' "$SUMMARY_FILE")

if [[ "$summary_points" -ne "$successes" ]]; then
    echo "ERROR: summary contains $summary_points points, expected $successes." >&2
    exit 1
fi

duplicates=$(
    awk '!/^#/ {print $1, $2}' "$SUMMARY_FILE" |
    sort |
    uniq -d
)

if [[ -n "$duplicates" ]]; then
    echo "ERROR: duplicate (L, beta) points found:" >&2
    echo "$duplicates" >&2
    exit 1
fi


# ============================================================
# SCALING PLOT
# ============================================================

echo
echo "Creating U(R_xi) scaling plot..."

if ! MPLBACKEND=Agg python3 plot_u_vs_rxi.py \
    "$SUMMARY_FILE" \
    --output "$PLOT_FILE" \
    >> "$LOG_FILE" 2>&1
then
    echo "ERROR: plot creation failed." >&2
    echo "See: $LOG_FILE" >&2
    exit 1
fi


# ============================================================
# FINAL REPORT
# ============================================================

echo
echo "========================================"
echo "Analysis completed successfully"
echo "========================================"
echo
echo "Analyzed files: $successes"
echo
echo "Summary:"
echo "  $SUMMARY_FILE"
echo
echo "Scaling plot:"
echo "  $PLOT_FILE"
echo
echo "Block averages:"
echo "  $RESULTS_DIR/*_blocks.txt"
echo
echo "Full analysis log:"
echo "  $LOG_FILE"
