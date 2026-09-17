#!/bin/bash

set -Eeuo pipefail
export LC_ALL=C

# Usage:
#   ./run_and_analyze.sh ALGORITHM TEMPLATE L:BLOCK_SIZE [L:BLOCK_SIZE ...]
#
# Examples:
#   ./run_and_analyze.sh wolff input_wolff.dat 8:32 16:32 24:64 32:64
#   ./run_and_analyze.sh metropolis input_metropolis.dat 48:256 64:512
#
# To use a different batch script, set RUN_SCRIPT. For example:
#   RUN_SCRIPT=./run_wolff_large.sh \
#       ./run_and_analyze.sh wolff input_wolff.dat 48:64 64:64

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 ALGORITHM TEMPLATE L:BLOCK_SIZE [L:BLOCK_SIZE ...]" >&2
    exit 2
fi

ALGORITHM="$1"
TEMPLATE="$2"
BLOCK_SPECS=("${@:3}")

case "$ALGORITHM" in
    wolff)
        DEFAULT_RUN_SCRIPT="./run_wolff.sh"
        ;;
    metropolis)
        DEFAULT_RUN_SCRIPT="./run_metropolis.sh"
        ;;
    *)
        echo "ERROR: ALGORITHM must be 'wolff' or 'metropolis'." >&2
        exit 2
        ;;
esac

RUN_SCRIPT="${RUN_SCRIPT:-$DEFAULT_RUN_SCRIPT}"

if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "ERROR: '$RUN_SCRIPT' does not exist or is not executable." >&2
    exit 1
fi

if [[ ! -x "./analyze_grid.sh" ]]; then
    echo "ERROR: './analyze_grid.sh' does not exist or is not executable." >&2
    exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template '$TEMPLATE' does not exist." >&2
    exit 1
fi

declare -A SEEN_L

for specification in "${BLOCK_SPECS[@]}"; do
    if [[ ! "$specification" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
        echo "ERROR: '$specification' must have the form L:BLOCK_SIZE with positive integers." >&2
        exit 1
    fi

    L="${BASH_REMATCH[1]}"

    if [[ -n "${SEEN_L[$L]+x}" ]]; then
        echo "ERROR: block size for L=$L was specified more than once." >&2
        exit 1
    fi

    SEEN_L[$L]=1
done

TMP_LOG=$(mktemp)
trap 'rm -f "$TMP_LOG"' EXIT

echo "Starting $ALGORITHM simulations..."
echo

if ! "$RUN_SCRIPT" "$TEMPLATE" 2>&1 | tee "$TMP_LOG"; then
    echo >&2
    echo "ERROR: simulation batch failed; analysis will not be started." >&2
    exit 1
fi

RUN_DIR=$(
    sed -n 's/^Batch directory: //p' "$TMP_LOG" |
    tail -n 1
)

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: could not determine the batch directory." >&2
    exit 1
fi

if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: batch directory '$RUN_DIR' does not exist." >&2
    exit 1
fi

echo
echo "Simulations completed successfully."
echo "Starting analysis of: $RUN_DIR"
echo

./analyze_grid.sh "$RUN_DIR" "${BLOCK_SPECS[@]}"

echo
echo "========================================"
echo "Simulation and analysis completed"
echo "========================================"
echo "Batch directory: $RUN_DIR"
echo "Scaling plot:   $RUN_DIR/u_vs_rxi.png"
