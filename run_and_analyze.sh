#!/bin/bash

set -Eeuo pipefail
export LC_ALL=C

# Usage:
#   ./run_and_analyze.sh ALGORITHM TEMPLATE B8 B16 B24 B32
#
# Examples:
#   ./run_and_analyze.sh wolff input_wolff.dat 32 32 64 64
#   ./run_and_analyze.sh metropolis input_metropolis.dat 128 256 256 512

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 ALGORITHM TEMPLATE B8 B16 B24 B32" >&2
    exit 2
fi

ALGORITHM="$1"
TEMPLATE="$2"
B8="$3"
B16="$4"
B24="$5"
B32="$6"

case "$ALGORITHM" in
    wolff)
        RUN_SCRIPT="./run_wolff.sh"
        ;;
    metropolis)
        RUN_SCRIPT="./run_metropolis.sh"
        ;;
    *)
        echo "ERROR: ALGORITHM must be 'wolff' or 'metropolis'." >&2
        exit 2
        ;;
esac

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

for block in "$B8" "$B16" "$B24" "$B32"; do
    if ! [[ "$block" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: block sizes must be positive integers." >&2
        exit 1
    fi
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

./analyze_grid.sh "$RUN_DIR" "$B8" "$B16" "$B24" "$B32"

echo
echo "========================================"
echo "Simulation and analysis completed"
echo "========================================"
echo "Batch directory: $RUN_DIR"
echo "Scaling plot:   $RUN_DIR/u_vs_rxi.png"
