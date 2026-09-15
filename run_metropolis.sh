#!/bin/bash

set -Eeuo pipefail
export LC_ALL=C

# Usage:
#   ./run_beta_grid_metropolis.sh path/to/valid_metropolis_input.dat

EXECUTABLE="${EXECUTABLE:-./ising}"
TEMPLATE="${1:-}"

MAX_JOBS="${MAX_JOBS:-20}"

N_THERM="${N_THERM:-50000}"
N_SWEEPS="${N_SWEEPS:-1000000}"
MEASURE_EVERY="${MEASURE_EVERY:-10}"

BASE_SEED="${BASE_SEED:-200000}"

ranges=(
    "8   0.206195  0.229220  33"
    "16  0.216169  0.224209  29"
    "24  0.218617  0.222978  16"
    "32  0.219781  0.222502  15"
)

if [[ -z "$TEMPLATE" ]]; then
    echo "Usage: $0 path/to/valid_metropolis_input.dat" >&2
    exit 2
fi

if [[ ! -x "$EXECUTABLE" ]]; then
    echo "ERROR: executable '$EXECUTABLE' does not exist or is not executable." >&2
    exit 1
fi

if [[ ! -f "$TEMPLATE" ]]; then
    echo "ERROR: template '$TEMPLATE' does not exist." >&2
    exit 1
fi

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_JOBS must be a positive integer." >&2
    exit 1
fi

required_fields=(
    L
    beta
    n_therm
    n_sweeps
    measure_every
    seed
    start
    config_file
    data_file
)

for field in "${required_fields[@]}"; do
    count=$(grep -Ec "^${field}[[:space:]]*=" "$TEMPLATE" || true)

    if [[ "$count" -ne 1 ]]; then
        echo "ERROR: template must contain exactly one '$field = ...' line; found $count." >&2
        exit 1
    fi
done

algorithm_count=$(grep -Ec '^algorithm[[:space:]]*=' "$TEMPLATE" || true)

if [[ "$algorithm_count" -ne 1 ]]; then
    echo "ERROR: template must contain exactly one 'algorithm = ...' line." >&2
    exit 1
fi

if ! grep -Eq \
    '^algorithm[[:space:]]*=[[:space:]]*metropolis[[:space:]]*$' \
    "$TEMPLATE"
then
    echo "ERROR: template is not configured with 'algorithm = metropolis'." >&2
    exit 1
fi

RUN_DIR="$(pwd)/metropolis_runs_$(date +%Y%m%d_%H%M%S)"
INPUT_DIR="$RUN_DIR/inputs"
RESULT_DIR="$RUN_DIR/results"
CSV_FILE="$RUN_DIR/runs.csv"

mkdir -p "$INPUT_DIR" "$RESULT_DIR"

replace_field()
{
    local file="$1"
    local field="$2"
    local value="$3"
    local count

    count=$(grep -Ec "^${field}[[:space:]]*=" "$file" || true)

    if [[ "$count" -ne 1 ]]; then
        echo "ERROR: '$file' contains $count lines for '$field'; expected 1." >&2
        return 1
    fi

    sed -i -E \
        "s|^${field}[[:space:]]*=.*$|${field} = ${value}|" \
        "$file" || return 1

    if ! grep -Fqx "${field} = ${value}" "$file"; then
        echo "ERROR: failed to set '$field' in '$file'." >&2
        return 1
    fi
}

update_status()
{
    local name="$1"
    local status="$2"
    local exit_code="$3"
    local tmp="$CSV_FILE.tmp"

    awk \
        -F',' \
        -v OFS=',' \
        -v name="$name" \
        -v status="$status" \
        -v code="$exit_code" \
        '
        NR == 1 {
            print
            next
        }

        $1 == name {
            $12 = status
            $13 = code
        }

        {
            print
        }
        ' "$CSV_FILE" > "$tmp" || return 1

    mv -- "$tmp" "$CSV_FILE" || return 1
}

RUN_NAMES=()
INPUT_FILES=()
LOG_FILES=()

echo \
"name,L,beta,seed,n_therm,n_sweeps,measure_every,input_file,data_file,config_file,log_file,status,exit_code" \
> "$CSV_FILE"

run_id=0

for range in "${ranges[@]}"; do
    read -r L beta_min beta_max npoints <<< "$range"

    for ((i=0; i<npoints; i++)); do

        beta=$(
            awk \
                -v a="$beta_min" \
                -v b="$beta_max" \
                -v i="$i" \
                -v n="$npoints" \
                'BEGIN {
                    if (n == 1)
                        printf "%.12f", a;
                    else
                        printf "%.12f", a + i*(b-a)/(n-1);
                }'
        )

        seed=$((BASE_SEED + run_id))

        beta_name=$(printf "%.9f" "$beta" | tr '.' 'p')
        name="L${L}_b${beta_name}"

        input_file="$INPUT_DIR/${name}.in"
        data_file="$RESULT_DIR/${name}_data.dat"
        config_file="$RESULT_DIR/${name}_config.dat"
        log_file="$RESULT_DIR/${name}.log"

        cp -- "$TEMPLATE" "$input_file"

        replace_field "$input_file" L "$L"
        replace_field "$input_file" beta "$beta"
        replace_field "$input_file" n_therm "$N_THERM"
        replace_field "$input_file" n_sweeps "$N_SWEEPS"
        replace_field "$input_file" measure_every "$MEASURE_EVERY"
        replace_field "$input_file" seed "$seed"
        replace_field "$input_file" start "random"
        replace_field "$input_file" config_file "$config_file"
        replace_field "$input_file" data_file "$data_file"

        if ! grep -Eq \
            '^algorithm[[:space:]]*=[[:space:]]*metropolis[[:space:]]*$' \
            "$input_file"
        then
            echo "ERROR: '$input_file' is not configured for Metropolis." >&2
            exit 1
        fi

        echo \
"$name,$L,$beta,$seed,$N_THERM,$N_SWEEPS,$MEASURE_EVERY,$input_file,$data_file,$config_file,$log_file,PENDING," \
        >> "$CSV_FILE"

        RUN_NAMES+=("$name")
        INPUT_FILES+=("$input_file")
        LOG_FILES+=("$log_file")

        run_id=$((run_id + 1))
    done
done

TOTAL_RUNS=${#RUN_NAMES[@]}

if [[ "$TOTAL_RUNS" -ne 93 ]]; then
    echo "ERROR: expected 93 runs, generated $TOTAL_RUNS." >&2
    exit 1
fi

echo "Prepared $TOTAL_RUNS Metropolis input files."
echo "Batch directory: $RUN_DIR"
echo "Maximum simultaneous jobs: $MAX_JOBS"
echo

PIDS=()
PID_NAMES=()
PID_LOGS=()

for ((idx=0; idx<TOTAL_RUNS; idx++)); do

    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
        sleep 1
    done

    name="${RUN_NAMES[$idx]}"
    input_file="${INPUT_FILES[$idx]}"
    log_file="${LOG_FILES[$idx]}"

    echo "START  $name"

    "$EXECUTABLE" "$input_file" > "$log_file" 2>&1 &

    pid=$!

    PIDS+=("$pid")
    PID_NAMES+=("$name")
    PID_LOGS+=("$log_file")
done

successes=0
failures=0

FAILED_NAMES=()
FAILED_LOGS=()
FAILED_CODES=()

for ((idx=0; idx<TOTAL_RUNS; idx++)); do

    pid="${PIDS[$idx]}"
    name="${PID_NAMES[$idx]}"
    log_file="${PID_LOGS[$idx]}"

    if wait "$pid"; then
        rc=0
        status="OK"
        successes=$((successes + 1))
        echo "OK     $name"
    else
        rc=$?
        status="FAILED"
        failures=$((failures + 1))

        FAILED_NAMES+=("$name")
        FAILED_LOGS+=("$log_file")
        FAILED_CODES+=("$rc")

        echo "FAILED $name  exit_code=$rc  log=$log_file" >&2
    fi

    if ! update_status "$name" "$status" "$rc"; then
        echo "WARNING: could not update runs.csv for $name." >&2
    fi
done

echo
echo "========================================"
echo "Metropolis batch finished"
echo "========================================"
echo
echo "Successful runs: $successes"
echo "Failed runs:     $failures"
echo
echo "Summary: $CSV_FILE"
echo "Results: $RESULT_DIR"

if (( failures > 0 )); then
    echo
    echo "Failed simulations:"

    for ((idx=0; idx<failures; idx++)); do
        echo "  ${FAILED_NAMES[$idx]}  exit_code=${FAILED_CODES[$idx]}  log=${FAILED_LOGS[$idx]}"
    done

    exit 1
fi

echo
echo "All $TOTAL_RUNS Metropolis simulations completed successfully."

exit 0
