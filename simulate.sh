#!/usr/bin/env bash
# Run a campaign from the project directory; paths in the config are relative to it.
set -Eeuo pipefail
export LC_ALL=C

CONF=$(realpath "${1:?Usage: bash simulate.sh campaign.conf}")
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$ROOT"
BETA_GRID=()
source "$CONF"

fail() { echo "ERROR: $*" >&2; exit 1; }
for field in N_THERM N_SWEEPS MEASURE_EVERY BASE_SEED MAX_JOBS; do
    value=${!field:-}
    [[ "$value" =~ ^(0|[1-9][0-9]{0,9})$ ]] || fail "$field must be a nonnegative C integer."
    (( value <= 2147483647 )) || fail "$field is too large."
    if [[ "$field" != N_THERM && "$field" != BASE_SEED ]]; then
        (( value > 0 )) || fail "$field must be positive."
    fi
done
[[ "${ALGORITHM:-}" == metropolis || "${ALGORITHM:-}" == wolff ]] || fail "Choose ALGORITHM=metropolis or wolff."
SIGMA=${SIGMA:-0}
LAMBDA=${LAMBDA:-0}
awk -v s="$SIGMA" -v l="$LAMBDA" -v a="$ALGORITHM" 'BEGIN {
    number="^[0-9]+([.][0-9]*)?([eE][+-]?[0-9]+)?$|^[.][0-9]+([eE][+-]?[0-9]+)?$";
    exit !(s ~ number && l ~ number && s+0 < 1e308 && l+0 < 1e307 &&
           !(s>0 && l>0) && !(a=="wolff" && (s>0 || l>0)));
}' || fail "Invalid SIGMA/LAMBDA: use one finite nonnegative perturbation; Wolff requires both zero."
[[ ${#BETA_GRID[@]} -gt 0 ]] || fail "BETA_GRID is empty."

OUTPUT_DIR=${OUTPUT_DIR:-runs}
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
[[ ! "$OUTPUT_DIR" =~ [[:space:],] ]] || fail "Output paths cannot contain whitespace or commas (input/CSV format)."
name=$(basename "$CONF" .conf)
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || fail "Use letters, digits, underscores or hyphens for the campaign filename."
BATCH=$(mktemp -d "$OUTPUT_DIR/${name}_$(date +%Y%m%d_%H%M%S)_XXXXXX")
# Failed preparation leaves no unused .in files that would reserve seeds.
trap 'rm -rf -- "$BATCH"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir "$BATCH/inputs" "$BATCH/results"
cp "$CONF" "$BATCH/campaign.conf"
printf 'name,L,beta,seed,n_therm,n_sweeps,measure_every,input_file,data_file,config_file,log_file,status,exit_code\n' > "$BATCH/runs.csv"
echo "Batch directory: $BATCH"

# Prepare every input before starting any simulation.
run_id=0
declare -A seen
for row in "${BETA_GRID[@]}"; do
    read -r -a values <<< "$row"
    L=${values[0]:-}
    # L^3 must fit the C simulator's signed integer site count.
    [[ "$L" =~ ^[1-9][0-9]{0,3}$ ]] && (( L >= 2 && L <= 1290 )) || fail "Invalid lattice size: $L."
    (( ${#values[@]} >= 2 )) || fail "Missing beta values for L=$L."
    for beta in "${values[@]:1}"; do
        awk -v b="$beta" 'BEGIN { exit !(b ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ && b+0 < 1e308) }' || fail "Invalid beta: $beta."
        seed=$((BASE_SEED + run_id))
        (( seed <= 2147483647 )) || fail "Seed range exceeds the C integer limit."
        beta_name=$(printf '%.9f' "$beta" | tr '.' p)
        name="L${L}_b${beta_name}"
        [[ -z ${seen[$name]+x} ]] || fail "Repeated point or filename collision: $name."
        seen[$name]=1
        input="$BATCH/inputs/$name.in"
        data="$BATCH/results/${name}_data.dat"
        config="$BATCH/results/${name}_config.dat"
        log="$BATCH/results/$name.log"
        (( ${#config} <= 511 && ${#data} <= 511 )) || fail "Output path exceeds the C input format's 511-character limit."
        cat > "$input" <<EOF
L = $L
beta = $beta
n_therm = $N_THERM
n_sweeps = $N_SWEEPS
measure_every = $MEASURE_EVERY
seed = $seed
start = random
algorithm = $ALGORITHM
config_file = $config
data_file = $data
sigma = $SIGMA
lambda = $LAMBDA
EOF
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,PENDING,\n' "$name" "$L" "$beta" "$seed" "$N_THERM" "$N_SWEEPS" "$MEASURE_EVERY" "$input" "$data" "$config" "$log" >> "$BATCH/runs.csv"
        run_id=$((run_id + 1))
    done
done
SEED_HISTORY=("${SEED_HISTORY[@]:-$ROOT}")
python3 cluster/check_seeds.py "$BATCH/inputs" --against "${SEED_HISTORY[@]}"
make ising
echo "Prepared $run_id simulations; at most $MAX_JOBS run together."

pids=()
names=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_batch() {
    local failed=0 i code status
    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then code=0; status=OK;
        else code=$?; status=FAILED; failed=1; fi
        echo "$status ${names[$i]} (exit $code)"
        awk -F, -v OFS=, -v n="${names[$i]}" -v s="$status" -v c="$code" \
            '$1==n {$12=s; $13=c} {print}' "$BATCH/runs.csv" > "$BATCH/runs.csv.tmp"
        mv "$BATCH/runs.csv.tmp" "$BATCH/runs.csv"
    done
    pids=(); names=()
    return "$failed"
}

# Larger lattices go first; seeds retain their order in BETA_GRID.
while IFS=, read -r name L beta seed therm sweeps every input data config log status code; do
    echo "START $name"
    ./ising "$input" > "$log" 2>&1 &
    pids+=("$!"); names+=("$name")
    if (( ${#pids[@]} >= MAX_JOBS )); then wait_batch; fi
done < <(tail -n +2 "$BATCH/runs.csv" | sort -t, -k2,2nr)
wait_batch
echo "All $run_id simulations completed successfully."
echo "Inspect blocking plateaus, then set BLOCK_SIZES in $BATCH/campaign.conf."
echo "Analyse with: bash analyse.sh $BATCH"
