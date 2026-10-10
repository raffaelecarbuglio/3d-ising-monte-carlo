#!/bin/bash
set -euo pipefail
export LC_ALL=C

if (( $# < 3 )); then
    echo "Usage: $0 NEW_RUN_DIR L BETA [BETA ...]" >&2; exit 2
fi
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
run_dir=$1
L=$2
shift 2
REPLICAS=${REPLICAS:-2}
N_THERM=${N_THERM:-50000}
N_SWEEPS=${N_SWEEPS:-1000000}
MEASURE_EVERY=${MEASURE_EVERY:-10}
SAVE_EVERY=${SAVE_EVERY:-10000}
SIGMA=${SIGMA:-0}
ALGORITHM=${ALGORITHM:-metropolis}
BASE_SEED=${BASE_SEED:-200000}

for value in "$L" "$REPLICAS" "$N_SWEEPS" "$MEASURE_EVERY" "$SAVE_EVERY"; do
    [[ $value =~ ^[1-9][0-9]{0,8}$ ]] || { echo "Invalid positive integer: $value" >&2; exit 2; }
done
for value in "$N_THERM" "$BASE_SEED"; do
    [[ $value =~ ^(0|[1-9][0-9]{0,9})$ ]] || { echo "Invalid nonnegative integer: $value" >&2; exit 2; }
done
tasks=$((REPLICAS * $#))
(( L >= 2 && L <= 512 && tasks <= 32 && N_THERM <= 2147483647 && BASE_SEED + tasks <= 2147483647 )) || exit 2
[[ $ALGORITHM == metropolis || $ALGORITHM == wolff ]] || exit 2
for value in "$SIGMA" "$@"; do
    [[ $value =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] &&
        awk -v x="$value" 'BEGIN { exit !(x >= 0 && x < 1e100) }' || exit 2
done
[[ $ALGORITHM != wolff ]] || awk -v x="$SIGMA" 'BEGIN { exit (x != 0) }'
[[ -x $repo/ising ]] || { echo "Compile with make on a compute node first." >&2; exit 1; }
[[ ! -e $run_dir ]] || { echo "Directory exists; submit it again to continue." >&2; exit 1; }
[[ $run_dir != *[[:space:]]* ]] || exit 2
declare -A seen=()
for beta in "$@"; do
    [[ ! ${seen[$beta]+yes} ]] || { echo "Duplicate beta: $beta" >&2; exit 2; }
    seen[$beta]=1
done
mkdir -p "$run_dir"/{inputs/new,inputs/restart,results,logs}
git -C "$repo" rev-parse HEAD > "$run_dir/revision.txt"

id=0
for beta in "$@"; do
    for (( replica=0; replica<REPLICAS; replica++ )); do
        name="L${L}_${ALGORITHM}_s${SIGMA}_b${beta}_r${replica}"
        seed=$((BASE_SEED + id))
        for mode in new restart; do
            start=random; thermal=$N_THERM
            if [[ $mode == restart ]]; then start=restart; thermal=0; fi
            cat > "$run_dir/inputs/$mode/$name.in" <<EOF
L = $L
beta = $beta
n_therm = $thermal
n_sweeps = $N_SWEEPS
measure_every = $MEASURE_EVERY
seed = $seed
start = $start
algorithm = $ALGORITHM
config_file = results/${name}_config.dat
data_file = results/${name}_data.dat
sigma = $SIGMA
save_every = $SAVE_EVERY
EOF
        done
        id=$((id + 1))
    done
done
sed "s/@TASKS@/$tasks/" "$repo/cluster/job.batch" > "$run_dir/job.batch"
echo "Prepared $tasks independent simulations in $run_dir"
