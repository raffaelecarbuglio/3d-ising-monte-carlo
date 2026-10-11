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
[[ $REPLICAS =~ ^[1-9][0-9]?$ ]] || exit 2
tasks=$((REPLICAS * $#))
(( tasks <= 32 )) || { echo "Maximum: 32 simulations per job." >&2; exit 2; }
[[ ! -e $run_dir ]] || { echo "Directory already exists." >&2; exit 1; }
mkdir -p "$run_dir"/{inputs,results,logs}

id=0
for beta in "$@"; do
    for (( replica=0; replica<REPLICAS; replica++ )); do
        name="chain${id}"
        cat > "$run_dir/inputs/$name.in" <<EOF
L = $L
beta = $beta
n_therm = ${N_THERM:-50000}
n_sweeps = ${N_SWEEPS:-1000000}
measure_every = ${MEASURE_EVERY:-10}
seed = $((${BASE_SEED:-200000} + id))
start = random
algorithm = ${ALGORITHM:-metropolis}
config_file = results/${name}_config.dat
data_file = results/${name}_data.dat
sigma = ${SIGMA:-0}
EOF
        id=$((id + 1))
    done
done
sed "s/@TASKS@/$tasks/" "$repo/cluster/job.batch" > "$run_dir/job.batch"
echo "Prepared $tasks simulations in $run_dir"
