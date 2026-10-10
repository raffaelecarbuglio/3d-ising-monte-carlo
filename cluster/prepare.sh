#!/bin/bash
set -euo pipefail
export LC_ALL=C

if (( $# < 3 )); then
    echo "Usage: $0 NEW_RUN_DIR L BETA [BETA ...]" >&2
    exit 2
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
START=${START:-random}
# Il tempo di avvio non e' un seed: processi simultanei devono essere distinti.
if [[ -z ${BASE_SEED:-} ]]; then
    random_seed=$(od -An -N4 -tu4 /dev/urandom)
    BASE_SEED=$((random_seed % 2000000000))
fi
for value in "$L" "$REPLICAS" "$N_SWEEPS" "$MEASURE_EVERY" "$SAVE_EVERY"; do
    [[ $value =~ ^[1-9][0-9]*$ && ${#value} -le 9 ]] || {
        echo "L, replicas, sweeps and intervals must be positive integers." >&2; exit 2;
    }
done
for value in "$N_THERM" "$BASE_SEED"; do
    [[ $value =~ ^(0|[1-9][0-9]*)$ && ${#value} -le 10 ]] || {
        echo "N_THERM and BASE_SEED must be nonnegative integers." >&2; exit 2;
    }
done
tasks=$((REPLICAS * $#))
if (( L < 2 || L > 512 || tasks > 32 || N_THERM > 2147483647 || BASE_SEED + tasks > 2147483647 )); then
    echo "Use 2 <= L <= 512, at most 32 simulations, and seeds/counts within INT_MAX." >&2
    exit 2
fi
[[ $ALGORITHM == metropolis || $ALGORITHM == wolff ]] || exit 2
[[ $START == random || $START == ordered ]] || exit 2
for value in "$SIGMA" "$@"; do
    [[ $value =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] &&
        awk -v x="$value" 'BEGIN { exit !(x >= 0 && x < 1e100) }' || {
        echo "Sigma and beta must be finite, nonnegative numbers." >&2; exit 2;
    }
done
declare -A seen_betas=()
for beta in "$@"; do
    [[ ! ${seen_betas[$beta]+yes} ]] || { echo "Duplicate beta: $beta" >&2; exit 2; }
    seen_betas[$beta]=1
done
if [[ $ALGORITHM == wolff ]] && ! awk -v x="$SIGMA" 'BEGIN { exit (x != 0) }'; then
    echo "Wolff requires SIGMA=0." >&2; exit 2
fi
[[ -x $repo/ising ]] || { echo "Compile with make on a compute node first." >&2; exit 1; }
[[ ! -e $run_dir ]] || { echo "Run directory already exists; use submit.sh to continue." >&2; exit 1; }
# Il parser C legge percorsi senza spazi; anche le direttive Slurm devono essere semplici.
[[ $run_dir != *[[:space:]]* ]] || { echo "Run directory must not contain whitespace." >&2; exit 2; }
mkdir -p -- "$(dirname -- "$run_dir")"
mkdir -- "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
mkdir -p "$run_dir"/{inputs/new,inputs/restart,results,logs}
cp "$repo/ising" "$run_dir/ising"

{
    printf 'created_utc=%s\n' "$(date -u +%FT%TZ)"
    printf 'git_commit=%s\n' "$(git -C "$repo" rev-parse HEAD)"
    printf 'L=%s algorithm=%s sigma=%s replicas=%s base_seed=%s\n' \
        "$L" "$ALGORITHM" "$SIGMA" "$REPLICAS" "$BASE_SEED"
    sha256sum "$run_dir/ising"
} > "$run_dir/provenance.txt"
git -C "$repo" diff HEAD -- > "$run_dir/code.patch"

id=0
for beta in "$@"; do
    for (( replica=0; replica<REPLICAS; replica++ )); do
        name="L${L}_${ALGORITHM}_s${SIGMA}_b${beta}_r${replica}"
        seed=$((BASE_SEED + id))
        input="$run_dir/inputs/new/$name.in"
        sed -e "s/@L@/$L/g" -e "s/@BETA@/$beta/g" \
            -e "s/@THERM@/$N_THERM/g" -e "s/@SWEEPS@/$N_SWEEPS/g" \
            -e "s/@MEASURE@/$MEASURE_EVERY/g" -e "s/@SAVE@/$SAVE_EVERY/g" \
            -e "s/@SEED@/$seed/g" -e "s/@START@/$START/g" \
            -e "s/@ALGORITHM@/$ALGORITHM/g" -e "s/@SIGMA@/$SIGMA/g" \
            -e "s/@NAME@/$name/g" "$repo/cluster/template_input.dat" > "$input"
        sed -e 's/^start = .*/start = restart/' -e 's/^n_therm = .*/n_therm = 0/' \
            "$input" > "$run_dir/inputs/restart/$name.in"
        id=$((id + 1))
    done
done
cp "$repo/cluster/job.batch" "$run_dir/job.batch"
printf 'Prepared %d independent simulations in %s\n' "$tasks" "$run_dir"
