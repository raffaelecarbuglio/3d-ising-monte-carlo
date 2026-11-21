#!/usr/bin/env bash
# Prepare a campaign once and submit one Slurm array per lattice size.
set -euo pipefail

fail() { echo "ERROR: $*" >&2; exit 1; }
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
retry=false
if [[ ${1:-} == --retry ]]; then
    retry=true
    shift
fi
(( $# == 1 )) || fail "Usage: bash submit_campaign.sh campaign.conf | --retry batch_directory"
argument=$(realpath -- "$1")
cd "$ROOT"

if $retry; then
    BATCH=$argument
    [[ -d "$BATCH/slurm/status" && -x "$BATCH/ising" ]] || fail "Not a prepared Slurm campaign: $BATCH"
    CONF=$BATCH/campaign.conf
else
    [[ -x "$ROOT/ising" ]] || fail "Build ising on a compute node with make ising first."
    CONF=$argument
fi
command -v sbatch > /dev/null || fail "Run this on a Slurm login node."

BETA_GRID=()
source "$CONF"
partitions=${SLURM_PARTITIONS:-levante,mefisto,zefiro}
default_time=${SLURM_TIME:-1-00:00:00}
array_limit=${SLURM_MAX_RUNNING:-}
[[ -z "$array_limit" || "$array_limit" =~ ^[1-9][0-9]*$ ]] || fail "SLURM_MAX_RUNNING must be a positive integer."

declare -A times=()
for setting in ${SLURM_TIME_BY_L:-}; do
    size=${setting%%:*}
    [[ "$size" =~ ^[1-9][0-9]*$ && "$setting" == "$size:"?* ]] ||
        fail 'Use SLURM_TIME_BY_L="48:12:00:00 64:18:00:00".'
    times[$size]=${setting#*:}
done

if ! $retry; then
    BATCH=$(bash "$ROOT/simulate.sh" --prepare-only "$CONF")
    echo "Batch directory: $BATCH"
    cp -- "$ROOT/ising" "$BATCH/ising"
    mkdir -p "$BATCH/slurm/status"
    awk -F, -v directory="$BATCH/slurm" 'NR > 1 { print $1 >> (directory "/L" $2 ".txt") }' "$BATCH/runs.csv"
fi

# With the lock held, only one launcher can submit or retry this folder.
exec 9>"$BATCH/slurm/submit.lock"
flock -x 9
job_prefix=${BATCH##*/}
task_lists=("$BATCH"/slurm/L*.txt)
[[ -f "${task_lists[0]}" ]] || fail "No task lists in $BATCH/slurm."
if $retry; then
    job_names=()
    for list in "${task_lists[@]}"; do
        file=${list##*/}
        job_names+=("$job_prefix-${file%.txt}")
    done
    names=$(IFS=,; printf '%s' "${job_names[*]}")
    active=$(squeue --noheader --user "$USER" --name="$names" --format='%i')
    [[ -z "$active" ]] || fail "Campaign jobs are still in the queue. Wait for them to finish before retrying."
    echo "Batch directory: $BATCH"
fi

for list in "${task_lists[@]}"; do
    file=${list##*/}
    size=${file#L}
    size=${size%.txt}
    index=0
    pending=()
    while IFS= read -r name; do
        status_file="$BATCH/slurm/status/$name.exit"
        if [[ ! -f "$status_file" || $(<"$status_file") != 0 ]]; then
            pending+=("$index")
        fi
        index=$((index + 1))
    done < "$list"
    if (( ${#pending[@]} == 0 )); then
        continue
    fi
    if (( ${#pending[@]} == index )); then
        indices="0-$((index - 1))"
    else
        indices=$(IFS=,; printf '%s' "${pending[*]}")
    fi
    if [[ -n "$array_limit" ]]; then
        indices+="%$array_limit"
    fi

    job=$(sbatch --parsable --partition="$partitions" --nodes=1 --ntasks=1 \
        --cpus-per-task=1 --hint=nomultithread --mem-per-cpu=1G \
        --time="${times[$size]:-$default_time}" --array="$indices" \
        --job-name="$job_prefix-L$size" --chdir="$BATCH" \
        --output="$BATCH/slurm/L$size-%A_%a.log" "$ROOT/cluster/campaign_task.sh" "$list") ||
        fail "Submission failed for L=$size. The campaign is saved in $BATCH."
    printf '%s\t%s\n' "$size" "$job" >> "$BATCH/slurm/jobs.tsv"
    printf 'L=%s: submitted %d task(s) as job %s\n' "$size" "${#pending[@]}" "$job"
done

echo "After the arrays finish, analyse with: bash analyse.sh $BATCH"
