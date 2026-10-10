#!/bin/bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
    echo "Usage: $0 RUN_DIR [NUMBER_OF_JOBS]" >&2; exit 2
fi
cd -- "$1"
count=${2:-1}
[[ $count =~ ^[1-9][0-9]*$ ]] || exit 2
[[ -f job.batch && -x ising ]] || { echo "Not a prepared campaign." >&2; exit 1; }
shopt -s nullglob
inputs=(inputs/new/*.in)
(( ${#inputs[@]} > 0 )) || { echo "No inputs found." >&2; exit 1; }

dependency=()
for (( i=0; i<count; i++ )); do
    job=$(sbatch --parsable --job-name="$(basename "$PWD")" --chdir="$PWD" \
        --ntasks="${#inputs[@]}" "${dependency[@]}" job.batch)
    job=${job%%;*}
    echo "$job"
    dependency=(--dependency="afterok:$job")
done
