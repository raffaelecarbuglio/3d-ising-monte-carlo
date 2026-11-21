#!/usr/bin/env bash
# Slurm starts each task in the shared campaign directory.
set -euo pipefail

list=${1:?Expected an array task list}
line=$((${SLURM_ARRAY_TASK_ID:?Expected a Slurm array index} + 1))
name=$(sed -n "${line}{p;q;}" "$list")
[[ -n "$name" ]] || { echo "No replica at line $line of $list" >&2; exit 1; }
status="slurm/status/$name.exit"

# A duplicate submission must not overwrite a completed or running replica.
exec 9>"slurm/status/$name.lock"
flock -x 9
if [[ -f "$status" && $(<"$status") == 0 ]]; then
    exit 0
fi

# A retry starts from the original input and keeps the previous attempt's files.
archive=
for file in "results/${name}_data.dat" "results/${name}_config.dat" "results/$name.log"; do
    if [[ -e "$file" ]]; then
        if [[ -z "$archive" ]]; then
            mkdir -p slurm/attempts
            archive=$(mktemp -d "slurm/attempts/$name.XXXXXX")
        fi
        mv -- "$file" "$archive/"
    fi
done

exit_code=0
./ising "inputs/$name.in" > "results/$name.log" 2>&1 || exit_code=$?
printf '%s\n' "$exit_code" > "$status.tmp"
mv -- "$status.tmp" "$status"
exit "$exit_code"
