#!/bin/bash
set -euo pipefail
if (( $# != 1 )); then echo "Usage: $0 RUN_DIR" >&2; exit 2; fi
repo=$(cd -- "$(dirname -- "$0")/.." && pwd)
[[ -x $repo/ising ]] || { echo "Compile with make first." >&2; exit 1; }
cd -- "$1"
exec sbatch --job-name="$(basename "$PWD")" --chdir="$PWD" job.batch "$repo/ising"
