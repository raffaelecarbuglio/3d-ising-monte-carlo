import re
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit("Usage: python3 cluster/check_seeds.py RUN_DIR [RUN_DIR ...]")

files = set()
for directory in sys.argv[1:]:
    root = Path(directory)
    if not root.is_dir():
        sys.exit(f"Directory not found: {root}")
    files.update(path.resolve() for path in root.rglob("*.in"))
if not files:
    sys.exit("No .in input files found.")

seen = {}
repeated = False
for path in sorted(files):
    seeds = re.findall(r"^\s*seed\s*=\s*([0-9]+)\s*$", path.read_text(), re.MULTILINE)
    if len(seeds) != 1:
        sys.exit(f"Expected one seed in {path}")
    seed = int(seeds[0])
    if seed in seen:
        print(f"Repeated seed {seed}:\n  {seen[seed]}\n  {path}")
        repeated = True
    else:
        seen[seed] = path

print(f"Checked {len(files)} inputs: " + ("repeated seeds found." if repeated else "all seeds distinct."))
sys.exit(1 if repeated else 0)
