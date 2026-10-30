import re
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Check saved simulation seeds for reuse.")
parser.add_argument("directories", nargs="+")
parser.add_argument("--against", nargs="+", help="Check new inputs against history; ignore history/history duplicates.")
args = parser.parse_args()

def input_files(directories):
    files = set()
    for directory in directories:
        root = Path(directory)
        if not root.is_dir():
            sys.exit(f"Directory not found: {root}")
        files.update(path.resolve() for path in root.rglob("*.in"))
    return files


files = input_files(args.directories)
if not files:
    sys.exit("No .in input files found.")

seen = {}
repeated = False
history = input_files(args.against) - files if args.against else set()
for path in sorted(files) + sorted(history):
    seeds = re.findall(r"^\s*seed\s*=\s*([0-9]+)\s*$", path.read_text(), re.MULTILINE)
    if len(seeds) != 1:
        sys.exit(f"Expected one seed in {path}")
    seed = int(seeds[0])
    if seed in seen:
        print(f"Repeated seed {seed}:\n  {seen[seed]}\n  {path}")
        repeated = True
    elif path in files:
        seen[seed] = path

print(f"Checked {len(files)} inputs against {len(history)} historical inputs: " +
      ("repeated seeds found." if repeated else "all seeds distinct."))
sys.exit(1 if repeated else 0)
