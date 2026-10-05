import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class GridTests(unittest.TestCase):
    def test_parallel_matches_serial_and_limits_processes(self):
        with tempfile.TemporaryDirectory(prefix="analysis test ") as directory:
            root = Path(directory)
            project = Path(__file__).resolve().parent
            for name in ("analyze.py", "analyze_grid.sh", "plot_u_vs_rxi.py"):
                shutil.copy(project / name, root / name)
            results = root / "batch" / "results"
            results.mkdir(parents=True)
            for i in range(10):
                suffix = ".dat.gz" if i % 2 else ".dat"
                path = results / f"L4_beta{i}_data{suffix}"
                opener = gzip.open if suffix.endswith(".gz") else open
                with opener(path, "wt") as file:
                    file.write(f"# L = 4\n# beta = {0.2 + i * 0.001}\n")
                    file.write("1 -64 32 1\n2 -128 -48 2\n3 -96 40 1.5\n4 -64 -32 1\n")

            # A wrapper records actual subprocess overlap and thread settings.
            binary = root / "bin"
            binary.mkdir()
            events = root / "events"
            events.mkdir()
            wrapper = binary / "python3"
            wrapper.write_text(f"#!{sys.executable}\n" + '''import json, os, pathlib, subprocess, sys, time
is_analysis = sys.argv[1] == "analyze.py"
start = time.monotonic()
if is_analysis:
    assert all(os.environ[k] == "1" for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))
    time.sleep(0.5)
code = subprocess.call([sys.executable, *sys.argv[1:]])
if is_analysis:
    event = {"start": start, "end": time.monotonic()}
    (pathlib.Path(os.environ["TEST_EVENTS"]) / str(os.getpid())).write_text(json.dumps(event))
sys.exit(code)
''')
            wrapper.chmod(0o755)
            env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       TEST_EVENTS=str(events), MAX_JOBS="20", OPENBLAS_NUM_THREADS="8")
            env.pop("ANALYSIS_JOBS", None)
            command = ["bash", "analyze_grid.sh", "batch", "4:2"]
            parallel = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
            self.assertIn("Concurrent analyses: 8", parallel.stdout)
            summary = (root / "batch/scaling_summary.txt").read_bytes()
            blocks = {p.name: p.read_bytes() for p in results.glob("*_blocks.txt")}
            timeline = []
            for p in events.iterdir():
                event = json.loads(p.read_text())
                timeline.extend(((event["start"], 1), (event["end"], -1)))
            active = maximum = 0
            for _, delta in sorted(timeline):
                active += delta
                maximum = max(maximum, active)
            self.assertEqual(maximum, 8)
            self.assertEqual(len(list(events.iterdir())), 10)

            serial = subprocess.run(command, cwd=root, env=dict(env, ANALYSIS_JOBS="1"),
                                    capture_output=True, text=True)
            self.assertEqual(serial.returncode, 0, serial.stderr)
            self.assertEqual(summary, (root / "batch/scaling_summary.txt").read_bytes())
            self.assertEqual(blocks, {p.name: p.read_bytes() for p in results.glob("*_blocks.txt")})
            self.assertFalse(list(results.glob("*.analysis-b*.npz")))

            # Failed workers stop the batch before plotting.
            (root / "batch/u_vs_rxi.png").unlink()
            (results / "L4_beta0_data.dat").write_text("invalid data\n")
            failed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("analysis failed", failed.stderr)
            self.assertFalse((root / "batch/u_vs_rxi.png").exists())

    def test_invalid_process_count_is_rejected(self):
        script = Path(__file__).resolve().parent / "analyze_grid.sh"
        for value in ("0", "-1", "two"):
            result = subprocess.run(["bash", str(script), "unused", "4:2"],
                                    env=dict(os.environ, ANALYSIS_JOBS=value), capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
