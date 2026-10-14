import gzip
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class GridTests(unittest.TestCase):
    def test_parallel_matches_serial_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory(prefix="analysis test ") as directory:
            root = Path(directory)
            project = Path(__file__).resolve().parent.parent
            for name in ("analyze.py", "analyze_grid.sh", "plot_u_vs_rxi.py"):
                shutil.copy(project / name, root / name)
            results = root / "batch" / "results"
            results.mkdir(parents=True)
            for i in range(4):
                suffix = ".dat.gz" if i % 2 else ".dat"
                path = results / f"L4_beta{i}_data{suffix}"
                opener = gzip.open if suffix.endswith(".gz") else open
                with opener(path, "wt") as file:
                    file.write(f"# L = 4\n# beta = {0.2 + i * 0.001}\n")
                    file.write("1 -64 32 1\n2 -128 -48 2\n3 -96 40 1.5\n4 -64 -32 1\n")

            env = dict(os.environ, ANALYSIS_JOBS="2")
            command = ["bash", "analyze_grid.sh", "batch", "4:2"]
            parallel = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(parallel.returncode, 0, parallel.stdout + parallel.stderr)
            summary = (root / "batch/scaling_summary.txt").read_bytes()
            blocks = {p.name: p.read_bytes() for p in results.glob("*_blocks.txt")}

            serial = subprocess.run(command, cwd=root, env=dict(env, ANALYSIS_JOBS="1"),
                                    capture_output=True, text=True)
            self.assertEqual(serial.returncode, 0, serial.stderr)
            self.assertEqual(summary, (root / "batch/scaling_summary.txt").read_bytes())
            self.assertEqual(blocks, {p.name: p.read_bytes() for p in results.glob("*_blocks.txt")})

            # Failed workers stop the batch before plotting.
            (root / "batch/u_vs_rxi.png").unlink()
            (results / "L4_beta0_data.dat").write_text("invalid data\n")
            failed = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("analysis failed", failed.stderr)
            self.assertFalse((root / "batch/u_vs_rxi.png").exists())

    def test_invalid_process_count_is_rejected(self):
        script = Path(__file__).resolve().parent.parent / "analyze_grid.sh"
        for value in ("0", "-1", "two"):
            result = subprocess.run(["bash", str(script), "unused", "4:2"],
                                    env=dict(os.environ, ANALYSIS_JOBS=value), capture_output=True)
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
