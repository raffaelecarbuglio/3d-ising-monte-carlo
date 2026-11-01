"""Exercise campaign orchestration with real tiny simulations and saved settings."""
import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parent.parent


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="campaign_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ("simulate.sh", "analyse.sh", "analyze_grid.sh", "analyze.py",
                     "plot_u_vs_rxi.py", "Makefile", "main.c", "input.c", "input.h",
                     "ising.c", "ising.h", "rng.c", "rng.h"):
            shutil.copy(PROJECT / name, self.root / name)
        (self.root / "cluster").mkdir()
        shutil.copy(PROJECT / "cluster/check_seeds.py", self.root / "cluster")
        self.conf = self.root / "tiny.conf"
        self.settings = '''ALGORITHM=metropolis
SIGMA=5
N_THERM=100
N_SWEEPS=1024
MEASURE_EVERY=1
BASE_SEED=100
MAX_JOBS=2
BETA_GRID=("4 0.38 0.39" "6 0.38")
BLOCK_SIZES="4:16 6:16"
ANALYSIS_JOBS=2
REFERENCE=""
'''
        self.conf.write_text(self.settings)

    def run_script(self, script, argument):
        return subprocess.run(["bash", str(self.root / script), str(argument)],
                              cwd=self.root.parent, capture_output=True, text=True,
                              env=dict(os.environ, MPLBACKEND="Agg"), timeout=60)

    def batches(self):
        return sorted((self.root / "runs").glob("tiny_*"))

    def test_real_campaign_analysis_and_saved_configuration(self):
        run = self.run_script("simulate.sh", self.conf)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        batch = self.batches()[0]
        rows = list(csv.DictReader((batch / "runs.csv").read_text().splitlines()))
        self.assertEqual([r["seed"] for r in rows], ["100", "101", "102"])
        self.assertTrue(all(r["status"] == "OK" for r in rows))
        self.assertLess(run.stdout.index("START L6"), run.stdout.index("START L4"))
        self.assertEqual((batch / "campaign.conf").read_text(), self.settings)
        # Editing the source config cannot change an existing batch's analysis.
        self.conf.write_text("BLOCK_SIZES=invalid\n")
        result = self.run_script("analyse.sh", batch)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(list((batch / "results").glob("*_blocks.txt"))), 3)
        self.assertTrue((batch / "u_vs_rxi.png").is_file())
        # A clean Wolff campaign uses the same interface.
        self.conf.write_text(self.settings.replace("metropolis", "wolff")
                             .replace("SIGMA=5", "SIGMA=0")
                             .replace("BASE_SEED=100", "BASE_SEED=200"))
        wolff = self.run_script("simulate.sh", self.conf)
        self.assertEqual(wolff.returncode, 0, wolff.stdout + wolff.stderr)
        data = list(self.batches()[-1].glob("results/*_data.dat"))
        self.assertEqual(len(data), 3)

    def test_selected_sizes_split_and_pool(self):
        self.conf.write_text(self.settings + 'REPLICAS_BY_L="6:2"\n')
        result = self.run_script("simulate.sh", self.conf)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        batch = self.batches()[0]
        rows = list(csv.DictReader((batch / "runs.csv").open()))
        self.assertEqual([r["seed"] for r in rows], ["100", "101", "102", "103"])
        self.assertEqual([int(r["n_sweeps"]) for r in rows], [1024, 1024, 512, 512])
        self.assertTrue(all(r["n_therm"] == "100" for r in rows))
        self.assertEqual([r["name"].split("_")[-1] for r in rows[2:]], ["r0", "r1"])
        result = self.run_script("analyse.sh", batch)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        points = [line for line in (batch / "scaling_summary.txt").read_text().splitlines()
                  if line and not line.startswith("#")]
        self.assertEqual(len(points), 3)
        pooled = next((batch / "results").glob("L6*_blocks.txt"))
        self.assertIn("# replicas = 2", pooled.read_text())
        self.assertIn("paired block bootstrap", pooled.read_text())
        # A missing replica must not silently reduce the requested statistics.
        (batch / "results" / (rows[-1]["name"] + "_data.dat")).unlink()
        self.assertNotEqual(self.run_script("analyse.sh", batch).returncode, 0)

    def test_preflight_rejects_reused_seeds_and_colliding_points(self):
        history = self.root / "old"
        history.mkdir()
        (history / "a.in").write_text("seed = 100\n")
        for settings, message in (
            (self.settings, "Repeated seed 100"),
            (self.settings.replace('"4 0.38 0.39"', '"4 0.38 0.38000000001"'), "filename collision"),
            (self.settings.replace("MAX_JOBS=2", "MAX_JOBS=0"), "MAX_JOBS"),
            (self.settings + 'REPLICAS_BY_L="6:3"\n', "divisible"),
            (self.settings + 'REPLICAS_BY_L="6:0"\n', "REPLICAS_BY_L"),
            (self.settings + 'REPLICAS_BY_L="8:2"\n', "absent from BETA_GRID"),
            (self.settings + 'REPLICAS_BY_L="6:2 6:4"\n', "Repeated replica"),
            (self.settings.replace("metropolis", "wolff"), "Wolff requires"),
        ):
            self.conf.write_text(settings)
            result = self.run_script("simulate.sh", self.conf)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stdout + result.stderr)
            self.assertEqual(self.batches(), [])
            self.assertNotIn("START ", result.stdout)
        # Unrelated duplicates in old history do not reject new seeds.
        (history / "b.in").write_text("seed = 100\n")
        self.conf.write_text(self.settings.replace("BASE_SEED=100", "BASE_SEED=300"))
        result = self.run_script("simulate.sh", self.conf)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_worker_failure_is_recorded_and_analysis_is_refused(self):
        subprocess.run(["make", "ising"], cwd=self.root, capture_output=True, check=True)
        # Simulate an executable failing at runtime after successful preparation.
        executable = self.root / "ising"
        executable.write_text("#!/bin/sh\nexit 7\n")
        executable.chmod(0o755)
        result = self.run_script("simulate.sh", self.conf)
        self.assertNotEqual(result.returncode, 0)
        batch = self.batches()[0]
        rows = list(csv.DictReader((batch / "runs.csv").read_text().splitlines()))
        self.assertEqual(sum(r["status"] == "FAILED" for r in rows), 2)
        self.assertEqual(sum(r["status"] == "PENDING" for r in rows), 1)
        refused = self.run_script("analyse.sh", batch)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("OK for every simulation", refused.stderr)
        self.assertFalse((batch / "scaling_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
