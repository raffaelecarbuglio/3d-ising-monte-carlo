"""Restart equivalence and local tests of the Slurm preparation workflow."""
import fcntl
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "ising"


class ClusterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)

    def run_input(self, name, *, start="random", algorithm="metropolis", sigma=0,
                  sweeps=30, thermal=7, seed=12345, L=8, measure=4, save=3, run=True):
        path = self.directory / (name + ".in")
        path.write_text(
            f"L = {L}\nbeta = 0.22\nn_therm = {thermal}\nn_sweeps = {sweeps}\n"
            f"measure_every = {measure}\nseed = {seed}\nstart = {start}\n"
            f"algorithm = {algorithm}\nconfig_file = {name}.cfg\n"
            f"data_file = {name}.dat\nsigma = {sigma}\nsave_every = {save}\n"
        )
        if not run:
            return path
        return subprocess.run([EXE, path], cwd=self.directory, capture_output=True, check=True)

    def rows(self, name):
        return [s for s in (self.directory / (name + ".dat")).read_text().splitlines()
                if s and not s.startswith("#")]

    def state(self, name):
        text = (self.directory / (name + ".cfg")).read_text()
        return text.split("\ncheckpoint 1\n")

    def test_split_runs_match_continuous(self):
        for algorithm, sigma in [("metropolis", 0), ("metropolis", 2), ("wolff", 0)]:
            with self.subTest(algorithm=algorithm, sigma=sigma):
                self.run_input("reference", algorithm=algorithm, sigma=sigma)
                self.run_input("split", algorithm=algorithm, sigma=sigma, sweeps=11)
                # Completed thermalization must not be repeated; the new seed is ignored.
                self.run_input("split", algorithm=algorithm, sigma=sigma, start="restart",
                               sweeps=19, thermal=100, seed=999)
                self.assertEqual(self.rows("reference"), self.rows("split"))
                self.assertEqual(self.state("reference")[0], self.state("split")[0])
                self.assertEqual(self.state("reference")[1].split("rng ")[1],
                                 self.state("split")[1].split("rng ")[1])
                for name in ("reference", "split"):
                    (self.directory / (name + ".dat")).unlink()

    def test_checkpoint_discards_uncommitted_tail(self):
        self.run_input("reference")
        self.run_input("split", sweeps=11)
        with (self.directory / "split.dat").open("a") as file:
            file.write("12 0 0 0\n16 0 0 0\n20 partial")
        self.run_input("split", start="restart", sweeps=19, thermal=0)
        self.assertEqual(self.rows("reference"), self.rows("split"))

    def test_signals_resume_unfinished_thermalization(self):
        self.run_input("reference", L=24, thermal=5000, sweeps=12, save=1000)
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            with self.subTest(signal=sig):
                name = "stop" + str(sig)
                path = self.run_input(name, L=24, thermal=5000, sweeps=12, save=1000, run=False)
                process = subprocess.Popen([EXE, path], cwd=self.directory,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 10
                    while not (self.directory / (name + ".cfg")).exists():
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("simulation did not create its initial checkpoint")
                        time.sleep(0.001)
                    process.send_signal(sig)
                    output, error = process.communicate(timeout=10)
                    self.assertEqual(process.returncode, 0, error.decode())
                    self.assertIn(b"Interrotto e salvato", output)
                    trailer = self.state(name)[1]
                    remaining = int(trailer.split("thermal_remaining ")[1].split()[0])
                    self.assertGreater(remaining, 0)
                    self.run_input(name, L=24, start="restart", thermal=0, sweeps=12, save=1000)
                    self.assertEqual(self.rows("reference"), self.rows(name))
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

    def test_incompatible_restart_changes_nothing(self):
        self.run_input("run")
        saved = (self.directory / "run.dat").read_bytes()
        checkpoint = (self.directory / "run.cfg").read_bytes()
        for old, new in [("beta = 0.22", "beta = 0.23"),
                         ("sigma = 0", "sigma = 2"),
                         ("measure_every = 4", "measure_every = 2"),
                         ("algorithm = metropolis", "algorithm = wolff")]:
            path = self.run_input("run", start="restart", run=False)
            path.write_text(path.read_text().replace(old, new))
            result = subprocess.run([EXE, path], cwd=self.directory, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((self.directory / "run.dat").read_bytes(), saved)
            self.assertEqual((self.directory / "run.cfg").read_bytes(), checkpoint)

    def test_hard_kill_resumes_last_checkpoint(self):
        path = self.run_input("killed", L=24, thermal=7, sweeps=1000000,
                              sigma=2, save=5, run=False)
        process = subprocess.Popen([EXE, path], cwd=self.directory,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 10
            checkpoint = self.directory / "killed.cfg"
            while True:
                if checkpoint.exists():
                    count = int(checkpoint.read_text().split("production_sweeps ")[1].split()[0])
                    if count >= 5:
                        break
                if process.poll() is not None or time.monotonic() > deadline:
                    self.fail("simulation did not reach a production checkpoint")
                time.sleep(0.001)
            process.kill()
            process.communicate(timeout=10)
            count = int(checkpoint.read_text().split("production_sweeps ")[1].split()[0])
            self.run_input("reference", L=24, sigma=2, sweeps=count + 12)
            self.run_input("killed", L=24, sigma=2, start="restart", thermal=0, sweeps=12)
            self.assertEqual(self.rows("reference"), self.rows("killed"))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_legacy_configuration_and_short_data(self):
        self.run_input("legacy", sweeps=8)
        prefix, _ = self.state("legacy")
        (self.directory / "legacy.cfg").write_text(prefix + "\n")
        self.run_input("legacy", start="restart", sweeps=4, thermal=0, seed=678)
        self.assertEqual(len(self.rows("legacy")), 3)
        (self.directory / "legacy.dat").write_text("# truncated\n")
        result = subprocess.run([EXE, self.directory / "legacy.in"],
                                cwd=self.directory, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual((self.directory / "legacy.dat").read_text(), "# truncated\n")

    def prepare(self):
        campaign = self.directory / "campaign"
        env = dict(os.environ, REPLICAS="2", N_THERM="7", N_SWEEPS="11",
                   MEASURE_EVERY="4", SAVE_EVERY="3", SIGMA="2", BASE_SEED="123")
        subprocess.run([ROOT / "cluster/prepare.sh", campaign, "8", "0.22", "0.23"],
                       env=env, capture_output=True, check=True)
        return campaign

    def mock_slurm(self):
        commands = self.directory / "bin"
        commands.mkdir()
        srun = commands / "srun"
        srun.write_text('#!/bin/bash\nwhile [[ $1 == --* ]]; do shift; done\nexec "$@"\n')
        sbatch = commands / "sbatch"
        sbatch.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$MOCK_LOG"\n'
                          'n=$(wc -l < "$MOCK_LOG")\necho "$n"\n')
        srun.chmod(0o755)
        sbatch.chmod(0o755)
        return dict(os.environ, PATH=str(commands) + ":" + os.environ["PATH"],
                    SLURM_JOB_ID="123", MOCK_LOG=str(self.directory / "submission.log"))

    def test_preparation_and_two_batch_segments(self):
        campaign = self.prepare()
        inputs = sorted((campaign / "inputs/new").glob("*.in"))
        self.assertEqual(len(inputs), 4)
        seeds = [int(p.read_text().split("seed = ")[1].split()[0]) for p in inputs]
        self.assertEqual(sorted(seeds), [123, 124, 125, 126])
        checksum = hashlib.sha256((campaign / "ising").read_bytes()).hexdigest()
        self.assertIn(checksum, (campaign / "provenance.txt").read_text())
        env = self.mock_slurm()
        for _ in range(2):
            subprocess.run(["bash", "job.batch"], cwd=campaign, env=env, check=True)
        for data in (campaign / "results").glob("*_data.dat"):
            sweeps = [int(s.split()[0]) for s in data.read_text().splitlines()
                      if s and not s.startswith("#")]
            self.assertEqual(sweeps, [4, 8, 12, 16, 20])
        subprocess.run([ROOT / "cluster/submit.sh", campaign, "3"], env=env,
                       capture_output=True, check=True)
        submissions = (self.directory / "submission.log").read_text().splitlines()
        self.assertIn("--ntasks=4", submissions[0])
        self.assertNotIn("--dependency", submissions[0])
        self.assertIn("--dependency=afterok:1", submissions[1])
        self.assertIn("--dependency=afterok:2", submissions[2])

    def test_batch_lock_and_failure(self):
        campaign = self.prepare()
        env = self.mock_slurm()
        with (campaign / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(["bash", "job.batch"], cwd=campaign, env=env,
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        (self.directory / "bin/srun").write_text("#!/bin/bash\nexit 1\n")
        result = subprocess.run(["bash", "job.batch"], cwd=campaign, env=env)
        self.assertNotEqual(result.returncode, 0)

    def test_invalid_preparation(self):
        for settings, betas in [({"REPLICAS": "33"}, ["0.22"]),
                                ({"ALGORITHM": "wolff", "SIGMA": "2"}, ["0.22"]),
                                ({"BASE_SEED": "2147483647"}, ["0.22"]),
                                ({}, ["nan"]), ({}, ["0.22", "0.22"])]:
            campaign = self.directory / "invalid"
            result = subprocess.run([ROOT / "cluster/prepare.sh", campaign, "8", *betas],
                                    env=dict(os.environ, **settings), capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(campaign.exists())


if __name__ == "__main__":
    unittest.main()
