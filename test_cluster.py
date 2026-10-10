"""Restart consistency and local Slurm smoke tests."""
import fcntl
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
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.directory = Path(tmp.name)

    def input(self, name, start="random", algorithm="metropolis", sigma=0,
              sweeps=30, thermal=7, L=8, save=3):
        path = self.directory / (name + ".in")
        path.write_text(
            f"L = {L}\nbeta = 0.22\nn_therm = {thermal}\nn_sweeps = {sweeps}\n"
            f"measure_every = 4\nseed = {999 if start == 'restart' else 12345}\n"
            f"start = {start}\nalgorithm = {algorithm}\nconfig_file = {name}.cfg\n"
            f"data_file = {name}.dat\nsigma = {sigma}\nsave_every = {save}\n"
        )
        return path

    def run_input(self, name, **options):
        subprocess.run([EXE, self.input(name, **options)], cwd=self.directory,
                       capture_output=True, check=True)

    def rows(self, name):
        return [s for s in (self.directory / (name + ".dat")).read_text().splitlines()
                if s and not s.startswith("#")]

    def test_restart_equivalence_and_data_recovery(self):
        for algorithm, sigma in [("metropolis", 0), ("metropolis", 2), ("wolff", 0)]:
            with self.subTest(algorithm=algorithm, sigma=sigma):
                a, b = f"ref_{algorithm}_{sigma}", f"split_{algorithm}_{sigma}"
                self.run_input(a, algorithm=algorithm, sigma=sigma)
                self.run_input(b, algorithm=algorithm, sigma=sigma, sweeps=11)
                with (self.directory / (b + ".dat")).open("a") as file:
                    file.write("12 0 0 0\n16 partial")
                self.run_input(b, algorithm=algorithm, sigma=sigma, start="restart",
                               sweeps=19, thermal=100)
                self.assertEqual(self.rows(a), self.rows(b))
                states = [(self.directory / (n + ".cfg")).read_text() for n in (a, b)]
                self.assertEqual(states[0].split("checkpoint")[0], states[1].split("checkpoint")[0])
                self.assertEqual(states[0].split("rng ")[1], states[1].split("rng ")[1])
                path = self.input(b, algorithm=algorithm, sigma=sigma, start="restart")
                path.write_text(path.read_text().replace("beta = 0.22", "beta = 0.23"))
                saved = (self.directory / (b + ".dat")).read_bytes()
                result = subprocess.run([EXE, path], cwd=self.directory, capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual((self.directory / (b + ".dat")).read_bytes(), saved)

    def test_interrupted_thermalization_and_production(self):
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT, signal.SIGKILL):
            with self.subTest(signal=sig):
                name, reference = f"stop_{sig}", f"ref_{sig}"
                hard = sig == signal.SIGKILL
                thermal = 7 if hard else 5000
                path = self.input(name, L=24, sigma=2, thermal=thermal,
                                  sweeps=1000000, save=5 if hard else 1000)
                process = subprocess.Popen([EXE, path], cwd=self.directory,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                try:
                    deadline = time.monotonic() + 10
                    checkpoint = self.directory / (name + ".cfg")
                    while True:
                        if checkpoint.exists():
                            text = checkpoint.read_text()
                            count = int(text.split("production_sweeps ")[1].split()[0])
                            if not hard or count >= 5: break
                        if process.poll() is not None or time.monotonic() > deadline:
                            self.fail("simulation did not reach the checkpoint")
                        time.sleep(0.001)
                    process.send_signal(sig)
                    output, error = process.communicate(timeout=10)
                    if not hard:
                        self.assertEqual(process.returncode, 0, error.decode())
                        self.assertIn(b"Interrotto e salvato", output)
                    text = checkpoint.read_text()
                    count = int(text.split("production_sweeps ")[1].split()[0])
                    if not hard:
                        self.assertGreater(int(text.split("thermal_remaining ")[1].split()[0]), 0)
                    self.run_input(reference, L=24, sigma=2, thermal=thermal, sweeps=count + 12, save=1000)
                    self.run_input(name, L=24, sigma=2, start="restart", thermal=0, sweeps=12, save=1000)
                    self.assertEqual(self.rows(reference), self.rows(name))
                finally:
                    if process.poll() is None: process.kill()
                    process.communicate()

    def test_legacy_restart(self):
        self.run_input("legacy", sweeps=8)
        path = self.directory / "legacy.cfg"
        path.write_text(path.read_text().split("checkpoint")[0])
        self.run_input("legacy", start="restart", sweeps=4, thermal=0)
        self.assertEqual(len(self.rows("legacy")), 3)

    def test_prepare_submit_and_continue(self):
        campaign, commands = self.directory / "campaign", self.directory / "bin"
        commands.mkdir()
        mocks = {
            "srun": '#!/bin/bash\nwhile [[ $1 == --* ]]; do shift; done\nexec "$@"\n',
            "sbatch": '#!/bin/bash\nwhile [[ $1 == --* ]]; do shift; done\nexec bash "$@"\n',
        }
        for name, text in mocks.items():
            path = commands / name
            path.write_text(text)
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(commands) + ":" + os.environ["PATH"], SLURM_JOB_ID="1",
                   REPLICAS="2", N_THERM="7", N_SWEEPS="11", MEASURE_EVERY="4", SAVE_EVERY="3",
                   SIGMA="2", BASE_SEED="123")
        subprocess.run([ROOT / "cluster/prepare.sh", campaign, "8", "0.22", "0.23"],
                       env=env, capture_output=True, check=True)
        inputs = list((campaign / "inputs/new").glob("*.in"))
        self.assertEqual(sorted(int(p.read_text().split("seed = ")[1].split()[0]) for p in inputs),
                         [123, 124, 125, 126])
        self.assertIn("#SBATCH --ntasks=4", (campaign / "job.batch").read_text())
        for _ in range(2):
            subprocess.run([ROOT / "cluster/submit.sh", campaign], env=env, check=True)
        for data in (campaign / "results").glob("*_data.dat"):
            self.assertEqual([int(s.split()[0]) for s in data.read_text().splitlines()
                              if s and not s.startswith("#")], [4, 8, 12, 16, 20])
        with (campaign / ".lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run([ROOT / "cluster/submit.sh", campaign], env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        (commands / "srun").write_text("#!/bin/bash\nexit 1\n")
        result = subprocess.run([ROOT / "cluster/submit.sh", campaign], env=env)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
