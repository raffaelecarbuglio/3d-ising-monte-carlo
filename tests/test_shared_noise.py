"""Check noise-mode inputs and restart compatibility through the executable."""
from pathlib import Path
import subprocess
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parent.parent


class SharedNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "ising"], cwd=PROJECT, check=True, capture_output=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / "run.in"
        self.data = self.root / "data.dat"

    def run_simulation(self, options="", start="ordered", algorithm="metropolis"):
        self.input.write_text(f"""L = 3
beta = 0.22
n_therm = 2
n_sweeps = 4
measure_every = 1
seed = 123
start = {start}
algorithm = {algorithm}
config_file = config.dat
data_file = data.dat
{options}""")
        return subprocess.run([str(PROJECT / "ising"), str(self.input)], cwd=self.root,
                              capture_output=True, text=True, timeout=10)

    def test_shared_restart_and_mode_mismatch(self):
        options = "noise_mode = shared_sweep\nsigma = 2\nlambda = 0\n"
        first = self.run_simulation(options)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("# noise_mode = shared_sweep\n", self.data.read_text())
        restart = self.run_simulation(options, start="restart")
        self.assertEqual(restart.returncode, 0, restart.stderr)
        before = self.data.read_bytes()
        wrong = self.run_simulation("sigma = 2\n", start="restart")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertIn("noise_mode", wrong.stderr)
        self.assertEqual(self.data.read_bytes(), before)

    def test_legacy_header_defaults_to_independent(self):
        first = self.run_simulation("sigma = 2\n")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.data.write_text(self.data.read_text().replace("# noise_mode = independent\n", ""))
        restart = self.run_simulation("sigma = 2\n", start="restart")
        self.assertEqual(restart.returncode, 0, restart.stderr)
        before = self.data.read_bytes()
        wrong = self.run_simulation("sigma = 2\nnoise_mode = shared_sweep\n", start="restart")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertEqual(self.data.read_bytes(), before)

    def test_invalid_modes_and_combinations_are_rejected(self):
        for options, algorithm in (
            ("noise_mode = unknown\n", "metropolis"),
            ("noise_mode = shared_sweep\nnoise_mode = shared_sweep\n", "metropolis"),
            ("noise_mode = shared_sweep\nlambda = 0.55\n", "metropolis"),
            ("noise_mode = shared_sweep\n", "wolff"),
        ):
            with self.subTest(options=options, algorithm=algorithm):
                run = self.run_simulation(options, algorithm=algorithm)
                self.assertNotEqual(run.returncode, 0)
                self.assertFalse(self.data.exists())


if __name__ == "__main__":
    unittest.main()
