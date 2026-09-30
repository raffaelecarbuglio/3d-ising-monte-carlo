import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from fit_beta_c import (
    NU,
    bootstrap_fits,
    bootstrap_rxi,
    design_matrix,
    fit_at_beta_c,
    fit_beta_c,
)
from fit_scaling import OMEGA


class FitBetaCTests(unittest.TestCase):
    @staticmethod
    def sample(sizes=(8, 16, 24, 32)):
        sizes = np.repeat(sizes, 9).astype(float)
        beta = 0.2234 + np.tile(np.linspace(-0.3, 0.3, 9), len(sizes) // 9) / sizes**(1 / NU)
        coefficients = np.array([0.64, 0.35, -0.04, 0.12, -0.03])
        r_xi = design_matrix(beta, sizes, 0.2234, 2, 1) @ coefficients
        return beta, r_xi, np.full(len(beta), 0.0002), sizes

    def test_two_sizes_have_identical_profiles_and_are_rejected(self):
        beta, r_xi, errors, sizes = self.sample((24, 32))
        for trial in (0.2220, 0.2234, 0.2250):
            _, chi2 = fit_at_beta_c(beta, r_xi, errors, sizes, trial, 2, 1)
            self.assertLess(chi2, 1e-12)
        with self.assertRaisesRegex(ValueError, "almeno 3 taglie"):
            fit_beta_c(beta, r_xi, errors, sizes, 2, 1, 0.221, 0.226)

    def test_flat_profile_is_rejected_even_with_four_sizes(self):
        beta, r_xi, errors, sizes = self.sample()
        r_xi[:] = 0.64  # Nessuna dipendenza da beta: beta_c non e' misurabile.
        with self.assertRaisesRegex(ValueError, "profilo chi2 piatto"):
            fit_beta_c(beta, r_xi, errors, sizes, 2, 1, 0.221, 0.226)

    def test_search_range_must_constrain_beta_c(self):
        beta, r_xi, errors, sizes = self.sample()
        with self.assertRaisesRegex(ValueError, "poco vincolato"):
            fit_beta_c(beta, r_xi, errors * 10000, sizes, 2, 1, 0.221, 0.226)
        with self.assertRaisesRegex(ValueError, "minimo al bordo"):
            fit_beta_c(beta, r_xi, errors, sizes, 2, 1, 0.221, 0.222)

    def test_invalid_errors_are_rejected(self):
        beta, r_xi, errors, sizes = self.sample()
        for invalid in (0, -1, np.nan, np.inf):
            errors[0] = invalid
            with self.assertRaises(ValueError):
                fit_beta_c(beta, r_xi, errors, sizes, 2, 1, 0.221, 0.226)

    def test_noisy_three_size_fit(self):
        beta, r_xi, errors, sizes = self.sample((16, 24, 32))
        r_xi += np.random.default_rng(9).normal(0, errors)
        beta_c, _, chi2, dof = fit_beta_c(beta, r_xi, errors, sizes, 2, 1, 0.221, 0.226)
        self.assertLess(abs(beta_c - 0.2234), 0.00003)
        self.assertTrue(0 < chi2 / dof < 3)

    def block_points(self):
        beta, r_xi, errors, sizes = self.sample((16, 24, 32))
        rng = np.random.default_rng(10)
        points = []
        for b, r, error, size in zip(beta, r_xi, errors, sizes):
            g_zero = 1 + (2 * size * np.sin(np.pi / size) * r)**2
            noise = rng.normal(0, 0.01, 64)
            noise -= noise.mean()
            blocks = np.column_stack((g_zero * (1 + noise), np.ones(64),
                                      np.ones(64), np.full(64, 1.6)))
            points.append(dict(L=int(size), beta=b, Rxi=r, err_Rxi=error,
                               U=1.6, err_U=0.01, blocks=blocks))
        return points

    def test_complete_bootstrap_is_reproducible(self):
        points = self.block_points()
        first = bootstrap_fits(points, 2, 1, 6, 123, 0.221, 0.226)
        second = bootstrap_fits(points, 2, 1, 6, 123, 0.221, 0.226)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertGreater(np.std(first[0]), 0)
        self.assertLess(abs(np.mean(first[0]) - 0.2234), 0.0001)

    def test_command_line_outputs_and_missing_size(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for index, point in enumerate(self.block_points()):
                header = "\n".join(f"{key} = {value}" for key, value in point.items()
                                   if key != "blocks")
                np.savetxt(directory / f"run{index}_blocks.txt", point["blocks"], header=header)
            command = [sys.executable, str(Path(__file__).with_name("fit_beta_c.py")),
                       str(directory), "--bootstrap", "3", "--beta-min", "0.221",
                       "--beta-max", "0.226", "--output-prefix", str(directory / "fit")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = (directory / "fit_summary.txt").read_text()
            self.assertIn("sizes = 16 24 32", report)
            self.assertIn("Rxi range = 0.45 0.75", report)
            for extension in ("png", "pdf"):
                self.assertGreater((directory / f"fit.{extension}").stat().st_size, 1000)
            result = subprocess.run(command + ["--sizes", "8", "16", "24", "32"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("taglia richiesta", result.stderr)

    def test_design_matrix(self):
        beta = np.array([0.220, 0.222])
        sizes = np.array([16.0, 32.0])
        beta_c = 0.221
        matrix = design_matrix(beta, sizes, beta_c, 2, 1)

        x = (beta - beta_c) * sizes ** (1.0 / NU)
        expected = np.column_stack((
            np.ones(2),
            x,
            x**2,
            sizes**(-OMEGA),
            sizes**(-OMEGA) * x,
        ))
        np.testing.assert_allclose(matrix, expected)

    def test_fit_recovers_exact_beta_c(self):
        beta_c_expected = 0.2234
        sizes = np.repeat([8.0, 16.0, 24.0, 32.0], 9)
        beta = np.tile(np.linspace(0.218, 0.229, 9), 4)
        coefficients_expected = np.array([0.64, 0.35, -0.04, 0.12, -0.03])
        r_xi = design_matrix(
            beta, sizes, beta_c_expected, 2, 1
        ) @ coefficients_expected
        errors = np.full(len(beta), 0.002)

        beta_c, coefficients, chi2, dof = fit_beta_c(
            beta,
            r_xi,
            errors,
            sizes,
            2,
            1,
            0.220,
            0.226,
        )

        self.assertAlmostEqual(beta_c, beta_c_expected, places=8)
        np.testing.assert_allclose(coefficients, coefficients_expected, atol=1e-7)
        self.assertLess(chi2, 1e-14)
        self.assertEqual(dof, len(beta) - len(coefficients_expected) - 1)

    def test_fit_at_beta_c_recovers_coefficients(self):
        beta_c = 0.222
        beta = np.tile(np.linspace(0.219, 0.225, 7), 3)
        sizes = np.repeat([12.0, 20.0, 28.0], 7)
        expected = np.array([0.61, 0.28, 0.10])
        r_xi = design_matrix(beta, sizes, beta_c, 1, 0) @ expected
        errors = np.full(len(beta), 0.01)

        coefficients, chi2 = fit_at_beta_c(
            beta, r_xi, errors, sizes, beta_c, 1, 0
        )

        np.testing.assert_allclose(coefficients, expected)
        self.assertAlmostEqual(chi2, 0.0, places=20)

    def test_bootstrap_rxi_is_reproducible(self):
        point = {
            "L": 8,
            "beta": 0.22,
            "blocks": np.array([
                [4.0, 1.0, 1.0, 1.2],
                [5.0, 1.0, 1.1, 1.4],
                [6.0, 1.0, 1.2, 1.7],
                [7.0, 1.0, 1.3, 2.0],
            ]),
        }

        first = bootstrap_rxi(point, np.random.default_rng(123))
        second = bootstrap_rxi(point, np.random.default_rng(123))

        self.assertTrue(math.isfinite(first))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
