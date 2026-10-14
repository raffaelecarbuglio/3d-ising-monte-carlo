import tempfile
import unittest
from pathlib import Path

import numpy as np

from fit_scaling import (
    OMEGA,
    asymptotic_curve,
    bootstrap_pair,
    design_matrix,
    fit,
    read_block_file,
)


class FitScalingTests(unittest.TestCase):
    def test_design_matrix(self):
        r_xi = np.array([0.5, 0.75])
        sizes = np.array([16.0, 32.0])
        matrix = design_matrix(r_xi, sizes, 2, 1)

        expected = np.column_stack((
            np.ones(2),
            r_xi,
            r_xi**2,
            sizes**(-OMEGA),
            sizes**(-OMEGA) * r_xi,
        ))
        np.testing.assert_allclose(matrix, expected)

    def test_fit_recovers_exact_coefficients(self):
        r_xi = np.tile(np.linspace(0.3, 1.0, 10), 2)
        sizes = np.repeat([16.0, 32.0], 10)
        expected = np.array([1.2, -0.4, 0.3, 0.8, -0.2])
        u = design_matrix(r_xi, sizes, 2, 1) @ expected
        err_u = np.full(len(u), 0.01)

        coefficients, chi2, dof = fit(
            r_xi, u, err_u, sizes, 2, 1
        )

        np.testing.assert_allclose(coefficients, expected)
        self.assertAlmostEqual(chi2, 0.0, places=20)
        self.assertEqual(dof, len(u) - len(expected))
        self.assertAlmostEqual(
            asymptotic_curve(coefficients, np.array([0.4]), 2)[0],
            expected[0] + expected[1] * 0.4 + expected[2] * 0.4**2,
        )

    def test_read_blocks_and_reproducible_bootstrap_pair(self):
        contents = """# L = 8
# beta = 0.22
# Rxi = 0.5
# err_Rxi = 0.01
# U = 1.2
# err_U = 0.02
# columns: g_zero g_min m2 m4
4.0 1.0 1.0 1.2
5.0 1.0 1.1 1.4
6.0 1.0 1.2 1.7
7.0 1.0 1.3 2.0
"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "sample_blocks.txt"
            filename.write_text(contents, encoding="utf-8")
            point = read_block_file(filename)

        first = bootstrap_pair(point, np.random.default_rng(123))
        second = bootstrap_pair(point, np.random.default_rng(123))

        self.assertEqual(point["L"], 8)
        self.assertEqual(point["blocks"].shape, (4, 4))
        np.testing.assert_allclose(first, second)


if __name__ == "__main__":
    unittest.main()
