import tempfile
import unittest
from pathlib import Path

import numpy as np

from fit_scaling import (
    OMEGA,
    asymptotic_curve,
    bootstrap_point,
    design_matrix,
    fit_polynomials,
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

    def test_weighted_fit_recovers_exact_coefficients(self):
        r_xi = np.tile(
            np.linspace(0.3, 1.0, 10), 2
        )
        sizes = np.repeat([16.0, 32.0], 10)
        coefficients = np.array([
            1.2, -0.4, 0.3, 0.8, -0.2
        ])
        matrix = design_matrix(r_xi, sizes, 2, 1)
        binder = matrix @ coefficients
        errors = np.full(len(r_xi), 0.01)

        fitted, chi2, dof = fit_polynomials(
            r_xi,
            binder,
            errors,
            sizes,
            2,
            1,
        )

        np.testing.assert_allclose(
            fitted,
            coefficients,
            rtol=1e-11,
            atol=1e-11,
        )
        self.assertAlmostEqual(chi2, 0.0, places=20)
        self.assertEqual(
            dof, len(r_xi) - len(coefficients)
        )
        np.testing.assert_allclose(
            asymptotic_curve(
                fitted, np.array([0.4]), 2
            ),
            (
                coefficients[0]
                + coefficients[1] * 0.4
                + coefficients[2] * 0.4**2
            ),
        )

    def test_read_block_file_and_paired_bootstrap(self):
        contents = """# L = 8
# beta = 0.22
# block_size = 2
# number_of_blocks = 4
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
            filename = (
                Path(directory) / "sample_blocks.txt"
            )
            filename.write_text(
                contents, encoding="utf-8"
            )
            point = read_block_file(filename)

        self.assertEqual(point["L"], 8)
        self.assertEqual(
            point["blocks"].shape, (4, 4)
        )

        first_rng = np.random.default_rng(123)
        second_rng = np.random.default_rng(123)
        first = bootstrap_point(
            point, 20, first_rng
        )
        second = bootstrap_point(
            point, 20, second_rng
        )
        np.testing.assert_allclose(
            first[0], second[0]
        )
        np.testing.assert_allclose(
            first[1], second[1]
        )


if __name__ == "__main__":
    unittest.main()
