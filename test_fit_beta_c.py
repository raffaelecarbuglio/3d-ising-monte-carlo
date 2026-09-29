import math
import unittest

import numpy as np

from fit_beta_c import (
    NU,
    bootstrap_rxi,
    design_matrix,
    fit_at_beta_c,
    fit_beta_c,
)
from fit_scaling import OMEGA


class FitBetaCTests(unittest.TestCase):
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
