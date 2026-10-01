import contextlib
import io
import tempfile
import shutil
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from compare_universality import (
    bootstrap_comparison, evaluate, jackknife_pair,
    load_points, load_reference, main, observable_pairs,
)
from prepare_clean_reference import basis, fit_clean, main as prepare_main
from fit_scaling import OMEGA


def make_point(size, r, u, seed, beta=0.22):
    rng = np.random.default_rng(seed)
    g0 = 1 + (2 * size * np.sin(np.pi / size) * r)**2
    m2 = g0 / size**3
    means = np.array([g0, 1, m2, u * m2**2])
    noise = rng.normal(size=(100, 3))
    noise -= noise.mean(axis=0)
    perturbations = np.column_stack((
        0.003 * noise[:, 0], 0.003 * noise[:, 1],
        0.003 * noise[:, 0], 0.002 * noise[:, 0] + 0.003 * noise[:, 2],
    ))
    point = dict(L=size, beta=beta, blocks=means * (1 + perturbations))
    point["pair"], point["covariance"] = jackknife_pair(point)
    point.update(Rxi=point["pair"][0], U=point["pair"][1],
                 err_Rxi=np.sqrt(point["covariance"][0, 0]),
                 err_U=np.sqrt(point["covariance"][1, 1]))
    return point


def save_point(path, point):
    with open(path, "w", encoding="utf-8") as file:
        for key in ("L", "beta", "Rxi", "err_Rxi", "U", "err_U"):
            file.write(f"# {key} = {point[key]:.17g}\n")
        np.savetxt(file, point["blocks"])


class UniversalityTests(unittest.TestCase):
    def test_correlated_line_matches_analytically_profiled_distance(self):
        rng = np.random.default_rng(23)
        r = np.tile(np.linspace(0.3, 1, 10), 3)
        sizes = np.repeat([16, 32, 64], 10)
        covariance = np.tile([[0.0004, -0.00015], [-0.00015, 0.0003]], (len(r), 1, 1))
        coefficients = np.array([2.0, -0.3, 0.08])
        pairs = np.column_stack((r, basis(r, sizes, 1, 0, OMEGA, (0.3, 1)) @ coefficients))
        pairs += rng.multivariate_normal([0, 0], covariance[0], size=len(r))
        result, chi2, dof = fit_clean(pairs, covariance, sizes, 1, 0, OMEGA, (0.3, 1))

        def exact_profile(c):
            slope = c[1] * 2 / 0.7
            residual = pairs[:, 1] - evaluate(c, pairs[:, 0], 1, (0.3, 1), sizes, OMEGA)
            variance = covariance[:, 1, 1] + slope**2 * covariance[:, 0, 0] - 2 * slope * covariance[:, 0, 1]
            return np.sum(residual**2 / variance)

        reference = minimize(exact_profile, coefficients, method="BFGS", tol=1e-9)
        np.testing.assert_allclose(result, reference.x, atol=1e-6)
        self.assertAlmostEqual(chi2, reference.fun, places=7)
        self.assertEqual(dof, len(r) - 3)

    def test_nonlinear_fit_recovers_reference_and_correction(self):
        r = np.tile(np.linspace(0.3, 1, 12), 4)
        sizes = np.repeat([16, 24, 32, 64], 12)
        domain = (0.3, 1)
        expected = np.array([2.1, -0.4, 0.03, 0.08, -0.02])
        pairs = np.column_stack((r, basis(r, sizes, 2, 1, OMEGA, domain) @ expected))
        covariance = np.tile([[1e-6, -1e-6], [-1e-6, 4e-6]], (len(r), 1, 1))
        actual, chi2, _ = fit_clean(pairs, covariance, sizes, 2, 1, OMEGA, domain)
        np.testing.assert_allclose(actual, expected, atol=1e-10)
        self.assertLess(chi2, 1e-15)

    def test_jackknife_retains_pair_covariance(self):
        point = make_point(16, 0.6, 1.8, 1)
        pair, covariance = jackknife_pair(point)
        np.testing.assert_allclose(pair, [0.6, 1.8], atol=1e-14)
        self.assertLess(covariance[0, 1], 0)
        self.assertGreater(np.linalg.det(covariance), 0)
        with self.assertRaises(ValueError):
            observable_pairs([1, 0, 1, 1], 16)

    def test_saved_reference_adds_cross_point_covariance(self):
        # Vary only the reference intercept, isolating its shared uncertainty.
        samples = np.zeros((30, 11))
        samples[:, 0] = np.linspace(1.99, 2.01, 30)
        perturbed = [make_point(16, 0.6, 2, 90), make_point(32, 0.6, 2, 91)]
        for point in perturbed:
            point["blocks"][:] = point["blocks"].mean(axis=0)
        with contextlib.redirect_stdout(io.StringIO()):
            first = bootstrap_comparison(perturbed, samples, (0.3, 1), 7)
            second = bootstrap_comparison(perturbed, samples, (0.3, 1), 7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(np.corrcoef(first[1].T)[0, 1], 1, places=10)
        np.testing.assert_allclose(first[1][:, 0] / 16**OMEGA,
                                   first[1][:, 1] / 32**OMEGA, atol=1e-13)

    def test_prepare_once_then_compare_without_clean_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean, perturbed = root / "clean", root / "perturbed"
            clean.mkdir()
            perturbed.mkdir()
            for i, (size, r) in enumerate((size, r) for size in [8, 16, 32, 64]
                                         for r in np.linspace(0.28, 1.02, 12)):
                u = 2.4 - 0.5*r + 0.05*(size/16)**(-OMEGA)
                save_point(clean / f"{i}_blocks.txt", make_point(size, r, u, i, 0.22+i*1e-5))
            for i, size in enumerate([8, 16, 24, 32]):
                u = 2.4 - 0.5*0.6 + 0.1*size**(-OMEGA)
                save_point(perturbed / f"{i}_blocks.txt", make_point(size, 0.6, u, 100+i))
            cache = root / "clean_reference.npz"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_main([str(clean), "--bootstrap", "5", "--output", str(cache)]), 0)
            ref = load_reference(cache)
            self.assertEqual(len(ref["clean_paths"]), 36)  # all L>=16, including edge points
            original_cache = cache.read_bytes()
            shutil.rmtree(clean)  # comparison must work with no clean blocks available
            prefix = root / "output" / "comparison"
            argv = ["--reference", str(cache), "--perturbed", str(perturbed),
                    "--output-prefix", str(prefix)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            self.assertEqual(cache.read_bytes(), original_cache)
            summary = Path(f"{prefix}_summary.txt").read_text()
            self.assertIn("clean points = 36; perturbed points = 4", summary)
            rows = np.loadtxt(f"{prefix}_points.txt")
            np.testing.assert_allclose(rows[:, 9], 0.1, atol=1e-9)
            self.assertTrue(np.all(rows[:, 10] > 0))
            with np.load(f"{prefix}_bootstrap.npz") as archive:
                self.assertEqual(archive["scaled_delta_U_covariance"].shape, (4, 4))
                self.assertEqual(archive["perturbed_pairs"].shape, (5, 4, 2))
                np.testing.assert_array_equal(archive["clean_coefficients"], ref["clean_coefficients"])
            for suffix in ("u_vs_rxi", "scaled_delta_u"):
                for extension in ("png", "pdf"):
                    self.assertGreater(Path(f"{prefix}_{suffix}.{extension}").stat().st_size, 1000)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv + ["--bootstrap", "6"]), 1)
                with self.assertRaises(SystemExit):
                    main(argv + ["--clean-lmin", "8"])
            self.assertEqual(len(load_points([perturbed, perturbed / "0_blocks.txt"])), 4)
            copy = perturbed / "copy_blocks.txt"
            copy.write_bytes((perturbed / "0_blocks.txt").read_bytes())
            with self.assertRaisesRegex(ValueError, "duplicate L,beta"):
                load_points([perturbed])
            copy.unlink()
            bad = perturbed / "0_blocks.txt"
            bad.write_text(bad.read_text().replace("# err_U =", "# old_err_U ="))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 1)
            for key, bad_value in (("clean_lmin", 8), ("omega", 0.8), ("degree_main", 5),
                                   ("r_min", 0.4), ("clean_coefficients", np.full((5, 11), np.nan))):
                np.savez(cache, **dict(ref, **{key: bad_value}))
                with self.assertRaises(ValueError):
                    load_reference(cache)


if __name__ == "__main__":
    unittest.main()
