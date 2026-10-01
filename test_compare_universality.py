import contextlib
import io
import tempfile
import hashlib
import os
import unittest
from pathlib import Path

import numpy as np

from compare_universality import (
    bootstrap_comparison, evaluate,
    load_points, load_reference, main,
)
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
    return dict(L=size, beta=beta, blocks=means * (1 + perturbations),
                Rxi=r, U=u, err_Rxi=0.001, err_U=0.002)


def save_point(path, point):
    with open(path, "w", encoding="utf-8") as file:
        for key in ("L", "beta", "Rxi", "err_Rxi", "U", "err_U"):
            file.write(f"# {key} = {point[key]:.17g}\n")
        np.savetxt(file, point["blocks"])


class UniversalityTests(unittest.TestCase):
    def test_bundled_reference_matches_reported_94_point_fit(self):
        reference = load_reference(Path(__file__).with_name("clean_reference.npz"))
        self.assertEqual(reference["clean_coefficients"].shape, (5000, 11))
        self.assertEqual(len(reference["clean_paths"]), 94)
        self.assertEqual(int(reference["dof"]), 83)
        self.assertAlmostEqual(float(reference["chi2"]), 44.21632986812246, places=9)
        np.testing.assert_allclose(reference["coefficients"], [
            2.8142240796356073, 5.821679480578112, -43.522794929376033,
            96.305141468186577, -106.14348481524891, 59.508782036064538,
            -13.539418997071191, -0.26729992731527624, -1.8571338563591855,
            4.0541527302765035, -2.0487227861018917,
        ], rtol=1e-14)
        np.testing.assert_allclose(reference["clean_coefficients"].std(axis=0, ddof=1), [
            0.13317781919578847, 1.375910597466804, 5.732777575133909,
            12.346766076790198, 14.530829154097166, 8.8825210849564513,
            2.2084541531345865, 0.13582240992101161, 0.64589173965727353,
            0.96736284486109481, 0.46106917110815937,
        ], rtol=1e-14)
        self.assertLess(float(reference["r_range"][0]), 0.30)
        self.assertGreater(float(reference["r_range"][1]), 1.00)
        # Correlated intercept/slope shifts cancel at r=0.6.
        samples = np.zeros((5, 11))
        samples[:, 0] = np.arange(5) * 0.6
        samples[:, 1] = -np.arange(5)
        np.testing.assert_allclose([evaluate(c, 0.6) for c in samples], 0, atol=1e-15)

    def test_reads_saved_errors_without_recomputing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "saved_blocks.txt"
            point = make_point(16, 0.6, 1.8, 1)
            point.update(err_Rxi=0.12, err_U=0.34)
            save_point(path, point)
            loaded = load_points([path])[0]
            self.assertEqual(loaded["err_Rxi"], 0.12)
            self.assertEqual(loaded["err_U"], 0.34)
            point["err_U"] = -0.1
            save_point(path, point)
            with self.assertRaises(ValueError):
                load_points([path])

    def test_saved_reference_adds_cross_point_covariance(self):
        # Vary only the reference intercept, isolating its shared uncertainty.
        samples = np.zeros((30, 11))
        samples[:, 0] = np.linspace(1.99, 2.01, 30)
        perturbed = [make_point(16, 0.6, 2, 90), make_point(32, 0.6, 2, 91)]
        for point in perturbed:
            point["blocks"][:] = point["blocks"].mean(axis=0)
        with contextlib.redirect_stdout(io.StringIO()):
            first = bootstrap_comparison(perturbed, samples, 7)
            second = bootstrap_comparison(perturbed, samples, 7)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        self.assertAlmostEqual(np.corrcoef(first[1].T)[0, 1], 1, places=10)
        np.testing.assert_allclose(first[1][:, 0] / 16**OMEGA,
                                   first[1][:, 1] / 32**OMEGA, atol=1e-13)

    def test_compare_saved_reference_without_clean_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            perturbed = root / "perturbed"
            perturbed.mkdir()
            for i, size in enumerate([8, 16, 24, 32]):
                u = 2.4 - 0.5*0.6 + 0.1*size**(-OMEGA)
                save_point(perturbed / f"{i}_blocks.txt", make_point(size, 0.6, u, 100+i))
            cache = root / "clean_reference.npz"
            coefficients = np.zeros(11)
            coefficients[:2] = [2.4, -0.5]
            samples = np.tile(coefficients, (5, 1))
            samples[:, 0] += np.linspace(-0.001, 0.001, 5)
            np.savez(cache, coefficients=coefficients, clean_coefficients=samples,
                     r_range=[0.28, 1.02], clean_lmin=16, degree_main=6, degree_correction=3,
                     omega=OMEGA, basis="power", fit_method="vertical", chi2=1., dof=25,
                     seed=12345, clean_paths=[f"missing/{i}_blocks.txt" for i in range(36)],
                     clean_hashes=["0"*64]*36)
            ref = load_reference(cache)
            original_cache = cache.read_bytes()
            prefix = root / "output" / "comparison"
            argv = ["--reference", str(cache), "--perturbed", str(perturbed),
                    "--output-prefix", str(prefix)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 0)
            self.assertEqual(cache.read_bytes(), original_cache)
            self.assertFalse(Path(f"{prefix}_summary.txt").exists())
            self.assertFalse(Path(f"{prefix}_clean_curve.txt").exists())
            rows = np.loadtxt(f"{prefix}_points.txt")
            np.testing.assert_array_equal(rows[:, 3], np.full(4, 0.001))
            np.testing.assert_array_equal(rows[:, 5], np.full(4, 0.002))
            np.testing.assert_allclose(rows[:, 9], 0.1, atol=1e-9)
            self.assertTrue(np.all(rows[:, 10] > 0))
            with np.load(f"{prefix}_bootstrap.npz") as archive:
                self.assertEqual(archive["scaled_delta_U_covariance"].shape, (4, 4))
                self.assertEqual(archive["perturbed_pairs"].shape, (5, 4, 2))
                self.assertEqual(archive["clean_curve"].shape, (300, 4))
                self.assertEqual(len(archive["clean_paths"]), 36)
                self.assertEqual(int(archive["reference_seed"]), 12345)
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
            save_point(bad, make_point(8, 0.6, 2.1, 100))
            same_clean = dict(ref, clean_hashes=np.array(
                [hashlib.sha256(bad.read_bytes()).hexdigest()]*36))
            np.savez(cache, **same_clean)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(argv), 1)
            # Default reference is relative to the script, independent of cwd.
            previous = Path.cwd()
            try:
                os.chdir(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["--perturbed", str(perturbed), "--bootstrap", "5",
                                           "--output-prefix", str(root / "default")]), 0)
            finally:
                os.chdir(previous)
            for key, bad_value in (("clean_lmin", 8), ("omega", 0.8), ("degree_main", 5),
                                   ("basis", "chebyshev"), ("fit_method", "correlated"), ("clean_coefficients", np.full((5, 11), np.nan))):
                np.savez(cache, **dict(ref, **{key: bad_value}))
                with self.assertRaises(ValueError):
                    load_reference(cache)


if __name__ == "__main__":
    unittest.main()
