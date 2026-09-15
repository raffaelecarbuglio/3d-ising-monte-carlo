import gzip
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze import (
    block_jackknife,
    calculate_binder,
    calculate_observables,
    calculate_susceptibility,
    calculate_xi,
    jackknife_error,
    read_data_file,
)


def make_row(sweep, energy, magnetization, g_zero=4.0, g_min=1.0):
    return [
        sweep,
        energy,
        magnetization,
        abs(magnetization),
        g_zero,
        g_min,
    ]


def brute_force_block_jackknife(data, lattice_size, beta, block_size):
    number_of_blocks = len(data) // block_size
    used_measurements = number_of_blocks * block_size
    excluded_measurements = len(data) - used_measurements
    used_data = data[:used_measurements]
    observables = calculate_observables(used_data, lattice_size, beta)

    names = [
        "energy",
        "abs_magnetization",
        "binder",
        "susceptibility",
        "xi",
        "r_xi",
    ]
    jackknife_values = {name: [] for name in names}

    for block in range(number_of_blocks):
        first = block * block_size
        last = first + block_size
        sample = np.concatenate((used_data[:first], used_data[last:]))
        sample_observables = calculate_observables(sample, lattice_size, beta)

        for name in names:
            jackknife_values[name].append(sample_observables[name])

    errors = {name: jackknife_error(jackknife_values[name]) for name in names}

    return observables, errors, number_of_blocks, used_measurements, excluded_measurements


class AnalysisTests(unittest.TestCase):
    def test_binder(self):
        magnetization = np.array([1.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(calculate_binder(magnetization), 1.36)

    def test_correlation_length(self):
        g_zero = np.array([5.0, 5.0])
        g_min = np.array([1.0, 1.0])
        self.assertAlmostEqual(calculate_xi(g_zero, g_min, 4), math.sqrt(2.0))

    def test_susceptibility(self):
        magnetization = np.array([0.0, 1.0])
        susceptibility = calculate_susceptibility(magnetization, 2)

        self.assertAlmostEqual(susceptibility, 4.0)

    def test_susceptibility_matches_g_zero_with_incomplete_tail(self):
        data = np.array([
            make_row(0, -1.0, 0.5, 2.0, 1.0),
            make_row(1, -1.0, -0.5, 2.0, 1.0),
            make_row(2, -2.0, 1.0, 8.0, 1.0),
            make_row(3, -2.0, -1.0, 8.0, 1.0),
            make_row(4, -1.0, 0.5, 2.0, 1.0),
        ])
        observables, errors, _, used, excluded = block_jackknife(data, 2, 0.5, 2)

        self.assertEqual(excluded, 1)
        self.assertEqual(observables["susceptibility"], 5.0)
        self.assertEqual(observables["susceptibility"], np.mean(data[:used, 4]))
        self.assertEqual(errors["susceptibility"], 3.0)

    def test_comments_in_the_middle_of_data_file(self):
        contents = """# Ising data
# L = 4
# beta = 0.22
# columns
0 -1.0 0.5 0.5 0.4 4.0 1.0
# restart information
1 -2.0 -0.5 0.5 0.6 4.0 1.0
"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "data.dat"
            filename.write_text(contents, encoding="utf-8")
            lattice_size, beta, data = read_data_file(filename)

        self.assertEqual(lattice_size, 4)
        self.assertEqual(beta, 0.22)
        self.assertEqual(data.shape, (2, 6))
        self.assertEqual(data[1, 0], 1.0)

    def test_compact_and_legacy_files_give_same_analysis(self):
        header = "# L = 4\n# beta = 0.22\n"
        compact = header + "1 -64 32 1\n2 -128 -48 2\n3 -96 40 1.5\n4 -64 -32 1\n"
        legacy = header + (
            "1 -1 0.5 0.5 0.4 16 1\n"
            "2 -2 -0.75 0.75 0.6 36 2\n"
            "3 -1.5 0.625 0.625 0.5 25 1.5\n"
            "4 -1 -0.5 0.5 0.4 16 1\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            old_file = Path(directory) / "old.dat"
            old_file.write_text(legacy, encoding="utf-8")
            size, beta, old_data = read_data_file(old_file)
            expected = block_jackknife(old_data, size, beta, 2)
            for contents in (compact, legacy):
                for suffix in (".dat", ".dat.gz"):
                    filename = Path(directory) / ("measurements" + suffix)
                    open_file = gzip.open if suffix.endswith(".gz") else open
                    with open_file(filename, "wt", encoding="utf-8") as file:
                        file.write(contents)
                    size, beta, data = read_data_file(filename)
                    np.testing.assert_array_equal(data, old_data)
                    self.assertEqual(block_jackknife(data, size, beta, 2), expected)
            self.assertEqual(expected[0]["energy"], 1.375 / 3)
            self.assertEqual(expected[0]["susceptibility"], 23.25)

    def test_block_jackknife_and_incomplete_tail(self):
        data = np.array(
            [
                make_row(0, 1.0, 1.0),
                make_row(1, 2.0, -1.0),
                make_row(2, 3.0, 1.0),
                make_row(3, 4.0, -1.0),
                make_row(4, 100.0, 1.0),
            ]
        )

        observables, errors, blocks, used, excluded = block_jackknife(
            data, 4, 0.2, 2
        )

        self.assertEqual(blocks, 2)
        self.assertEqual(used, 4)
        self.assertEqual(excluded, 1)
        self.assertAlmostEqual(observables["energy"], -2.5 / 3.0)
        self.assertAlmostEqual(errors["energy"], 1.0 / 3.0)

    def test_optimized_jackknife_matches_brute_force(self):
        data = np.array(
            [
                make_row(0, -2.0, 0.8, 8.0, 1.0),
                make_row(1, -1.8, -0.6, 7.0, 1.2),
                make_row(2, -1.5, 0.4, 6.0, 1.1),
                make_row(3, -1.2, -0.3, 5.0, 1.3),
                make_row(4, -1.0, 0.2, 4.0, 1.0),
                make_row(5, -0.8, -0.1, 3.0, 0.9),
                make_row(6, 100.0, 1.0, 1.0, 2.0),
            ]
        )

        for block_size in (1, 2, 3):
            with self.subTest(block_size=block_size):
                optimized = block_jackknife(data, 6, 0.22, block_size)
                brute_force = brute_force_block_jackknife(data, 6, 0.22, block_size)

                for name in optimized[0]:
                    self.assertAlmostEqual(optimized[0][name], brute_force[0][name])
                for name in optimized[1]:
                    self.assertAlmostEqual(optimized[1][name], brute_force[1][name])
                self.assertEqual(optimized[2:], brute_force[2:])

    def test_negative_xi_radicand_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "negativo"):
            calculate_xi(np.array([1.0]), np.array([2.0]), 4)

    def test_invalid_jackknife_sample_is_an_error(self):
        data = np.array(
            [
                make_row(0, 1.0, 1.0, 4.0, 1.0),
                make_row(1, 1.0, -1.0, 4.0, 1.0),
                make_row(2, 1.0, 1.0, 1.0, 2.0),
                make_row(3, 1.0, -1.0, 1.0, 2.0),
            ]
        )

        with self.assertRaisesRegex(ValueError, "campione jackknife.*negativo"):
            block_jackknife(data, 4, 0.2, 2)

    def test_nonpositive_jackknife_g_min_is_an_error(self):
        for g_min in (0.0, -1.0):
            with self.subTest(g_min=g_min):
                data = np.array([
                    make_row(0, -1.0, 0.5, 4.0, 2.0),
                    make_row(1, -1.0, 0.5, 4.0, g_min),
                ])
                with self.assertRaisesRegex(ValueError, "campione jackknife.*positivo"):
                    block_jackknife(data, 4, 0.2, 1)

    def test_zero_jackknife_m2_is_an_error(self):
        data = np.array([
            make_row(0, -1.0, 0.5),
            make_row(1, -1.0, 0.0),
        ])
        with self.assertRaisesRegex(ValueError, "campione jackknife.*Binder.*zero"):
            block_jackknife(data, 4, 0.2, 1)


if __name__ == "__main__":
    unittest.main()
