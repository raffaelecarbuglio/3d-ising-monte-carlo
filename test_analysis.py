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
        0.5,
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
        abs_magnetization = np.array([0.0, 1.0])

        susceptibility = calculate_susceptibility(
            magnetization, abs_magnetization, 2, 0.5
        )

        self.assertAlmostEqual(susceptibility, 1.0)

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
        self.assertEqual(data.shape, (2, 7))
        self.assertEqual(data[1, 0], 1.0)

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
        self.assertAlmostEqual(observables["energy"], 2.5)
        self.assertAlmostEqual(errors["energy"], 1.0)

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

        optimized = block_jackknife(data, 6, 0.22, 2)
        brute_force = brute_force_block_jackknife(data, 6, 0.22, 2)

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

        with self.assertRaisesRegex(ValueError, "campione jackknife 1"):
            block_jackknife(data, 4, 0.2, 2)


if __name__ == "__main__":
    unittest.main()
