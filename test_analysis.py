import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze import block_jackknife, calculate_binder, calculate_xi, read_data_file


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


class AnalysisTests(unittest.TestCase):
    def test_binder(self):
        magnetization = np.array([1.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(calculate_binder(magnetization), 1.36)

    def test_correlation_length(self):
        g_zero = np.array([5.0, 5.0])
        g_min = np.array([1.0, 1.0])
        self.assertAlmostEqual(calculate_xi(g_zero, g_min, 4), math.sqrt(2.0))

    def test_comments_in_the_middle_of_data_file(self):
        contents = """# Ising data
# L = 4
# columns
0 -1.0 0.5 0.5 0.4 4.0 1.0
# restart information
1 -2.0 -0.5 0.5 0.6 4.0 1.0
"""
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "data.dat"
            filename.write_text(contents, encoding="utf-8")
            lattice_size, data = read_data_file(filename)

        self.assertEqual(lattice_size, 4)
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

        observables, errors, blocks, used, excluded = block_jackknife(data, 4, 2)

        self.assertEqual(blocks, 2)
        self.assertEqual(used, 4)
        self.assertEqual(excluded, 1)
        self.assertAlmostEqual(observables["energy"], 2.5)
        self.assertAlmostEqual(errors["energy"], 1.0)

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
            block_jackknife(data, 4, 2)


if __name__ == "__main__":
    unittest.main()
