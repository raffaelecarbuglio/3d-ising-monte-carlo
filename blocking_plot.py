#!/usr/bin/env python3

import argparse
import sys

import matplotlib.pyplot as plt

from analyze import block_jackknife, read_data_file


def main():
    parser = argparse.ArgumentParser(
        description="Mostra l'errore stimato al variare della dimensione dei blocchi"
    )
    parser.add_argument("filename", help="file delle misure prodotto dal simulatore")
    arguments = parser.parse_args()

    try:
        lattice_size, beta, data = read_data_file(arguments.filename)

        block_sizes = []
        energy_errors = []
        chi_errors = []
        binder_errors = []
        r_xi_errors = []

        block_size = 1
        while len(data) // block_size >= 20:
            results = block_jackknife(data, lattice_size, beta, block_size)
            _, errors, _, _, _ = results

            block_sizes.append(block_size)
            energy_errors.append(errors["energy"])
            chi_errors.append(errors["susceptibility"])
            binder_errors.append(errors["binder"])
            r_xi_errors.append(errors["r_xi"])

            block_size *= 2
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(block_sizes, energy_errors, "o-")
    axes[0, 0].set_title("Energy density e")
    axes[0, 0].set_xlabel("Block size")
    axes[0, 0].set_ylabel("Estimated error")
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_xticks(block_sizes)
    axes[0, 0].set_xticklabels(block_sizes)
    axes[0, 0].grid(True)

    axes[0, 1].plot(block_sizes, chi_errors, "o-")
    axes[0, 1].set_title("Susceptibility chi")
    axes[0, 1].set_xlabel("Block size")
    axes[0, 1].set_ylabel("Estimated error")
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_xticks(block_sizes)
    axes[0, 1].set_xticklabels(block_sizes)
    axes[0, 1].grid(True)

    axes[1, 0].plot(block_sizes, binder_errors, "o-")
    axes[1, 0].set_title("Binder U")
    axes[1, 0].set_xlabel("Block size")
    axes[1, 0].set_ylabel("Estimated error")
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_xticks(block_sizes)
    axes[1, 0].set_xticklabels(block_sizes)
    axes[1, 0].grid(True)

    axes[1, 1].plot(block_sizes, r_xi_errors, "o-")
    axes[1, 1].set_title("R_xi")
    axes[1, 1].set_xlabel("Block size")
    axes[1, 1].set_ylabel("Estimated error")
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(block_sizes)
    axes[1, 1].set_xticklabels(block_sizes)
    axes[1, 1].grid(True)

    figure.tight_layout()
    figure.savefig("blocking_plateau.png")
    plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
