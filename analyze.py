#!/usr/bin/env python3

import argparse
import math
import sys

import numpy as np


ENERGY = 1
MAGNETIZATION = 2
ABS_MAGNETIZATION = 3
G_ZERO = 5
G_MIN = 6


def read_data_file(filename):
    lattice_size = None

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("# L ="):
                try:
                    lattice_size = int(line.split("=", 1)[1])
                except ValueError as error:
                    raise ValueError("valore di L non valido nell'intestazione") from error

    if lattice_size is None:
        raise ValueError("L non trovato nell'intestazione del file")

    try:
        data = np.loadtxt(filename, comments="#", ndmin=2)
    except ValueError as error:
        raise ValueError("dati numerici non validi") from error

    if data.size == 0:
        raise ValueError("il file non contiene misure")
    if data.shape[1] != 7:
        raise ValueError(
            f"il file deve avere 7 colonne numeriche, ne ha {data.shape[1]}"
        )

    return lattice_size, data


def calculate_binder(magnetization):
    mean_m2 = np.mean(magnetization**2)
    mean_m4 = np.mean(magnetization**4)

    if mean_m2 == 0.0:
        raise ValueError("Binder non definito: <m^2> e' zero")

    return mean_m4 / (mean_m2**2)


def calculate_xi(g_zero, g_min, lattice_size):
    mean_g_zero = np.mean(g_zero)
    mean_g_min = np.mean(g_min)

    if mean_g_min <= 0.0:
        raise ValueError("xi non definita: <g_min> deve essere positivo")

    radicand = mean_g_zero / mean_g_min - 1.0
    if radicand < 0.0:
        raise ValueError(
            "xi non valida: <g_zero>/<g_min> - 1 e' negativo; "
            "la statistica puo' essere insufficiente"
        )

    return math.sqrt(radicand) / (2.0 * math.sin(math.pi / lattice_size))


def calculate_observables(data, lattice_size):
    xi = calculate_xi(data[:, G_ZERO], data[:, G_MIN], lattice_size)

    return {
        "energy": np.mean(data[:, ENERGY]),
        "magnetization": np.mean(data[:, MAGNETIZATION]),
        "abs_magnetization": np.mean(data[:, ABS_MAGNETIZATION]),
        "binder": calculate_binder(data[:, MAGNETIZATION]),
        "g_zero": np.mean(data[:, G_ZERO]),
        "g_min": np.mean(data[:, G_MIN]),
        "xi": xi,
        "r_xi": xi / lattice_size,
    }


def jackknife_error(values):
    values = np.asarray(values)
    mean_value = np.mean(values)
    number_of_blocks = len(values)

    return math.sqrt(
        (number_of_blocks - 1)
        / number_of_blocks
        * np.sum((values - mean_value) ** 2)
    )


def block_jackknife(data, lattice_size, block_size):
    if block_size <= 0:
        raise ValueError("block-size deve essere un intero positivo")

    number_of_blocks = len(data) // block_size
    if number_of_blocks < 2:
        raise ValueError("servono almeno 2 blocchi completi per il jackknife")

    used_measurements = number_of_blocks * block_size
    excluded_measurements = len(data) - used_measurements
    used_data = data[:used_measurements]
    observables = calculate_observables(used_data, lattice_size)

    blocks = used_data.reshape(number_of_blocks, block_size, 7)
    block_energy_sums = np.sum(blocks[:, :, ENERGY], axis=1)
    block_abs_magnetization_sums = np.sum(
        blocks[:, :, ABS_MAGNETIZATION], axis=1
    )
    block_m2_sums = np.sum(blocks[:, :, MAGNETIZATION] ** 2, axis=1)
    block_m4_sums = np.sum(blocks[:, :, MAGNETIZATION] ** 4, axis=1)
    block_g_zero_sums = np.sum(blocks[:, :, G_ZERO], axis=1)
    block_g_min_sums = np.sum(blocks[:, :, G_MIN], axis=1)

    total_energy_sum = np.sum(block_energy_sums)
    total_abs_magnetization_sum = np.sum(block_abs_magnetization_sums)
    total_m2_sum = np.sum(block_m2_sums)
    total_m4_sum = np.sum(block_m4_sums)
    total_g_zero_sum = np.sum(block_g_zero_sums)
    total_g_min_sum = np.sum(block_g_min_sums)
    remaining_measurements = used_measurements - block_size

    names_with_error = [
        "energy",
        "abs_magnetization",
        "binder",
        "xi",
        "r_xi",
    ]
    jackknife_values = {name: [] for name in names_with_error}

    for block in range(number_of_blocks):
        mean_energy = (
            total_energy_sum - block_energy_sums[block]
        ) / remaining_measurements
        mean_abs_magnetization = (
            total_abs_magnetization_sum - block_abs_magnetization_sums[block]
        ) / remaining_measurements
        mean_m2 = (total_m2_sum - block_m2_sums[block]) / remaining_measurements
        mean_m4 = (total_m4_sum - block_m4_sums[block]) / remaining_measurements
        mean_g_zero = (
            total_g_zero_sum - block_g_zero_sums[block]
        ) / remaining_measurements
        mean_g_min = (
            total_g_min_sum - block_g_min_sums[block]
        ) / remaining_measurements

        try:
            xi = calculate_xi([mean_g_zero], [mean_g_min], lattice_size)
            if mean_m2 == 0.0:
                raise ValueError("Binder non definito: <m^2> e' zero")
        except ValueError as error:
            raise ValueError(
                f"campione jackknife {block + 1} non valido: {error}"
            ) from error

        jackknife_values["energy"].append(mean_energy)
        jackknife_values["abs_magnetization"].append(mean_abs_magnetization)
        jackknife_values["binder"].append(mean_m4 / mean_m2**2)
        jackknife_values["xi"].append(xi)
        jackknife_values["r_xi"].append(xi / lattice_size)

    errors = {
        name: jackknife_error(jackknife_values[name]) for name in names_with_error
    }

    return observables, errors, number_of_blocks, used_measurements, excluded_measurements


def print_results(
    filename,
    lattice_size,
    total_measurements,
    block_size,
    number_of_blocks,
    used_measurements,
    excluded_measurements,
    observables,
    errors,
):
    print(f"file: {filename}")
    print(f"L = {lattice_size}")
    print(f"measurements = {total_measurements}")
    print(f"block size = {block_size}")
    print(f"blocks = {number_of_blocks}")
    print(f"used measurements = {used_measurements}")
    if excluded_measurements > 0:
        print(f"excluded measurements = {excluded_measurements}")
    print()
    print(f"energy/spin = {observables['energy']:.10g} +/- {errors['energy']:.3g}")
    print(f"<m> = {observables['magnetization']:.10g}")
    print(
        f"<|m|> = {observables['abs_magnetization']:.10g} "
        f"+/- {errors['abs_magnetization']:.3g}"
    )
    print(f"Binder U = {observables['binder']:.10g} +/- {errors['binder']:.3g}")
    print(f"<g_zero> = {observables['g_zero']:.10g}")
    print(f"<g_min> = {observables['g_min']:.10g}")
    print(f"xi = {observables['xi']:.10g} +/- {errors['xi']:.3g}")
    print(f"R_xi = {observables['r_xi']:.10g} +/- {errors['r_xi']:.3g}")


def main():
    parser = argparse.ArgumentParser(description="Analizza le misure del modello di Ising")
    parser.add_argument("filename", help="file delle misure prodotto dal simulatore")
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="numero di misure in ogni blocco jackknife",
    )
    arguments = parser.parse_args()

    try:
        lattice_size, data = read_data_file(arguments.filename)
        results = block_jackknife(data, lattice_size, arguments.block_size)
        observables, errors, blocks, used, excluded = results
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print_results(
        arguments.filename,
        lattice_size,
        len(data),
        arguments.block_size,
        blocks,
        used,
        excluded,
        observables,
        errors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
