#!/usr/bin/env python3

import argparse
import gzip
import math
import sys

import numpy as np


ENERGY = 1
MAGNETIZATION = 2
ABS_MAGNETIZATION = 3
G_ZERO = 4
G_MIN = 5


def read_data_file(filename):
    lattice_size = None
    beta = None

    open_file = gzip.open if str(filename).endswith(".gz") else open
    with open_file(filename, "rt", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("# L ="):
                try:
                    lattice_size = int(line.split("=", 1)[1])
                except ValueError as error:
                    raise ValueError("valore di L non valido nell'intestazione") from error
            elif line.startswith("# beta ="):
                try:
                    beta = float(line.split("=", 1)[1])
                except ValueError as error:
                    raise ValueError(
                        "valore di beta non valido nell'intestazione"
                    ) from error

            if lattice_size is not None and beta is not None:
                break

    if lattice_size is None:
        raise ValueError("L non trovato nell'intestazione del file")
    if beta is None:
        raise ValueError("beta non trovato nell'intestazione del file")

    try:
        data = np.loadtxt(filename, comments="#", ndmin=2)
    except ValueError as error:
        raise ValueError("dati numerici non validi") from error

    if data.size == 0:
        raise ValueError("il file non contiene misure")
    if data.shape[1] == 4:
        volume = lattice_size**3
        energy = data[:, 1] / volume
        magnetization = data[:, 2] / volume
        g_zero = data[:, 2] ** 2 / volume
        # In memoria usiamo E/V, m=M/V, |m|, G(0) e G(p_min).
        data = np.column_stack((
            data[:, 0], energy, magnetization, np.abs(magnetization),
            g_zero, data[:, 3],
        ))
    elif data.shape[1] == 7:
        # I vecchi file restano leggibili: scartiamo solo la diagnostica.
        data = np.delete(data, 4, axis=1)
    else:
        raise ValueError(
            f"il file deve avere 4 o 7 colonne numeriche, ne ha {data.shape[1]}"
        )

    return lattice_size, beta, data


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


def calculate_susceptibility(magnetization, lattice_size):
    mean_m2 = np.mean(magnetization**2)

    return lattice_size**3 * mean_m2


def calculate_observables(data, lattice_size, beta):
    xi = calculate_xi(data[:, G_ZERO], data[:, G_MIN], lattice_size)

    return {
        # read_data_file restituisce E/V; riportiamo e = -(E/V)/3.
        "energy": -np.mean(data[:, ENERGY]) / 3.0,
        "magnetization": np.mean(data[:, MAGNETIZATION]),
        "abs_magnetization": np.mean(data[:, ABS_MAGNETIZATION]),
        "binder": calculate_binder(data[:, MAGNETIZATION]),
        "susceptibility": calculate_susceptibility(
            data[:, MAGNETIZATION],
            lattice_size,
        ),
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


def block_jackknife(data, lattice_size, beta, block_size):
    if block_size <= 0:
        raise ValueError("block-size deve essere un intero positivo")

    number_of_blocks = len(data) // block_size
    if number_of_blocks < 2:
        raise ValueError("servono almeno 2 blocchi completi per il jackknife")

    used_measurements = number_of_blocks * block_size
    excluded_measurements = len(data) - used_measurements
    used_data = data[:used_measurements]
    observables = calculate_observables(used_data, lattice_size, beta)

    blocks = used_data.reshape(number_of_blocks, block_size, 6)
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

    # Ogni elemento contiene la media con un blocco escluso.
    mean_energy = (
        total_energy_sum - block_energy_sums
    ) / remaining_measurements
    mean_abs_magnetization = (
        total_abs_magnetization_sum - block_abs_magnetization_sums
    ) / remaining_measurements
    mean_m2 = (total_m2_sum - block_m2_sums) / remaining_measurements
    mean_m4 = (total_m4_sum - block_m4_sums) / remaining_measurements
    mean_g_zero = (
        total_g_zero_sum - block_g_zero_sums
    ) / remaining_measurements
    mean_g_min = (
        total_g_min_sum - block_g_min_sums
    ) / remaining_measurements

    if np.any(mean_g_min <= 0.0):
        raise ValueError(
            "campione jackknife non valido: <g_min> deve essere positivo"
        )

    radicand = mean_g_zero / mean_g_min - 1.0
    if np.any(radicand < 0.0):
        raise ValueError(
            "campione jackknife non valido: <g_zero>/<g_min> - 1 e' negativo; "
            "la statistica puo' essere insufficiente"
        )
    if np.any(mean_m2 == 0.0):
        raise ValueError(
            "campione jackknife non valido: Binder non definito, <m^2> e' zero"
        )

    xi = np.sqrt(radicand) / (2.0 * math.sin(math.pi / lattice_size))
    errors = {
        "energy": jackknife_error(-mean_energy / 3.0),
        "abs_magnetization": jackknife_error(mean_abs_magnetization),
        "binder": jackknife_error(mean_m4 / mean_m2**2),
        "susceptibility": jackknife_error(lattice_size**3 * mean_m2),
        "xi": jackknife_error(xi),
        "r_xi": jackknife_error(xi / lattice_size),
    }

    return observables, errors, number_of_blocks, used_measurements, excluded_measurements


def calculate_block_averages(data, block_size):
    number_of_blocks = len(data) // block_size
    used_measurements = number_of_blocks * block_size
    used_data = data[:used_measurements]
    blocks = used_data.reshape(number_of_blocks, block_size, 6)

    return np.column_stack((
        np.mean(blocks[:, :, G_ZERO], axis=1),
        np.mean(blocks[:, :, G_MIN], axis=1),
        np.mean(blocks[:, :, MAGNETIZATION] ** 2, axis=1),
        np.mean(blocks[:, :, MAGNETIZATION] ** 4, axis=1),
    ))


def default_blocks_filename(filename):
    path = str(filename)
    if path.endswith(".gz"):
        path = path[:-3]
    if path.endswith(".dat"):
        path = path[:-4]
    return path + "_blocks.txt"


def save_blocks(
    filename,
    lattice_size,
    beta,
    block_size,
    block_averages,
    observables,
    errors,
):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# L = {lattice_size}\n")
        file.write(f"# beta = {beta:.17g}\n")
        file.write(f"# block_size = {block_size}\n")
        file.write(f"# number_of_blocks = {len(block_averages)}\n")
        file.write(f"# Rxi = {observables['r_xi']:.17g}\n")
        file.write(f"# err_Rxi = {errors['r_xi']:.17g}\n")
        file.write(f"# U = {observables['binder']:.17g}\n")
        file.write(f"# err_U = {errors['binder']:.17g}\n")
        file.write("# columns: g_zero g_min m2 m4\n")
        np.savetxt(file, block_averages, fmt="%.17g")


def print_results(
    filename,
    lattice_size,
    beta,
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
    print(f"beta = {beta:.10g}")
    print(f"measurements = {total_measurements}")
    print(f"block size = {block_size}")
    print(f"blocks = {number_of_blocks}")
    print(f"used measurements = {used_measurements}")
    if excluded_measurements > 0:
        print(f"excluded measurements = {excluded_measurements}")
    print()
    print(f"energy density e = {observables['energy']:.10g} +/- {errors['energy']:.3g}")
    print(
        f"susceptibility chi = {observables['susceptibility']:.10g} "
        f"+/- {errors['susceptibility']:.3g}"
    )
    print(f"Binder U = {observables['binder']:.10g} +/- {errors['binder']:.3g}")
    print(f"R_xi = {observables['r_xi']:.10g} +/- {errors['r_xi']:.3g}")


def save_summary(filename, lattice_size, beta, observables, errors):
    # Riusa i risultati del jackknife: non ricalcola le osservabili.
    with open(filename, "a", encoding="utf-8") as file:
        if file.tell() == 0:
            file.write("# L beta Rxi err_Rxi U err_U\n")
        file.write(
            f"{lattice_size} {beta:.17g} "
            f"{observables['r_xi']:.17g} {errors['r_xi']:.17g} "
            f"{observables['binder']:.17g} {errors['binder']:.17g}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Analizza le misure del modello di Ising")
    parser.add_argument("filename", help="file delle misure prodotto dal simulatore")
    parser.add_argument(
        "--summary-output",
        help="aggiunge i risultati al file di riepilogo indicato",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        required=True,
        help="numero di misure in ogni blocco jackknife",
    )
    parser.add_argument(
        "--blocks-output",
        help=(
            "file per le medie dei blocchi; se omesso usa "
            "<file_misure>_blocks.txt"
        ),
    )
    arguments = parser.parse_args()

    try:
        lattice_size, beta, data = read_data_file(arguments.filename)
        results = block_jackknife(
            data, lattice_size, beta, arguments.block_size
        )
        observables, errors, blocks, used, excluded = results
        block_averages = calculate_block_averages(
            data, arguments.block_size
        )
        blocks_output = (
            arguments.blocks_output
            if arguments.blocks_output is not None
            else default_blocks_filename(arguments.filename)
        )
        save_blocks(
            blocks_output,
            lattice_size,
            beta,
            arguments.block_size,
            block_averages,
            observables,
            errors,
        )
        if arguments.summary_output is not None:
            save_summary(
                arguments.summary_output, lattice_size, beta, observables, errors
            )
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"block averages: {blocks_output}")
    print()
    print_results(
        arguments.filename,
        lattice_size,
        beta,
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
