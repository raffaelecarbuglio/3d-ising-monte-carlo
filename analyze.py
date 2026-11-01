#!/usr/bin/env python3

import argparse
import gzip
from itertools import islice
import math
import sys

import numpy as np


ENERGY = 1
MAGNETIZATION = 2
ABS_MAGNETIZATION = 3
G_ZERO = 4
G_MIN = 5


def read_header(filename):
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

    return lattice_size, beta


def read_perturbation_header(filename):
    # Conserva la descrizione della dinamica nei risultati dell'analisi.
    keys = ("sigma", "lambda", "noise_mode", "q4", "q8", "q12", "u", "v")
    lines = []
    open_file = gzip.open if str(filename).endswith(".gz") else open
    with open_file(filename, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                break
            if any(line.startswith(f"# {key} =") for key in keys):
                lines.append(line.rstrip() + "\n")
    return "".join(lines)


def normalize_data(data, lattice_size):
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

    return data


def read_data_file(filename):
    lattice_size, beta = read_header(filename)
    try:
        data = np.loadtxt(filename, comments="#", ndmin=2)
    except ValueError as error:
        raise ValueError("dati numerici non validi") from error
    return lattice_size, beta, normalize_data(data, lattice_size)


def read_data_chunks(filename, lattice_size, chunk_rows):
    open_file = gzip.open if str(filename).endswith(".gz") else open
    columns = None
    with open_file(filename, "rt", encoding="utf-8") as file:
        # Conta misure, non righe di commento o righe vuote.
        rows = (line for line in file
                if not line.isspace() and not line.lstrip().startswith("#"))
        while True:
            lines = list(islice(rows, chunk_rows))
            if not lines:
                break
            try:
                data = np.loadtxt(lines, comments="#", ndmin=2)
            except ValueError as error:
                raise ValueError("dati numerici non validi") from error
            del lines
            if columns is not None and data.shape[1] != columns:
                raise ValueError("numero di colonne cambiato nel file delle misure")
            columns = data.shape[1]
            yield normalize_data(data, lattice_size)


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


def calculate_block_sums(data, block_size):
    number_of_blocks = len(data) // block_size
    blocks = data[:number_of_blocks * block_size].reshape(number_of_blocks, block_size, 6)
    # Colonne: E/V, m, |m|, m^2, m^4, G(0), G(p_min).
    return np.column_stack((
        np.sum(blocks[:, :, ENERGY], axis=1),
        np.sum(blocks[:, :, MAGNETIZATION], axis=1),
        np.sum(blocks[:, :, ABS_MAGNETIZATION], axis=1),
        np.sum(blocks[:, :, MAGNETIZATION] ** 2, axis=1),
        np.sum(blocks[:, :, MAGNETIZATION] ** 4, axis=1),
        np.sum(blocks[:, :, G_ZERO], axis=1),
        np.sum(blocks[:, :, G_MIN], axis=1),
    ))


def observables_from_means(means, lattice_size):
    """Convert moment means into observables, for one sample or many samples."""
    energy, magnetization, abs_m, m2, m4, g_zero, g_min = means.T
    if not np.isfinite(means).all():
        raise ValueError("osservabili non finite")
    if np.any(g_min <= 0):
        raise ValueError("<g_min> deve essere positivo")
    if np.any(g_zero < g_min):
        raise ValueError("xi non valida: <g_zero>/<g_min> - 1 e' negativo")
    if np.any(m2 <= 0):
        raise ValueError("Binder non definito: <m^2> e' zero o negativo")
    xi = np.sqrt(g_zero / g_min - 1) / (2 * np.sin(np.pi / lattice_size))
    return dict(energy=-energy/3, magnetization=magnetization,
                abs_magnetization=abs_m, binder=m4/m2**2,
                susceptibility=lattice_size**3*m2, g_zero=g_zero, g_min=g_min,
                xi=xi, r_xi=xi/lattice_size)


def jackknife_from_block_sums(block_sums, lattice_size, block_size,
                             total_measurements, observables=None):
    number_of_blocks = len(block_sums)
    if number_of_blocks < 2:
        raise ValueError("servono almeno 2 blocchi completi per il jackknife")
    used = number_of_blocks * block_size
    totals = block_sums.sum(axis=0)
    if observables is None:
        observables = observables_from_means(totals / used, lattice_size)
    # Each row omits one block; reuse the same observable formulas.
    means = (totals - block_sums) / (used - block_size)
    try:
        samples = observables_from_means(means, lattice_size)
    except ValueError as error:
        raise ValueError(f"campione jackknife non valido: {error}") from error
    errors = {key: jackknife_error(samples[key]) for key in
              ("energy", "abs_magnetization", "binder", "susceptibility", "xi", "r_xi")}
    return observables, errors, number_of_blocks, used, total_measurements-used


def block_jackknife(data, lattice_size, beta, block_size):
    if block_size <= 0:
        raise ValueError("block-size deve essere un intero positivo")
    used = (len(data) // block_size) * block_size
    if used < 2 * block_size:
        raise ValueError("servono almeno 2 blocchi completi per il jackknife")
    observables = calculate_observables(data[:used], lattice_size, beta)
    return jackknife_from_block_sums(
        calculate_block_sums(data, block_size), lattice_size, block_size,
        len(data), observables,
    )


def read_block_sums(filename, block_size, chunk_rows=100_000):
    if block_size <= 0:
        raise ValueError("block-size deve essere un intero positivo")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows deve essere un intero positivo")
    lattice_size, beta = read_header(filename)
    # Ogni lettura contiene blocchi interi: nessun blocco viene spezzato.
    # Se un singolo blocco supera il target, leggiamo un blocco alla volta.
    chunk_rows = max(block_size, (chunk_rows // block_size) * block_size)
    sums = []
    total = 0
    for data in read_data_chunks(filename, lattice_size, chunk_rows):
        total += len(data)
        if len(data) >= block_size:
            sums.append(calculate_block_sums(data, block_size))
        del data
    if total == 0:
        raise ValueError("il file non contiene misure")
    if not sums:
        raise ValueError("servono almeno 2 blocchi completi per il jackknife")
    block_sums = np.concatenate(sums)
    del sums
    return lattice_size, beta, total, block_sums


def analyze_streaming(filename, block_size, chunk_rows=100_000):
    lattice_size, beta, total, block_sums = read_block_sums(filename, block_size, chunk_rows)
    results = jackknife_from_block_sums(block_sums, lattice_size, block_size, total)
    averages = block_sums[:, [5, 6, 3, 4]] / block_size
    return lattice_size, beta, total, results, averages


def analyze_replicas(filenames, block_size):
    """Pool complete blocks, never joining a block across two chains."""
    identity = read_header(filenames[0])
    perturbation = read_perturbation_header(filenames[0])
    sums, total, seeds, dynamics = [], 0, set(), None
    for filename in filenames:
        if read_header(filename) != identity or read_perturbation_header(filename) != perturbation:
            raise ValueError("replicas must have the same L, beta and perturbation")
        opener = gzip.open if str(filename).endswith(".gz") else open
        with opener(filename, "rt") as file:
            header = {}
            for line in file:
                if not line.startswith("#"):
                    break
                if " = " in line:
                    key, value = line[2:].strip().split(" = ", 1)
                    header[key] = value
        current = (header.get("algorithm"), header.get("measure_every"))
        seed = header.get("seed")
        if seed is None or seed in seeds or (dynamics is not None and current != dynamics):
            raise ValueError("replicas need distinct seeds and matching dynamics/sampling")
        seeds.add(seed)
        dynamics = current
        _, _, n, blocks = read_block_sums(filename, block_size)
        if len(blocks) < 2:
            raise ValueError("each replica needs at least two complete blocks")
        sums.append(blocks)
        total += n
    means = np.concatenate(sums) / block_size
    L, beta = identity

    central = observables_from_means(means.mean(0), L)
    rng = np.random.default_rng(20261101)
    # Same indices for every moment preserve the U/Rxi covariance.
    draws = np.array([means[rng.integers(len(means), size=len(means))].mean(0)
                      for _ in range(3000)])
    errors = {key: np.std(value, ddof=1) for key, value in observables_from_means(draws, L).items()}
    used = len(means) * block_size
    results = central, errors, len(means), used, total-used
    return L, beta, total, results, means[:, [5, 6, 3, 4]]


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
    perturbation_header="",
):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(f"# L = {lattice_size}\n")
        file.write(f"# beta = {beta:.17g}\n")
        file.write(perturbation_header)
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


def save_summary(filename, lattice_size, beta, observables, errors, perturbation_header=""):
    # Riusa i risultati del jackknife: non ricalcola le osservabili.
    with open(filename, "a", encoding="utf-8") as file:
        if file.tell() == 0:
            file.write("# L beta Rxi err_Rxi U err_U\n")
        file.write(perturbation_header)
        file.write(
            f"{lattice_size} {beta:.17g} "
            f"{observables['r_xi']:.17g} {errors['r_xi']:.17g} "
            f"{observables['binder']:.17g} {errors['binder']:.17g}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="Analizza le misure del modello di Ising")
    parser.add_argument("filename", nargs="+", help="file delle misure prodotto dal simulatore")
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
        filenames = arguments.filename
        perturbation_header = read_perturbation_header(filenames[0])
        if len(filenames) > 1:
            perturbation_header += (f"# replicas = {len(filenames)}\n"
                                    "# errors = paired block bootstrap, 3000 resamples, seed 20261101\n")
            lattice_size, beta, total, results, block_averages = analyze_replicas(filenames, arguments.block_size)
        else:
            lattice_size, beta, total, results, block_averages = analyze_streaming(filenames[0], arguments.block_size)
        observables, errors, blocks, used, excluded = results
        blocks_output = (
            arguments.blocks_output
            if arguments.blocks_output is not None
            else default_blocks_filename(filenames[0])
        )
        save_blocks(
            blocks_output,
            lattice_size,
            beta,
            arguments.block_size,
            block_averages,
            observables,
            errors,
            perturbation_header,
        )
        if arguments.summary_output is not None:
            save_summary(
                arguments.summary_output, lattice_size, beta, observables, errors,
                perturbation_header,
            )
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"block averages: {blocks_output}")
    print(perturbation_header, end="")
    print()
    print_results(
        ", ".join(filenames),
        lattice_size,
        beta,
        total,
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
