#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import numpy as np


OMEGA = 0.8295  # 3D Ising, M. Reehorst, JHEP 09 (2022) 177


def collect_block_files(inputs):
    files = []

    for name in inputs:
        path = Path(name)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*_blocks.txt")))
        else:
            raise ValueError(f"'{name}' non esiste")

    unique_files = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    if not unique_files:
        raise ValueError("nessun file *_blocks.txt trovato")

    return unique_files


def read_block_file(filename):
    header = {}

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                continue
            text = line[1:].strip()
            if "=" in text:
                key, value = text.split("=", 1)
                header[key.strip()] = value.strip()

    required = [
        "L", "beta", "block_size", "number_of_blocks",
        "Rxi", "err_Rxi", "U", "err_U",
    ]
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(
            f"{filename}: intestazione incompleta, mancano {', '.join(missing)}"
        )

    try:
        lattice_size = int(header["L"])
        beta = float(header["beta"])
        block_size = int(header["block_size"])
        number_of_blocks = int(header["number_of_blocks"])
        r_xi = float(header["Rxi"])
        err_r_xi = float(header["err_Rxi"])
        binder = float(header["U"])
        err_binder = float(header["err_U"])
        blocks = np.loadtxt(filename, comments="#", ndmin=2)
    except ValueError as error:
        raise ValueError(f"{filename}: valori non validi") from error

    if lattice_size <= 0 or block_size <= 0 or number_of_blocks < 2:
        raise ValueError(f"{filename}: metadati dei blocchi non validi")
    if blocks.shape[1] != 4:
        raise ValueError(
            f"{filename}: servono 4 colonne: g_zero g_min m2 m4"
        )
    if len(blocks) != number_of_blocks:
        raise ValueError(
            f"{filename}: trovati {len(blocks)} blocchi, "
            f"attesi {number_of_blocks}"
        )
    if not np.all(np.isfinite(blocks)):
        raise ValueError(f"{filename}: il file contiene valori non finiti")
    if err_r_xi < 0.0 or err_binder <= 0.0:
        raise ValueError(f"{filename}: errori non validi")

    return {
        "filename": str(filename),
        "L": lattice_size,
        "beta": beta,
        "block_size": block_size,
        "r_xi": r_xi,
        "err_r_xi": err_r_xi,
        "binder": binder,
        "err_binder": err_binder,
        "blocks": blocks,
    }


def load_points(block_files):
    points = [read_block_file(filename) for filename in block_files]
    seen = set()

    for point in points:
        key = (point["L"], point["beta"])
        if key in seen:
            raise ValueError(
                f"punto duplicato: L={key[0]}, beta={key[1]:.17g}"
            )
        seen.add(key)

    return points


def select_points(points, sizes, r_min, r_max):
    if r_min >= r_max:
        raise ValueError("r-min deve essere minore di r-max")

    available_sizes = sorted({point["L"] for point in points})
    selected_sizes = (
        available_sizes if sizes is None else sorted(set(sizes))
    )

    missing_sizes = [
        size for size in selected_sizes if size not in available_sizes
    ]
    if missing_sizes:
        raise ValueError(
            "taglie richieste non presenti: "
            + " ".join(str(size) for size in missing_sizes)
        )

    selected = [
        point
        for point in points
        if point["L"] in selected_sizes
        and r_min <= point["r_xi"] <= r_max
    ]
    selected.sort(key=lambda point: (point["L"], point["r_xi"]))

    if not selected:
        raise ValueError("nessun punto nel range richiesto")

    empty_sizes = [
        size
        for size in selected_sizes
        if not any(point["L"] == size for point in selected)
    ]
    if empty_sizes:
        raise ValueError(
            "nessun punto nel range per L="
            + " ".join(str(size) for size in empty_sizes)
        )

    return selected, selected_sizes


def design_matrix(r_xi, lattice_sizes, degree_main, degree_correction):
    r_xi = np.asarray(r_xi, dtype=float)
    lattice_sizes = np.asarray(lattice_sizes, dtype=float)

    columns = [
        r_xi**degree for degree in range(degree_main + 1)
    ]
    correction = lattice_sizes ** (-OMEGA)
    columns.extend(
        correction * r_xi**degree
        for degree in range(degree_correction + 1)
    )
    return np.column_stack(columns)


def fit_polynomials(
    r_xi,
    binder,
    err_binder,
    lattice_sizes,
    degree_main,
    degree_correction,
):
    matrix = design_matrix(
        r_xi, lattice_sizes, degree_main, degree_correction
    )
    weights = 1.0 / np.asarray(err_binder, dtype=float)
    weighted_matrix = matrix * weights[:, None]
    weighted_binder = np.asarray(binder, dtype=float) * weights

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_matrix, weighted_binder, rcond=None
    )
    if rank != matrix.shape[1]:
        raise ValueError(
            "fit singolare: ridurre il grado dei polinomi o aggiungere punti"
        )

    residuals = np.asarray(binder, dtype=float) - matrix @ coefficients
    chi2 = np.sum((residuals * weights) ** 2)
    dof = len(binder) - matrix.shape[1]
    if dof <= 0:
        raise ValueError("servono piu' punti dei parametri di fit")

    return coefficients, chi2, dof


def asymptotic_curve(coefficients, r_xi, degree_main):
    r_xi = np.asarray(r_xi, dtype=float)
    result = np.zeros_like(r_xi)

    for degree in range(degree_main + 1):
        result += coefficients[degree] * r_xi**degree

    return result


def bootstrap_point(
    point, number_of_replicas, rng, chunk_size=200
):
    blocks = point["blocks"]
    number_of_blocks = len(blocks)
    r_xi = np.empty(number_of_replicas)
    binder = np.empty(number_of_replicas)

    denominator = (
        2.0
        * point["L"]
        * math.sin(math.pi / point["L"])
    )

    for first in range(0, number_of_replicas, chunk_size):
        last = min(first + chunk_size, number_of_replicas)
        count = last - first
        indices = rng.integers(
            0,
            number_of_blocks,
            size=(count, number_of_blocks),
        )
        means = np.mean(blocks[indices], axis=1)

        mean_g_zero = means[:, 0]
        mean_g_min = means[:, 1]
        mean_m2 = means[:, 2]
        mean_m4 = means[:, 3]

        if np.any(mean_g_min <= 0.0) or np.any(mean_m2 <= 0.0):
            raise ValueError(
                "bootstrap non valido per "
                f"L={point['L']}, beta={point['beta']:.17g}"
            )

        radicand = mean_g_zero / mean_g_min - 1.0
        if np.any(radicand < 0.0):
            raise ValueError(
                "bootstrap con xi non valida per "
                f"L={point['L']}, beta={point['beta']:.17g}"
            )

        r_xi[first:last] = np.sqrt(radicand) / denominator
        binder[first:last] = mean_m4 / mean_m2**2

    return r_xi, binder


def paired_bootstrap(
    points,
    degree_main,
    degree_correction,
    number_of_replicas,
    seed,
    r_grid,
):
    number_of_points = len(points)
    bootstrap_r_xi = np.empty(
        (number_of_replicas, number_of_points)
    )
    bootstrap_binder = np.empty(
        (number_of_replicas, number_of_points)
    )
    rng = np.random.default_rng(seed)

    lattice_sizes = np.array(
        [point["L"] for point in points], dtype=float
    )
    errors = np.array(
        [point["err_binder"] for point in points], dtype=float
    )

    for column, point in enumerate(points):
        r_xi, binder = bootstrap_point(
            point, number_of_replicas, rng
        )
        bootstrap_r_xi[:, column] = r_xi
        bootstrap_binder[:, column] = binder

    number_of_parameters = (
        degree_main + degree_correction + 2
    )
    coefficients = np.empty(
        (number_of_replicas, number_of_parameters)
    )
    curves = np.empty((number_of_replicas, len(r_grid)))

    for replica in range(number_of_replicas):
        fit, _, _ = fit_polynomials(
            bootstrap_r_xi[replica],
            bootstrap_binder[replica],
            errors,
            lattice_sizes,
            degree_main,
            degree_correction,
        )
        coefficients[replica] = fit
        curves[replica] = asymptotic_curve(
            fit, r_grid, degree_main
        )

    return coefficients, curves


def output_paths(prefix):
    prefix = Path(prefix)
    parent = prefix.parent
    stem = prefix.name
    parent.mkdir(parents=True, exist_ok=True)

    return {
        "summary": parent / f"{stem}_summary.txt",
        "curve": parent / f"{stem}_curve.txt",
        "png": parent / f"{stem}.png",
        "pdf": parent / f"{stem}.pdf",
    }


def save_fit_summary(
    path,
    points,
    sizes,
    degree_main,
    degree_correction,
    number_of_replicas,
    seed,
    coefficients,
    coefficient_errors,
    chi2,
    dof,
    r_min,
    r_max,
):
    labels = [
        f"b{degree}" for degree in range(degree_main + 1)
    ]
    labels.extend(
        f"c{degree}"
        for degree in range(degree_correction + 1)
    )

    with open(path, "w", encoding="utf-8") as file:
        file.write(f"omega = {OMEGA} (fixed)\n")
        file.write(
            "sizes = "
            + " ".join(str(size) for size in sizes)
            + "\n"
        )
        file.write(f"Rxi range = {r_min} {r_max}\n")
        file.write(f"degree main = {degree_main}\n")
        file.write(
            f"degree correction = {degree_correction}\n"
        )
        file.write(f"points = {len(points)}\n")
        file.write(f"parameters = {len(coefficients)}\n")
        file.write(f"chi2 = {chi2:.10g}\n")
        file.write(f"dof = {dof}\n")
        file.write(f"chi2/dof = {chi2 / dof:.10g}\n")
        file.write(
            f"bootstrap replicas = {number_of_replicas}\n"
        )
        file.write(f"bootstrap seed = {seed}\n")
        file.write("\n# parameter value bootstrap_std\n")

        for label, value, error in zip(
            labels, coefficients, coefficient_errors
        ):
            file.write(
                f"{label} {value:.17g} {error:.17g}\n"
            )


def save_curve(
    path, r_grid, central_curve, low_curve, high_curve
):
    data = np.column_stack(
        (r_grid, central_curve, low_curve, high_curve)
    )
    np.savetxt(
        path,
        data,
        fmt="%.17g",
        header="Rxi U_infinity low_68 high_68",
    )


def save_plot(
    path_png,
    path_pdf,
    points,
    r_grid,
    central_curve,
    low_curve,
    high_curve,
):
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.3, 4.7))

    for index, lattice_size in enumerate(
        sorted({point["L"] for point in points})
    ):
        selected = [
            point
            for point in points
            if point["L"] == lattice_size
        ]
        selected.sort(key=lambda point: point["r_xi"])

        axis.errorbar(
            [point["r_xi"] for point in selected],
            [point["binder"] for point in selected],
            xerr=[
                point["err_r_xi"] for point in selected
            ],
            yerr=[
                point["err_binder"] for point in selected
            ],
            fmt="none",
            color=f"C{index}",
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            label=fr"$L={lattice_size}$",
        )

    axis.fill_between(
        r_grid,
        low_curve,
        high_curve,
        alpha=0.2,
        label="68% bootstrap",
    )
    axis.plot(
        r_grid,
        central_curve,
        linewidth=1.4,
        label=r"$U_\infty(R_\xi)$",
    )
    axis.set_xlabel(r"$R_\xi$", fontsize=13)
    axis.set_ylabel(r"$U$", fontsize=13)
    axis.tick_params(
        direction="in",
        top=True,
        right=True,
        labelsize=11,
    )
    axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.tight_layout()
    figure.savefig(path_png, dpi=300, bbox_inches="tight")
    figure.savefig(path_pdf, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit globale di U(R_xi, L) con "
            "bootstrap a blocchi accoppiato"
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "file *_blocks.txt oppure directory "
            "che li contengono"
        ),
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        help=(
            "taglie L da includere; se omesso usa "
            "tutte quelle disponibili"
        ),
    )
    parser.add_argument(
        "--degree-main", type=int, default=6
    )
    parser.add_argument(
        "--degree-correction", type=int, default=3
    )
    parser.add_argument("--r-min", type=float, default=0.30)
    parser.add_argument("--r-max", type=float, default=1.00)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--output-prefix", default="scaling_fit"
    )
    arguments = parser.parse_args()

    try:
        if (
            arguments.degree_main < 0
            or arguments.degree_correction < 0
        ):
            raise ValueError(
                "i gradi dei polinomi devono essere non negativi"
            )
        if arguments.bootstrap < 2:
            raise ValueError(
                "servono almeno 2 repliche bootstrap"
            )
        if (
            arguments.sizes is not None
            and any(size <= 0 for size in arguments.sizes)
        ):
            raise ValueError(
                "le taglie devono essere interi positivi"
            )

        block_files = collect_block_files(arguments.inputs)
        points = load_points(block_files)
        points, sizes = select_points(
            points,
            arguments.sizes,
            arguments.r_min,
            arguments.r_max,
        )

        r_xi = np.array(
            [point["r_xi"] for point in points]
        )
        binder = np.array(
            [point["binder"] for point in points]
        )
        err_binder = np.array(
            [point["err_binder"] for point in points]
        )
        lattice_sizes = np.array(
            [point["L"] for point in points]
        )

        coefficients, chi2, dof = fit_polynomials(
            r_xi,
            binder,
            err_binder,
            lattice_sizes,
            arguments.degree_main,
            arguments.degree_correction,
        )

        r_grid = np.linspace(
            arguments.r_min, arguments.r_max, 200
        )
        central_curve = asymptotic_curve(
            coefficients,
            r_grid,
            arguments.degree_main,
        )
        (
            bootstrap_coefficients,
            bootstrap_curves,
        ) = paired_bootstrap(
            points,
            arguments.degree_main,
            arguments.degree_correction,
            arguments.bootstrap,
            arguments.seed,
            r_grid,
        )
        coefficient_errors = np.std(
            bootstrap_coefficients, axis=0, ddof=1
        )
        low_curve, high_curve = np.percentile(
            bootstrap_curves,
            [16.0, 84.0],
            axis=0,
        )

        paths = output_paths(arguments.output_prefix)
        save_fit_summary(
            paths["summary"],
            points,
            sizes,
            arguments.degree_main,
            arguments.degree_correction,
            arguments.bootstrap,
            arguments.seed,
            coefficients,
            coefficient_errors,
            chi2,
            dof,
            arguments.r_min,
            arguments.r_max,
        )
        save_curve(
            paths["curve"],
            r_grid,
            central_curve,
            low_curve,
            high_curve,
        )
        save_plot(
            paths["png"],
            paths["pdf"],
            points,
            r_grid,
            central_curve,
            low_curve,
            high_curve,
        )
    except (
        OSError,
        ValueError,
        np.linalg.LinAlgError,
    ) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"omega = {OMEGA} (fixed)")
    print(
        "sizes = "
        + " ".join(str(size) for size in sizes)
    )
    print(f"points = {len(points)}")
    print(
        f"chi2/dof = {chi2:.6g}/{dof} "
        f"= {chi2 / dof:.6g}"
    )
    print(
        f"bootstrap replicas = {arguments.bootstrap}"
    )
    print(f"summary: {paths['summary']}")
    print(f"curve:   {paths['curve']}")
    print(f"plot:    {paths['png']}")
    print(f"pdf:     {paths['pdf']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
