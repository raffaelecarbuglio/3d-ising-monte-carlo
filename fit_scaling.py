#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_u_vs_rxi import plot_points


OMEGA = 0.8295  # 3D Ising, Reehorst, JHEP 09 (2022) 177


def find_block_files(inputs):
    files = []
    for name in inputs:
        path = Path(name)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*_blocks.txt")))
        else:
            raise ValueError(f"'{name}' non esiste")

    files = list(dict.fromkeys(files))
    if not files:
        raise ValueError("nessun file *_blocks.txt trovato")
    return files


def read_block_file(filename):
    info = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") and "=" in line:
                key, value = line[1:].split("=", 1)
                info[key.strip()] = value.strip()

    blocks = np.loadtxt(filename, comments="#", ndmin=2)
    if blocks.shape[1] != 4:
        raise ValueError(f"{filename}: servono 4 colonne")

    try:
        return {
            "L": int(info["L"]),
            "beta": float(info["beta"]),
            "Rxi": float(info["Rxi"]),
            "err_Rxi": float(info["err_Rxi"]),
            "U": float(info["U"]),
            "err_U": float(info["err_U"]),
            "blocks": blocks,
        }
    except (KeyError, ValueError) as error:
        raise ValueError(f"{filename}: intestazione non valida") from error


def design_matrix(r_xi, sizes, degree_main, degree_correction):
    main = np.polynomial.polynomial.polyvander(r_xi, degree_main)
    correction = np.polynomial.polynomial.polyvander(r_xi, degree_correction)
    correction *= np.asarray(sizes)[:, None] ** (-OMEGA)
    return np.column_stack((main, correction))


def fit(r_xi, u, err_u, sizes, degree_main, degree_correction):
    matrix = design_matrix(r_xi, sizes, degree_main, degree_correction)
    weighted_matrix = matrix / err_u[:, None]
    weighted_u = u / err_u

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_matrix, weighted_u, rcond=None
    )
    if rank < matrix.shape[1]:
        raise ValueError("fit singolare: ridurre il grado del polinomio")

    residuals = (u - matrix @ coefficients) / err_u
    chi2 = np.sum(residuals**2)
    dof = len(u) - len(coefficients)
    if dof <= 0:
        raise ValueError("servono piu' punti dei parametri di fit")

    return coefficients, chi2, dof


def asymptotic_curve(coefficients, r_xi, degree_main):
    return np.polynomial.polynomial.polyval(
        r_xi, coefficients[: degree_main + 1]
    )


def observable_pairs(means, size):
    """Reconstruct (Rxi,U) from rows of (g_zero,g_min,m2,m4) means."""
    means = np.asarray(means)
    g0, gp, m2, m4 = means.T
    if (not np.isfinite(means).all() or np.any(gp <= 0)
            or np.any(g0 < gp) or np.any(m2 <= 0) or np.any(m4 <= 0)):
        raise ValueError(f"invalid block observables for L={size}")
    r = np.sqrt(g0 / gp - 1) / (2 * size * math.sin(math.pi / size))
    return np.stack((r, m4 / m2**2), axis=-1)


def bootstrap_pair(point, rng):
    blocks = point["blocks"]
    counts = np.bincount(rng.integers(len(blocks), size=len(blocks)), minlength=len(blocks))
    return tuple(observable_pairs(counts @ blocks / len(blocks), point["L"]))


def bootstrap_curves(points, degree_main, degree_correction, replicas, seed, r_grid):
    rng = np.random.default_rng(seed)
    sizes = np.array([point["L"] for point in points], dtype=float)
    err_u = np.array([point["err_U"] for point in points])
    number_of_parameters = degree_main + degree_correction + 2
    coefficients_all = np.empty((replicas, number_of_parameters))
    curves = np.empty((replicas, len(r_grid)))

    for replica in range(replicas):
        pairs = [bootstrap_pair(point, rng) for point in points]
        r_xi = np.array([pair[0] for pair in pairs])
        u = np.array([pair[1] for pair in pairs])
        coefficients, _, _ = fit(
            r_xi, u, err_u, sizes, degree_main, degree_correction
        )
        coefficients_all[replica] = coefficients
        curves[replica] = asymptotic_curve(coefficients, r_grid, degree_main)

    return coefficients_all, curves


def save_plot(points, r_grid, central, low, high, output):
    figure, axis = plt.subplots(figsize=(6.3, 4.7))

    plot_points(axis, np.array([
        [point[key] for key in ("L", "beta", "Rxi", "err_Rxi", "U", "err_U")]
        for point in points
    ]))

    axis.fill_between(r_grid, low, high, alpha=0.2, label="68% bootstrap")
    axis.plot(r_grid, central, linewidth=1.4, label=r"$U_\infty(R_\xi)$")
    axis.set_xlabel(r"$R_\xi$")
    axis.set_ylabel(r"$U$")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(Path(output).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Fit di scaling U(R_xi, L)")
    parser.add_argument("inputs", nargs="+", help="file o directory con *_blocks.txt")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--degree-main", type=int, default=6)
    parser.add_argument("--degree-correction", type=int, default=3)
    parser.add_argument("--r-min", type=float, default=0.30)
    parser.add_argument("--r-max", type=float, default=1.00)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-prefix", default="scaling_fit")
    args = parser.parse_args()

    try:
        points = [read_block_file(path) for path in find_block_files(args.inputs)]
        sizes = sorted({point["L"] for point in points}) if args.sizes is None else args.sizes
        points = [
            point for point in points
            if point["L"] in sizes and args.r_min <= point["Rxi"] <= args.r_max
        ]
        points.sort(key=lambda point: (point["L"], point["Rxi"]))

        if not points:
            raise ValueError("nessun punto selezionato")
        if args.bootstrap < 2:
            raise ValueError("servono almeno 2 repliche bootstrap")

        r_xi = np.array([point["Rxi"] for point in points])
        u = np.array([point["U"] for point in points])
        err_u = np.array([point["err_U"] for point in points])
        lattice_sizes = np.array([point["L"] for point in points], dtype=float)

        coefficients, chi2, dof = fit(
            r_xi,
            u,
            err_u,
            lattice_sizes,
            args.degree_main,
            args.degree_correction,
        )

        r_grid = np.linspace(args.r_min, args.r_max, 200)
        central = asymptotic_curve(coefficients, r_grid, args.degree_main)
        bootstrap_coefficients, curves = bootstrap_curves(
            points,
            args.degree_main,
            args.degree_correction,
            args.bootstrap,
            args.seed,
            r_grid,
        )
        low, high = np.percentile(curves, [16.0, 84.0], axis=0)
        coefficient_errors = np.std(bootstrap_coefficients, axis=0, ddof=1)

        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        curve_file = prefix.with_name(prefix.name + "_curve.txt")
        summary_file = prefix.with_name(prefix.name + "_summary.txt")
        plot_file = prefix.with_suffix(".png")

        np.savetxt(
            curve_file,
            np.column_stack((r_grid, central, low, high)),
            header="Rxi U_infinity low_68 high_68",
        )

        with open(summary_file, "w", encoding="utf-8") as file:
            file.write(f"omega = {OMEGA} (fixed)\n")
            file.write("sizes = " + " ".join(map(str, sizes)) + "\n")
            file.write(f"degree main = {args.degree_main}\n")
            file.write(f"degree correction = {args.degree_correction}\n")
            file.write(f"points = {len(points)}\n")
            file.write(f"chi2/dof = {chi2:.10g}/{dof} = {chi2 / dof:.10g}\n")
            file.write(f"bootstrap replicas = {args.bootstrap}\n")
            file.write(f"bootstrap seed = {args.seed}\n\n")
            file.write("# parameter value bootstrap_std\n")
            labels = [f"b{k}" for k in range(args.degree_main + 1)]
            labels += [f"c{k}" for k in range(args.degree_correction + 1)]
            for label, value, error in zip(labels, coefficients, coefficient_errors):
                file.write(f"{label} {value:.17g} {error:.17g}\n")

        save_plot(points, r_grid, central, low, high, plot_file)

    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"omega = {OMEGA} (fixed)")
    print("sizes = " + " ".join(map(str, sizes)))
    print(f"points = {len(points)}")
    print(f"chi2/dof = {chi2:.6g}/{dof} = {chi2 / dof:.6g}")
    print(f"curve:   {curve_file}")
    print(f"summary: {summary_file}")
    print(f"plot:    {plot_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
