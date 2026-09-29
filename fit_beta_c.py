#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_scaling import OMEGA, find_block_files, read_block_file


NU = 0.62997097  # 3D Ising, Chang et al., JHEP 03 (2025) 136


def design_matrix(beta, sizes, beta_c, degree_main, degree_correction):
    sizes = np.asarray(sizes, dtype=float)
    x = (np.asarray(beta) - beta_c) * sizes ** (1.0 / NU)
    main = np.polynomial.polynomial.polyvander(x, degree_main)
    correction = np.polynomial.polynomial.polyvander(x, degree_correction)
    correction *= sizes[:, None] ** (-OMEGA)
    return np.column_stack((main, correction))


def fit_at_beta_c(
    beta,
    r_xi,
    err_r_xi,
    sizes,
    beta_c,
    degree_main,
    degree_correction,
):
    matrix = design_matrix(
        beta, sizes, beta_c, degree_main, degree_correction
    )
    weighted_matrix = matrix / err_r_xi[:, None]
    weighted_r_xi = r_xi / err_r_xi

    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_matrix, weighted_r_xi, rcond=None
    )
    if rank < matrix.shape[1]:
        raise ValueError("fit singolare: ridurre il grado del polinomio")

    residuals = (r_xi - matrix @ coefficients) / err_r_xi
    chi2 = np.sum(residuals**2)
    return coefficients, chi2


def fit_beta_c(
    beta,
    r_xi,
    err_r_xi,
    sizes,
    degree_main,
    degree_correction,
    beta_min,
    beta_max,
):
    if not beta_min < beta_max:
        raise ValueError("beta-min deve essere minore di beta-max")

    def objective(value):
        return fit_at_beta_c(
            beta,
            r_xi,
            err_r_xi,
            sizes,
            value,
            degree_main,
            degree_correction,
        )[1]

    # Una piccola scansione trova un intervallo che contiene il minimo.
    grid = np.linspace(beta_min, beta_max, 31)
    values = np.array([objective(value) for value in grid])
    best = int(np.argmin(values))
    if best == 0 or best == len(grid) - 1:
        raise ValueError(
            "il minimo di chi2 cade al bordo: ampliare l'intervallo in beta"
        )

    left = grid[best - 1]
    right = grid[best + 1]
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = objective(x1)
    f2 = objective(x2)

    for _ in range(40):
        if f1 < f2:
            right = x2
            x2, f2 = x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left = x1
            x1, f1 = x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)

    beta_c = 0.5 * (left + right)
    coefficients, chi2 = fit_at_beta_c(
        beta,
        r_xi,
        err_r_xi,
        sizes,
        beta_c,
        degree_main,
        degree_correction,
    )
    dof = len(r_xi) - len(coefficients) - 1
    if dof <= 0:
        raise ValueError("servono piu' punti dei parametri di fit")

    return beta_c, coefficients, chi2, dof


def bootstrap_rxi(point, rng):
    blocks = point["blocks"]
    sampled = blocks[rng.integers(0, len(blocks), len(blocks))]
    g_zero, g_min = np.mean(sampled[:, :2], axis=0)

    if g_min <= 0.0 or g_zero < g_min:
        raise ValueError(
            f"replica bootstrap non valida per L={point['L']}, beta={point['beta']}"
        )

    r_xi = math.sqrt(g_zero / g_min - 1.0)
    r_xi /= 2.0 * point["L"] * math.sin(math.pi / point["L"])
    return r_xi


def bootstrap_fits(
    points,
    degree_main,
    degree_correction,
    replicas,
    seed,
    beta_min,
    beta_max,
):
    rng = np.random.default_rng(seed)
    beta = np.array([point["beta"] for point in points])
    sizes = np.array([point["L"] for point in points], dtype=float)
    errors = np.array([point["err_Rxi"] for point in points])
    number_of_coefficients = degree_main + degree_correction + 2
    beta_c_all = np.empty(replicas)
    coefficients_all = np.empty((replicas, number_of_coefficients))

    for replica in range(replicas):
        r_xi = np.array([bootstrap_rxi(point, rng) for point in points])
        beta_c, coefficients, _, _ = fit_beta_c(
            beta,
            r_xi,
            errors,
            sizes,
            degree_main,
            degree_correction,
            beta_min,
            beta_max,
        )
        beta_c_all[replica] = beta_c
        coefficients_all[replica] = coefficients

    return beta_c_all, coefficients_all


def fitted_rxi(beta, size, beta_c, coefficients, degree_main, degree_correction):
    beta = np.asarray(beta)
    sizes = np.full(beta.shape, float(size))
    matrix = design_matrix(
        beta, sizes, beta_c, degree_main, degree_correction
    )
    return matrix @ coefficients


def save_plot(
    points,
    beta_c,
    coefficients,
    degree_main,
    degree_correction,
    output,
):
    figure, axis = plt.subplots(figsize=(6.3, 4.7))

    for index, size in enumerate(sorted({point["L"] for point in points})):
        selected = [point for point in points if point["L"] == size]
        selected.sort(key=lambda point: point["beta"])
        beta = np.array([point["beta"] for point in selected])
        beta_grid = np.linspace(beta.min(), beta.max(), 200)

        axis.errorbar(
            beta,
            [point["Rxi"] for point in selected],
            yerr=[point["err_Rxi"] for point in selected],
            fmt="none",
            color=f"C{index}",
            elinewidth=0.8,
            capsize=2.0,
            capthick=0.8,
            label=fr"$L={size}$",
        )
        axis.plot(
            beta_grid,
            fitted_rxi(
                beta_grid,
                size,
                beta_c,
                coefficients,
                degree_main,
                degree_correction,
            ),
            color=f"C{index}",
            linewidth=1.0,
        )

    axis.axvline(beta_c, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel(r"$\beta$")
    axis.set_ylabel(r"$R_\xi$")
    axis.tick_params(direction="in", top=True, right=True)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output, dpi=300, bbox_inches="tight")
    figure.savefig(Path(output).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Fit FSS di R_xi(beta, L) per beta_c")
    parser.add_argument("inputs", nargs="+", help="file o directory con *_blocks.txt")
    parser.add_argument("--sizes", nargs="+", type=int)
    parser.add_argument("--degree-main", type=int, default=2)
    parser.add_argument("--degree-correction", type=int, default=1)
    parser.add_argument("--r-min", type=float, default=0.30)
    parser.add_argument("--r-max", type=float, default=1.00)
    parser.add_argument("--beta-min", type=float)
    parser.add_argument("--beta-max", type=float)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-prefix", default="beta_c_fit")
    args = parser.parse_args()

    try:
        points = [read_block_file(path) for path in find_block_files(args.inputs)]
        sizes = sorted({point["L"] for point in points}) if args.sizes is None else args.sizes
        points = [
            point for point in points
            if point["L"] in sizes and args.r_min <= point["Rxi"] <= args.r_max
        ]
        points.sort(key=lambda point: (point["L"], point["beta"]))

        if not points:
            raise ValueError("nessun punto selezionato")
        if args.bootstrap < 2:
            raise ValueError("servono almeno 2 repliche bootstrap")
        if args.r_min >= args.r_max:
            raise ValueError("r-min deve essere minore di r-max")

        beta = np.array([point["beta"] for point in points])
        r_xi = np.array([point["Rxi"] for point in points])
        err_r_xi = np.array([point["err_Rxi"] for point in points])
        lattice_sizes = np.array([point["L"] for point in points], dtype=float)
        if np.any(err_r_xi <= 0.0):
            raise ValueError("gli errori di R_xi devono essere positivi")

        minimum_by_size = [
            min(point["beta"] for point in points if point["L"] == size)
            for size in sizes
        ]
        maximum_by_size = [
            max(point["beta"] for point in points if point["L"] == size)
            for size in sizes
        ]
        default_beta_min = max(minimum_by_size)
        default_beta_max = min(maximum_by_size)
        beta_min = default_beta_min if args.beta_min is None else args.beta_min
        beta_max = default_beta_max if args.beta_max is None else args.beta_max

        beta_c, coefficients, chi2, dof = fit_beta_c(
            beta,
            r_xi,
            err_r_xi,
            lattice_sizes,
            args.degree_main,
            args.degree_correction,
            beta_min,
            beta_max,
        )
        bootstrap_beta_c, bootstrap_coefficients = bootstrap_fits(
            points,
            args.degree_main,
            args.degree_correction,
            args.bootstrap,
            args.seed,
            beta_min,
            beta_max,
        )

        beta_c_error = np.std(bootstrap_beta_c, ddof=1)
        coefficient_errors = np.std(bootstrap_coefficients, axis=0, ddof=1)
        rxi_star = coefficients[0]
        rxi_star_error = coefficient_errors[0]

        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        summary_file = prefix.with_name(prefix.name + "_summary.txt")
        plot_file = prefix.with_suffix(".png")

        with open(summary_file, "w", encoding="utf-8") as file:
            file.write(f"nu = {NU} (fixed)\n")
            file.write(f"omega = {OMEGA} (fixed)\n")
            file.write("sizes = " + " ".join(map(str, sizes)) + "\n")
            file.write(f"Rxi range = {args.r_min} {args.r_max}\n")
            file.write(f"beta search range = {beta_min:.17g} {beta_max:.17g}\n")
            file.write(f"degree main = {args.degree_main}\n")
            file.write(f"degree correction = {args.degree_correction}\n")
            file.write(f"points = {len(points)}\n")
            file.write(f"chi2/dof = {chi2:.10g}/{dof} = {chi2 / dof:.10g}\n")
            file.write(f"bootstrap replicas = {args.bootstrap}\n")
            file.write(f"bootstrap seed = {args.seed}\n")
            file.write(f"beta_c = {beta_c:.17g} {beta_c_error:.17g}\n")
            file.write(f"Rxi_star = {rxi_star:.17g} {rxi_star_error:.17g}\n\n")
            file.write("# parameter value bootstrap_std\n")
            labels = [f"a{k}" for k in range(args.degree_main + 1)]
            labels += [f"b{k}" for k in range(args.degree_correction + 1)]
            for label, value, error in zip(labels, coefficients, coefficient_errors):
                file.write(f"{label} {value:.17g} {error:.17g}\n")

        save_plot(
            points,
            beta_c,
            coefficients,
            args.degree_main,
            args.degree_correction,
            plot_file,
        )

    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"nu = {NU} (fixed)")
    print(f"omega = {OMEGA} (fixed)")
    print("sizes = " + " ".join(map(str, sizes)))
    print(f"points = {len(points)}")
    print(f"beta_c = {beta_c:.10g} +/- {beta_c_error:.3g}")
    print(f"Rxi_star = {rxi_star:.10g} +/- {rxi_star_error:.3g}")
    print(f"chi2/dof = {chi2:.6g}/{dof} = {chi2 / dof:.6g}")
    print(f"summary: {summary_file}")
    print(f"plot:    {plot_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
