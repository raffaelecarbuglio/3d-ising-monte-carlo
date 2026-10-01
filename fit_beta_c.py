#!/usr/bin/env python3
"""Fit di R_xi(beta,L): fit lineare a beta_c fissato, poi bootstrap dei blocchi."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_scaling import OMEGA, bootstrap_pair, find_block_files, read_block_file

NU = 0.62997097  # 3D Ising, Chang et al., JHEP 03 (2025) 136


def design_matrix(beta, sizes, beta_c, degree_main, degree_correction):
    sizes = np.asarray(sizes, dtype=float)
    x = (np.asarray(beta) - beta_c) * sizes ** (1.0 / NU)
    main = np.polynomial.polynomial.polyvander(x, degree_main)
    correction = np.polynomial.polynomial.polyvander(x, degree_correction)
    return np.column_stack((main, sizes[:, None] ** (-OMEGA) * correction))


def fit_at_beta_c(beta, r_xi, errors, sizes, beta_c, p, q):
    matrix = design_matrix(beta, sizes, beta_c, p, q)
    coefficients, _, rank, _ = np.linalg.lstsq(
        matrix / errors[:, None], r_xi / errors, rcond=None
    )
    if rank < matrix.shape[1]:
        raise ValueError("fit singolare: ridurre il grado del polinomio")
    residuals = (r_xi - matrix @ coefficients) / errors
    return coefficients, float(residuals @ residuals)


def make_fitter(beta, errors, sizes, p, q, beta_min, beta_max):
    """Prepare fixed inputs once; return a fit usable for bootstrap replicas."""
    beta, errors, sizes = [np.asarray(v, dtype=float) for v in (beta, errors, sizes)]
    if any(v.ndim != 1 or v.shape != beta.shape or not np.all(np.isfinite(v))
           for v in (beta, errors, sizes)):
        raise ValueError("i dati devono essere vettori finiti della stessa lunghezza")
    if np.any(errors <= 0) or np.any(sizes <= 1):
        raise ValueError("servono errori positivi e L > 1")
    if p < 1 or q < 0:
        raise ValueError("servono degree-main >= 1 e degree-correction >= 0")
    if len(np.unique(sizes)) < 3:
        raise ValueError("servono almeno 3 taglie: con due taglie beta_c puo' essere indeterminato")
    dof = len(beta) - (p + q + 2) - 1
    if dof <= 0:
        raise ValueError("servono piu' punti dei parametri di fit")
    if not np.isfinite([beta_min, beta_max]).all() or beta_min >= beta_max:
        raise ValueError("servono beta-min < beta-max finiti")

    def fit(r_xi):
        r_xi = np.asarray(r_xi, dtype=float)
        if r_xi.shape != beta.shape or not np.isfinite(r_xi).all():
            raise ValueError("Rxi deve essere un vettore finito della stessa lunghezza di beta")

        def objective(value):
            return fit_at_beta_c(beta, r_xi, errors, sizes, value, p, q)[1]

        grid = np.linspace(beta_min, beta_max, 31)
        values = np.array([objective(value) for value in grid])
        if np.ptp(values) <= 1e-8 * max(1.0, values.min()):
            raise ValueError("beta_c indeterminato: profilo chi2 piatto")
        best = int(np.argmin(values))
        # Refine even when the coarse scan chooses an endpoint.
        left, right = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        golden = (math.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = right - golden * (right - left), left + golden * (right - left)
        f1, f2 = objective(x1), objective(x2)
        for _ in range(40):
            if f1 < f2:
                right, x2, f2 = x2, x1, f1
                x1 = right - golden * (right - left)
                f1 = objective(x1)
            else:
                left, x1, f1 = x1, x2, f2
                x2 = left + golden * (right - left)
                f2 = objective(x2)
        beta_c = (left + right) / 2.0
        coefficients, chi2 = fit_at_beta_c(beta, r_xi, errors, sizes, beta_c, p, q)
        if min(beta_c - beta_min, beta_max - beta_c) <= (beta_max - beta_min) * 1e-8:
            raise ValueError("minimo al bordo: ampliare l'intervallo in beta")
        if min(values[0], values[-1]) <= chi2 + 1.0:
            raise ValueError("beta_c poco vincolato nell'intervallo: ampliare i limiti o aggiungere dati")
        return beta_c, coefficients, chi2, dof

    return fit


def bootstrap_fits(points, fit, replicas, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for replica in range(replicas):
        try:
            beta_c, coefficients, _, _ = fit([bootstrap_pair(point, rng)[0] for point in points])
        except ValueError as error:
            raise ValueError(f"bootstrap {replica + 1}/{replicas}: {error}") from error
        samples.append(np.r_[beta_c, coefficients])
    return np.array(samples)


def save_plot(points, beta_c, coefficients, p, q, output):
    figure, axis = plt.subplots(figsize=(6.3, 4.7))
    for index, size in enumerate(sorted({point["L"] for point in points})):
        selected = [point for point in points if point["L"] == size]
        beta = np.array([point["beta"] for point in selected])
        axis.errorbar(beta, [point["Rxi"] for point in selected],
                      yerr=[point["err_Rxi"] for point in selected],
                      fmt="none", color=f"C{index}", elinewidth=0.8,
                      capsize=2, capthick=0.8, label=fr"$L={size}$")
        grid = np.linspace(beta.min(), beta.max(), 200)
        axis.plot(grid, design_matrix(grid, np.full_like(grid, size), beta_c, p, q) @ coefficients,
                  color=f"C{index}", linewidth=1)
    axis.axvline(beta_c, color="black", linewidth=1, linestyle="--")
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
    parser.add_argument("--r-min", type=float, default=0.45)
    parser.add_argument("--r-max", type=float, default=0.75)
    parser.add_argument("--beta-min", type=float)
    parser.add_argument("--beta-max", type=float)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-prefix", default="beta_c_fit")
    args = parser.parse_args()

    try:
        if not np.isfinite([args.r_min, args.r_max]).all() or args.r_min >= args.r_max:
            raise ValueError("servono r-min < r-max finiti")
        if args.bootstrap < 2:
            raise ValueError("servono almeno 2 repliche bootstrap")
        points = [read_block_file(path) for path in find_block_files(args.inputs)]
        points = [point for point in points
                  if (args.sizes is None or point["L"] in args.sizes)
                  and args.r_min <= point["Rxi"] <= args.r_max]
        if not points:
            raise ValueError("nessun punto selezionato")
        sizes = sorted({point["L"] for point in points})
        if args.sizes is not None and set(args.sizes) != set(sizes):
            raise ValueError("una taglia richiesta non ha punti nella finestra Rxi")
        points.sort(key=lambda point: (point["L"], point["beta"]))
        beta, r_xi, errors, lattice_sizes = (
            np.array([point[key] for point in points]) for key in ("beta", "Rxi", "err_Rxi", "L")
        )
        beta_min = beta.min() if args.beta_min is None else args.beta_min
        beta_max = beta.max() if args.beta_max is None else args.beta_max
        p, q = args.degree_main, args.degree_correction
        fit = make_fitter(beta, errors, lattice_sizes, p, q, beta_min, beta_max)
        beta_c, coefficients, chi2, dof = fit(r_xi)
        samples = bootstrap_fits(points, fit, args.bootstrap, args.seed)
        beta_c_error, *coefficient_errors = np.std(samples, axis=0, ddof=1)

        # Un solo resoconto, usato sia per il file sia per il terminale.
        report = [f"nu = {NU} (fixed)", f"omega = {OMEGA} (fixed)",
                  "sizes = " + " ".join(map(str, sizes)),
                  f"Rxi range = {args.r_min} {args.r_max}",
                  f"beta search range = {beta_min:.17g} {beta_max:.17g}",
                  f"degree main = {p}", f"degree correction = {q}",
                  f"points = {len(points)}",
                  f"chi2/dof = {chi2:.10g}/{dof} = {chi2 / dof:.10g}",
                  f"bootstrap replicas = {args.bootstrap}", f"bootstrap seed = {args.seed}",
                  f"beta_c = {beta_c:.17g} {beta_c_error:.17g}",
                  f"Rxi_star = {coefficients[0]:.17g} {coefficient_errors[0]:.17g}",
                  "", "# parameter value bootstrap_std"]
        labels = [f"a{k}" for k in range(p + 1)] + [f"b{k}" for k in range(q + 1)]
        report += [f"{name} {value:.17g} {error:.17g}"
                   for name, value, error in zip(labels, coefficients, coefficient_errors)]
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        summary_file = prefix.with_name(prefix.name + "_summary.txt")
        plot_file = prefix.with_suffix(".png")
        save_plot(points, beta_c, coefficients, p, q, plot_file)
        summary_file.write_text("\n".join(report) + "\n", encoding="utf-8")
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1
    print("\n".join(report))
    print(f"summary: {summary_file}\nplot: {plot_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
