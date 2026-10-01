#!/usr/bin/env python3
"""Compare perturbed U(Rxi,L) with a correlated clean scaling fit.

Only the clean data are fitted. Independent block bootstraps propagate the
uncertainty of that reference into every perturbed residual.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from fit_scaling import OMEGA, bootstrap_pair, find_block_files, observable_pairs, read_block_file
from plot_u_vs_rxi import plot_points


def jackknife_pair(point):
    blocks = point["blocks"]
    if len(blocks) < 3 or not np.isfinite(blocks).all():
        raise ValueError("at least three finite blocks are required")
    central = observable_pairs(blocks.mean(axis=0), point["L"])
    leave_one_out = observable_pairs(
        (blocks.sum(axis=0) - blocks) / (len(blocks) - 1), point["L"]
    )
    centered = leave_one_out - leave_one_out.mean(axis=0)
    covariance = (len(blocks) - 1) / len(blocks) * centered.T @ centered
    if not np.isfinite(covariance).all():
        raise ValueError("non-finite jackknife covariance")
    # Singular error ellipses cannot be whitened; do not silently regularize.
    np.linalg.cholesky(covariance)
    return central, covariance


def load_points(inputs, lmin=2, sizes=None, rmin=None, rmax=None):
    files = sorted({path.resolve() for path in find_block_files(inputs)})
    points = []
    identities = {}
    for path in files:
        point = read_block_file(path)
        if point["L"] < lmin or (sizes is not None and point["L"] not in sizes):
            continue
        try:
            pair, covariance = jackknife_pair(point)
            if not np.isfinite(point["beta"]):
                raise ValueError("beta must be finite")
            if not np.allclose(pair, [point["Rxi"], point["U"]], rtol=1e-8, atol=1e-10):
                raise ValueError("header observables disagree with the block means; rerun analyze.py")
            if not np.allclose(np.sqrt(np.diag(covariance)),
                               [point["err_Rxi"], point["err_U"]], rtol=1e-6, atol=1e-12):
                raise ValueError("header errors disagree with the block jackknife; rerun analyze.py")
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"{path}: {error}") from error
        if (rmin is not None and pair[0] < rmin) or (rmax is not None and pair[0] > rmax):
            continue
        identity = (point["L"], point["beta"])
        if identity in identities:
            raise ValueError(f"duplicate L,beta: {identities[identity]} and {path}; "
                             "supply one analysis file per simulation point")
        identities[identity] = path
        point.update(path=path, pair=pair, covariance=covariance)
        points.append(point)
    points.sort(key=lambda point: (point["L"], point["beta"]))
    if not points:
        raise ValueError("no points selected")
    if sizes is not None and set(sizes) != {point["L"] for point in points}:
        raise ValueError("a requested size has no points in the selected Rxi window")
    return points


def basis(r, sizes, p, q, omega, domain):
    """Chebyshev basis: same polynomial space, better numerical conditioning."""
    low, high = domain
    t = (2 * np.asarray(r) - low - high) / (high - low)
    main = np.polynomial.chebyshev.chebvander(t, p)
    correction = np.polynomial.chebyshev.chebvander(t, q)
    # Normalizing L at 16 changes only the correction coefficients.
    factor = (np.asarray(sizes, dtype=float) / 16) ** (-omega)
    return np.column_stack((main, factor[:, None] * correction))


def evaluate(coefficients, r, p, domain, sizes=None, omega=OMEGA):
    low, high = domain
    t = (2 * np.asarray(r) - low - high) / (high - low)
    result = np.polynomial.chebyshev.chebval(t, np.asarray(coefficients)[:p + 1])
    if sizes is not None:
        result += (np.asarray(sizes) / 16) ** (-omega) * np.polynomial.chebyshev.chebval(
            t, np.asarray(coefficients)[p + 1:]
        )
    return result


def derivative(coefficients, r, sizes, p, omega, domain):
    t = (2 * np.asarray(r) - sum(domain)) / (domain[1] - domain[0])
    dmain = np.polynomial.chebyshev.chebder(coefficients[:p + 1])
    dcorr = np.polynomial.chebyshev.chebder(coefficients[p + 1:])
    return 2 / (domain[1] - domain[0]) * (
        np.polynomial.chebyshev.chebval(t, dmain)
        + (np.asarray(sizes) / 16) ** (-omega) * np.polynomial.chebyshev.chebval(t, dcorr)
    )


def fit_clean(pairs, covariances, sizes, p, q, omega, domain, initial=None):
    """Minimize sum (observed-model)^T C^-1 (observed-model).

    There is one fitted true abscissa per clean point. Covariances stay fixed
    to their original block-jackknife estimates, including in bootstrap fits.
    """
    pairs, covariances = np.asarray(pairs), np.asarray(covariances)
    sizes = np.asarray(sizes, dtype=float)
    n, k = len(pairs), p + q + 2
    if len(np.unique(sizes)) < 2 or n <= k:
        raise ValueError("clean fit requires at least two sizes and more points than coefficients")
    sx = np.sqrt(covariances[:, 0, 0])
    cross = covariances[:, 1, 0] / sx
    conditional = np.sqrt(covariances[:, 1, 1] - cross**2)
    if not np.isfinite(conditional).all() or np.any(conditional <= 0):
        raise ValueError("clean covariance matrices must be positive definite")
    matrix = basis(pairs[:, 0], sizes, p, q, omega, domain)
    weighted = matrix / np.sqrt(covariances[:, 1, 1])[:, None]
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted, pairs[:, 1] / np.sqrt(covariances[:, 1, 1]), rcond=None
    )
    if rank < k:
        raise ValueError("singular clean fit: reduce polynomial degrees")
    if initial is not None:
        coefficients = initial

    def residual(parameters):
        c, true_r = parameters[:k], parameters[k:]
        dx = (pairs[:, 0] - true_r) / sx
        dy = (pairs[:, 1] - evaluate(c, true_r, p, domain, sizes, omega) - cross * dx) / conditional
        return np.concatenate((dx, dy))

    def jacobian(parameters):
        c, true_r = parameters[:k], parameters[k:]
        jac = np.zeros((2 * n, k + n))
        jac[np.arange(n), k + np.arange(n)] = -1 / sx
        jac[n:, :k] = -basis(true_r, sizes, p, q, omega, domain) / conditional[:, None]
        jac[n + np.arange(n), k + np.arange(n)] = (
            -derivative(c, true_r, sizes, p, omega, domain) + cross / sx
        ) / conditional
        return jac

    solution = least_squares(
        residual, np.r_[coefficients, pairs[:, 0]], jac=jacobian, method="lm",
        x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise ValueError(f"clean EIV fit failed: {solution.message}")
    if np.linalg.matrix_rank(solution.jac) < k + n:
        raise ValueError("singular clean EIV fit")
    return solution.x[:k], float(solution.fun @ solution.fun), n - k


def resample_pairs(points, rng):
    return np.array([bootstrap_pair(point, rng) for point in points])


def bootstrap_comparison(clean, perturbed, coefficients, p, q, omega, domain, replicas, seed):
    # Separate streams keep the two independently simulated ensembles independent.
    streams = np.random.SeedSequence(seed).spawn(2)
    clean_rng, perturbed_rng = [np.random.default_rng(stream) for stream in streams]
    sizes = np.array([point["L"] for point in clean])
    covariance = np.array([point["covariance"] for point in clean])
    factors = np.array([point["L"] for point in perturbed]) ** omega
    coefficient_samples = np.empty((replicas, len(coefficients)))
    pair_samples = np.empty((replicas, len(perturbed), 2))
    delta_samples = np.empty((replicas, len(perturbed)))
    for b in range(replicas):
        try:
            sampled_clean = resample_pairs(clean, clean_rng)
            c, _, _ = fit_clean(sampled_clean, covariance, sizes, p, q, omega, domain, coefficients)
            pairs = resample_pairs(perturbed, perturbed_rng)
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(f"bootstrap {b + 1}/{replicas}: {error}") from error
        coefficient_samples[b] = c
        pair_samples[b] = pairs
        # One shared clean replica for ALL perturbed points in this replica.
        delta_samples[b] = factors * (pairs[:, 1] - evaluate(c, pairs[:, 0], p, domain))
        if (b + 1) % 50 == 0 or b + 1 == replicas:
            print(f"bootstrap {b + 1}/{replicas}", flush=True)
    return coefficient_samples, pair_samples, delta_samples


def save_plots(points, grid, reference, low, high, scaled, scaled_errors, prefix):
    pairs = np.array([point["pair"] for point in points])
    errors = np.sqrt([np.diag(point["covariance"]) for point in points])
    data = np.column_stack(([point["L"] for point in points],
                            [point["beta"] for point in points],
                            pairs[:, 0], errors[:, 0], pairs[:, 1], errors[:, 1]))
    for suffix, ylabel in (("u_vs_rxi", r"$U$"),
                           ("scaled_delta_u", r"$L^\omega[U-U_\infty^{\rm clean}(R_\xi)]$")):
        figure, axis = plt.subplots(figsize=(6.3, 4.7))
        if suffix == "scaled_delta_u":
            data[:, 4], data[:, 5] = scaled, scaled_errors
        plot_points(axis, data)
        if suffix == "u_vs_rxi":
            axis.fill_between(grid, low, high, color="black", alpha=0.15,
                              label="Clean reference: pointwise 68% bootstrap")
            axis.plot(grid, reference, color="black", linewidth=1.3,
                      label=r"$U_\infty^{\rm clean}(R_\xi)$")
        else:
            axis.axhline(0, color="black", linewidth=0.7, linestyle="--")
        axis.set_xlabel(r"$R_\xi$")
        axis.set_ylabel(ylabel)
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(frameon=False, fontsize=8)
        figure.tight_layout()
        for extension in ("png", "pdf"):
            figure.savefig(f"{prefix}_{suffix}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", nargs="+", required=True, help="clean block files or directories")
    parser.add_argument("--perturbed", nargs="+", required=True, help="perturbed block files or directories")
    parser.add_argument("--clean-lmin", type=int, default=16)
    parser.add_argument("--clean-r-min", type=float, help="optional clean-fit cut; default uses all clean points")
    parser.add_argument("--clean-r-max", type=float, help="optional clean-fit cut; default uses all clean points")
    parser.add_argument("--perturbed-sizes", nargs="+", type=int)
    parser.add_argument("--degree-main", type=int, default=6)
    parser.add_argument("--degree-correction", type=int, default=3)
    parser.add_argument("--omega", type=float, default=OMEGA)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--r-min", type=float, default=0.30, help="perturbed selection and plotting window")
    parser.add_argument("--r-max", type=float, default=1.00, help="perturbed selection and plotting window")
    parser.add_argument("--output-prefix", default="universality")
    args = parser.parse_args(argv)
    try:
        if args.degree_main < 0 or args.degree_correction < 0 or args.bootstrap < 2:
            raise ValueError("nonnegative degrees and at least two bootstrap replicas are required")
        if (not np.isfinite([args.r_min, args.r_max, args.omega]).all()
                or args.r_min >= args.r_max or args.omega <= 0 or args.clean_lmin < 2 or args.seed < 0):
            raise ValueError("invalid window, omega, clean-lmin or seed")
        for cut in (args.clean_r_min, args.clean_r_max):
            if cut is not None and not np.isfinite(cut):
                raise ValueError("clean Rxi cuts must be finite")
        if (args.clean_r_min is not None and args.clean_r_max is not None
                and args.clean_r_min >= args.clean_r_max):
            raise ValueError("clean-r-min must be smaller than clean-r-max")
        clean = load_points(args.clean, args.clean_lmin, rmin=args.clean_r_min, rmax=args.clean_r_max)
        perturbed = load_points(args.perturbed, sizes=args.perturbed_sizes,
                                rmin=args.r_min, rmax=args.r_max)
        if {point["path"] for point in clean} & {point["path"] for point in perturbed}:
            raise ValueError("clean and perturbed inputs must be independent, disjoint ensembles")
        pairs = np.array([point["pair"] for point in clean])
        covariance = np.array([point["covariance"] for point in clean])
        sizes = np.array([point["L"] for point in clean])
        domain = (float(pairs[:, 0].min()), float(pairs[:, 0].max()))
        measured = np.array([point["pair"] for point in perturbed])
        errors = np.sqrt([np.diag(point["covariance"]) for point in perturbed])
        if np.any(measured[:, 0] < domain[0]) or np.any(measured[:, 0] > domain[1]):
            raise ValueError("perturbed central points extend beyond the clean reference range; "
                             "narrow --r-min/--r-max or extend the clean data")
        coefficients, chi2, dof = fit_clean(
            pairs, covariance, sizes, args.degree_main, args.degree_correction, args.omega, domain
        )
        print(f"clean points = {len(clean)}; perturbed points = {len(perturbed)}", flush=True)
        print(f"clean chi2/dof = {chi2:.8g}/{dof} = {chi2/dof:.8g}", flush=True)
        if chi2 / dof > 2:
            print("Warning: assess clean fit quality and stability before interpreting the comparison.",
                  file=sys.stderr, flush=True)
        samples, pair_samples, scaled_samples = bootstrap_comparison(
            clean, perturbed, coefficients, args.degree_main, args.degree_correction,
            args.omega, domain, args.bootstrap, args.seed,
        )
        grid = np.linspace(max(args.r_min, domain[0]), min(args.r_max, domain[1]), 300)
        reference = evaluate(coefficients, grid, args.degree_main, domain)
        curves = np.array([evaluate(c, grid, args.degree_main, domain) for c in samples])
        low, high = np.percentile(curves, [16, 84], axis=0)
        perturbed_sizes = np.array([point["L"] for point in perturbed])
        reference_at_points = evaluate(coefficients, measured[:, 0], args.degree_main, domain)
        delta = measured[:, 1] - reference_at_points
        scaled = perturbed_sizes ** args.omega * delta
        scaled_errors = scaled_samples.std(axis=0, ddof=1)
        scaled_covariance = np.atleast_2d(np.cov(scaled_samples, rowvar=False, ddof=1))
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(f"{prefix}_points.txt", np.column_stack((
            perturbed_sizes, [point["beta"] for point in perturbed], measured[:, 0],
            errors[:, 0], measured[:, 1], errors[:, 1], reference_at_points,
            delta, scaled_errors / perturbed_sizes**args.omega, scaled, scaled_errors,
        )), header="L beta Rxi err_Rxi U err_U U_clean delta_U err_delta_U scaled_delta_U err_scaled_delta_U")
        np.savetxt(f"{prefix}_clean_curve.txt", np.column_stack((grid, reference, low, high)),
                   header="Rxi U_clean low_68 high_68")
        np.savez_compressed(
            f"{prefix}_bootstrap.npz", coefficients=coefficients, clean_coefficients=samples,
            chebyshev_domain=domain, degree_main=args.degree_main, degree_correction=args.degree_correction,
            omega=args.omega, correction_size_normalization=16, seed=args.seed,
            perturbed_pairs=pair_samples, scaled_delta_U=scaled_samples,
            scaled_delta_U_covariance=scaled_covariance,
            clean_paths=[str(point["path"]) for point in clean],
            perturbed_paths=[str(point["path"]) for point in perturbed],
        )
        with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as file:
            file.write("Clean model: U = P(Rxi) + (L/16)^(-omega) Q(Rxi)\n"
                       "Correlated EIV fit; fixed block-jackknife covariance.\n"
                       "Fixed selections; independent clean/perturbed block resampling.\n"
                       "Paired Rxi,U within runs; one shared clean curve per replica.\n"
                       "Statistical errors conditional on the clean model; no perturbed fit.\n"
                       "Chebyshev basis on clean range; full residual covariance in NPZ.\n")
            for name, value in vars(args).items():
                file.write(f"{name} = {value}\n")
            file.write(f"clean observed Rxi range = {domain}\n"
                       f"clean points = {len(clean)}; perturbed points = {len(perturbed)}\n"
                       f"clean chi2/dof = {chi2:.10g}/{dof} = {chi2/dof:.10g}\n")
            np.savetxt(file, np.column_stack((coefficients, samples.std(axis=0, ddof=1))),
                       header="P0..Pp then Q0..Qq: coefficient bootstrap_std")
        save_plots(perturbed, grid, reference, low, high, scaled, scaled_errors, prefix)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Outputs: {prefix}_u_vs_rxi.png/.pdf, {prefix}_scaled_delta_u.png/.pdf")
    print(f"Data: {prefix}_points.txt, {prefix}_clean_curve.txt, {prefix}_summary.txt, {prefix}_bootstrap.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
