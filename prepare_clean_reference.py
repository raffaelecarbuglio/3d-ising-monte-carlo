#!/usr/bin/env python3
"""Prepare the fixed clean Wolff reference once, including its bootstrap fits."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from compare_universality import (
    CLEAN_LMIN, DEGREE_MAIN, DEGREE_CORRECTION, R_MIN, R_MAX,
    evaluate, load_points, resample_pairs,
)
from fit_scaling import OMEGA


def basis(r, sizes, p, q, omega, domain):
    """Chebyshev basis: same polynomial space, better numerical conditioning."""
    low, high = domain
    t = (2 * np.asarray(r) - low - high) / (high - low)
    main = np.polynomial.chebyshev.chebvander(t, p)
    correction = np.polynomial.chebyshev.chebvander(t, q)
    # Normalizing L at 16 changes only the correction coefficients.
    factor = (np.asarray(sizes, dtype=float) / 16) ** (-omega)
    return np.column_stack((main, factor[:, None] * correction))


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="clean block files or directories")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="clean_reference.npz")
    args = parser.parse_args(argv)
    try:
        if args.bootstrap < 2 or args.seed < 0:
            raise ValueError("at least two bootstrap replicas and a nonnegative seed are required")
        if Path(args.output).suffix != ".npz":
            raise ValueError("reference output must end in .npz")
        points = load_points(args.inputs, lmin=CLEAN_LMIN)
        pairs = np.array([point["pair"] for point in points])
        covariance = np.array([point["covariance"] for point in points])
        sizes = np.array([point["L"] for point in points])
        domain = (float(pairs[:, 0].min()), float(pairs[:, 0].max()))
        coefficients, chi2, dof = fit_clean(
            pairs, covariance, sizes, DEGREE_MAIN, DEGREE_CORRECTION, OMEGA, domain
        )
        print(f"clean points = {len(points)}; chi2/dof = {chi2:.8g}/{dof} = {chi2/dof:.8g}", flush=True)
        if chi2 / dof > 2:
            print("Warning: assess clean fit quality before using this reference.", file=sys.stderr)
        rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[0])
        samples = np.empty((args.bootstrap, len(coefficients)))
        for b in range(args.bootstrap):
            samples[b], _, _ = fit_clean(
                resample_pairs(points, rng), covariance, sizes,
                DEGREE_MAIN, DEGREE_CORRECTION, OMEGA, domain, coefficients,
            )
            if (b + 1) % 50 == 0 or b + 1 == args.bootstrap:
                print(f"clean bootstrap {b + 1}/{args.bootstrap}", flush=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output, coefficients=coefficients, clean_coefficients=samples,
            chebyshev_domain=domain, degree_main=DEGREE_MAIN, degree_correction=DEGREE_CORRECTION,
            omega=OMEGA, correction_size_normalization=16, clean_lmin=CLEAN_LMIN,
            r_min=R_MIN, r_max=R_MAX, chi2=chi2, dof=dof, seed=args.seed,
            clean_paths=[str(point["path"]) for point in points],
        )
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Saved reference: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
