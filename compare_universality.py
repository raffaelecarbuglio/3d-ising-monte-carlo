#!/usr/bin/env python3
"""Compare perturbed U(Rxi,L) with a saved clean Wolff reference."""

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_scaling import OMEGA, asymptotic_curve, bootstrap_pair, find_block_files, read_block_file
from plot_u_vs_rxi import plot_points

CLEAN_LMIN, DEGREE_MAIN, DEGREE_CORRECTION = 16, 6, 3
R_MIN, R_MAX = 0.30, 1.00


def load_points(inputs, sizes=None):
    points, identities = [], set()
    for path in sorted({path.resolve() for path in find_block_files(inputs)}):
        point = read_block_file(path)
        if (sizes is not None and point["L"] not in sizes) or not R_MIN <= point["Rxi"] <= R_MAX:
            continue
        values = [point[key] for key in ("L", "beta", "Rxi", "err_Rxi", "U", "err_U")]
        if (not np.isfinite(values).all() or point["L"] < 2
                or min(point["err_Rxi"], point["err_U"]) < 0
                or len(point["blocks"]) < 2 or not np.isfinite(point["blocks"]).all()):
            raise ValueError(f"{path}: invalid observables, errors or blocks")
        identity = (point["L"], point["beta"])
        if identity in identities:
            raise ValueError(f"duplicate L,beta: {identity}; supply one analysis per simulation")
        identities.add(identity)
        point["path"] = path
        points.append(point)
    points.sort(key=lambda point: (point["L"], point["beta"]))
    if not points:
        raise ValueError("no points selected")
    if sizes is not None and set(sizes) != {point["L"] for point in points}:
        raise ValueError("a requested size has no points in the selected Rxi window")
    return points


def evaluate(coefficients, r):
    return asymptotic_curve(coefficients, r, DEGREE_MAIN)


def load_reference(filename):
    with np.load(filename, allow_pickle=False) as archive:
        ref = dict(archive)
    settings = dict(clean_lmin=CLEAN_LMIN, degree_main=DEGREE_MAIN,
                    degree_correction=DEGREE_CORRECTION, omega=OMEGA,
                    basis="power", fit_method="vertical")
    c, samples, domain = (ref[key] for key in ("coefficients", "clean_coefficients", "r_range"))
    if (any(ref[key] != value for key, value in settings.items()) or c.shape != (11,)
            or samples.ndim != 2 or samples.shape[1] != 11 or len(samples) < 2
            or domain.shape != (2,) or not domain[0] < domain[1]
            or not all(np.isfinite(value).all() for value in (c, samples, domain))):
        raise ValueError("invalid reference; use the bundled clean_reference.npz")
    return ref


def bootstrap_comparison(perturbed, samples, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    factors = np.array([point["L"] for point in perturbed]) ** OMEGA
    pair_samples = np.empty((len(samples), len(perturbed), 2))
    delta_samples = np.empty((len(samples), len(perturbed)))
    for b, coefficients in enumerate(samples):
        pairs = np.array([bootstrap_pair(point, rng) for point in perturbed])
        pair_samples[b] = pairs
        # The same saved clean replica is shared by ALL perturbed points.
        delta_samples[b] = factors * (pairs[:, 1] - evaluate(coefficients, pairs[:, 0]))
        if (b + 1) % 50 == 0 or b + 1 == len(samples):
            print(f"bootstrap {b + 1}/{len(samples)}", flush=True)
    return pair_samples, delta_samples


def save_plots(data, grid, reference, low, high, scaled, scaled_errors, prefix):
    data = data.copy()
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
    parser.add_argument("--reference", default=str(Path(__file__).with_name("clean_reference.npz")))
    parser.add_argument("--perturbed", nargs="+", required=True, help="block files or directories")
    parser.add_argument("--perturbed-sizes", nargs="+", type=int)
    parser.add_argument("--bootstrap", type=int, help="use the first N saved replicas; default uses all")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-prefix", default="universality")
    args = parser.parse_args(argv)
    try:
        ref = load_reference(args.reference)
        replicas = len(ref["clean_coefficients"]) if args.bootstrap is None else args.bootstrap
        if not 2 <= replicas <= len(ref["clean_coefficients"]) or args.seed < 0:
            raise ValueError("request 2..N saved replicas and a nonnegative seed")
        samples = ref["clean_coefficients"][:replicas]
        coefficients, domain = ref["coefficients"], ref["r_range"]
        perturbed = load_points(args.perturbed, sizes=args.perturbed_sizes)
        if set(ref["clean_hashes"]) & {
            hashlib.sha256(point["path"].read_bytes()).hexdigest() for point in perturbed
        }:
            raise ValueError("clean and perturbed inputs must be disjoint, independent ensembles")
        data = np.array([[point[key] for key in ("L", "beta", "Rxi", "err_Rxi", "U", "err_U")]
                         for point in perturbed])
        sizes, r, u = data[:, 0], data[:, 2], data[:, 4]
        if np.any(r < domain[0]) or np.any(r > domain[1]):
            raise ValueError("perturbed central points extend beyond the saved clean reference range")
        prefix = Path(args.output_prefix)
        if Path(f"{prefix}_bootstrap.npz").resolve() == Path(args.reference).resolve():
            raise ValueError("output would overwrite the clean reference; choose another prefix")
        chi2, dof = float(ref["chi2"]), int(ref["dof"])
        print(f"loaded clean reference: {args.reference}; chi2/dof = {chi2:.8g}/{dof} = {chi2/dof:.8g}")
        pair_samples, scaled_samples = bootstrap_comparison(perturbed, samples, args.seed)
        grid = np.linspace(max(R_MIN, domain[0]), min(R_MAX, domain[1]), 300)
        reference = evaluate(coefficients, grid)
        curves = np.array([evaluate(c, grid) for c in samples])
        low, high = np.percentile(curves, [16, 84], axis=0)
        reference_at_points = evaluate(coefficients, r)
        delta = u - reference_at_points
        scaled = sizes ** OMEGA * delta
        scaled_errors = scaled_samples.std(axis=0, ddof=1)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(f"{prefix}_points.txt", np.column_stack((
            data, reference_at_points, delta,
            scaled_errors / sizes**OMEGA, scaled, scaled_errors,
        )), header="L beta Rxi err_Rxi U err_U U_clean delta_U err_delta_U scaled_delta_U err_scaled_delta_U")
        np.savez_compressed(f"{prefix}_bootstrap.npz", **dict(
            ref, clean_coefficients=samples, reference_seed=ref["seed"], seed=args.seed,
            reference_file=str(Path(args.reference).resolve()), perturbed_pairs=pair_samples,
            scaled_delta_U=scaled_samples,
            scaled_delta_U_covariance=np.atleast_2d(np.cov(scaled_samples, rowvar=False, ddof=1)),
            clean_curve=np.column_stack((grid, reference, low, high)),
            perturbed_paths=[str(point["path"]) for point in perturbed],
        ))
        save_plots(data, grid, reference, low, high, scaled, scaled_errors, prefix)
    except (OSError, KeyError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Outputs: {prefix}_u_vs_rxi.png/.pdf, {prefix}_scaled_delta_u.png/.pdf")
    print(f"Data: {prefix}_points.txt, {prefix}_bootstrap.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
