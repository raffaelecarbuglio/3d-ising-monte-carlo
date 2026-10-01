#!/usr/bin/env python3
"""Compare perturbed U(Rxi,L) with a saved clean Wolff reference."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_scaling import OMEGA, bootstrap_pair, find_block_files, observable_pairs, read_block_file
from plot_u_vs_rxi import plot_points

CLEAN_LMIN, DEGREE_MAIN, DEGREE_CORRECTION = 16, 6, 3
R_MIN, R_MAX = 0.30, 1.00


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


def evaluate(coefficients, r, p, domain, sizes=None, omega=OMEGA):
    low, high = domain
    t = (2 * np.asarray(r) - low - high) / (high - low)
    result = np.polynomial.chebyshev.chebval(t, np.asarray(coefficients)[:p + 1])
    if sizes is not None:
        result += (np.asarray(sizes) / 16) ** (-omega) * np.polynomial.chebyshev.chebval(
            t, np.asarray(coefficients)[p + 1:]
        )
    return result


def resample_pairs(points, rng):
    return np.array([bootstrap_pair(point, rng) for point in points])


def load_reference(filename):
    keys = ("coefficients", "clean_coefficients", "chebyshev_domain", "clean_paths",
            "clean_lmin", "degree_main", "degree_correction", "omega", "r_min", "r_max",
            "chi2", "dof", "seed", "correction_size_normalization")
    try:
        with np.load(filename, allow_pickle=False) as archive:
            reference = {key: archive[key] for key in keys}
    except KeyError as error:
        raise ValueError("incomplete reference; generate it with prepare_clean_reference.py") from error
    settings = dict(clean_lmin=CLEAN_LMIN, degree_main=DEGREE_MAIN,
                    degree_correction=DEGREE_CORRECTION, omega=OMEGA,
                    r_min=R_MIN, r_max=R_MAX, correction_size_normalization=16)
    for name, expected in settings.items():
        if reference[name].ndim != 0 or reference[name] != expected:
            raise ValueError(f"reference {name} must be {expected}")
    c, samples, domain = [reference[key] for key in
                          ("coefficients", "clean_coefficients", "chebyshev_domain")]
    if (c.shape != (DEGREE_MAIN + DEGREE_CORRECTION + 2,) or samples.ndim != 2
            or samples.shape[1:] != c.shape or len(samples) < 2
            or domain.shape != (2,) or not domain[0] < domain[1]
            or not all(np.isfinite(value).all() for value in (c, samples, domain))):
        raise ValueError("invalid reference coefficients, replicas or domain")
    if (reference["chi2"].ndim != 0 or not np.isfinite(reference["chi2"])
            or reference["chi2"] < 0 or reference["dof"].ndim != 0 or reference["dof"] <= 0
            or not np.isfinite(reference["dof"]) or reference["clean_paths"].ndim != 1
            or not len(reference["clean_paths"])):
        raise ValueError("invalid reference fit metadata")
    return reference


def bootstrap_comparison(perturbed, samples, domain, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
    factors = np.array([point["L"] for point in perturbed]) ** OMEGA
    pair_samples = np.empty((len(samples), len(perturbed), 2))
    delta_samples = np.empty((len(samples), len(perturbed)))
    for b, coefficients in enumerate(samples):
        pairs = resample_pairs(perturbed, rng)
        pair_samples[b] = pairs
        # The same saved clean replica is shared by ALL perturbed points.
        delta_samples[b] = factors * (pairs[:, 1] - evaluate(coefficients, pairs[:, 0], DEGREE_MAIN, domain))
        if (b + 1) % 50 == 0 or b + 1 == len(samples):
            print(f"bootstrap {b + 1}/{len(samples)}", flush=True)
    return pair_samples, delta_samples


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
    parser.add_argument("--reference", default="clean_reference.npz")
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
            raise ValueError("request 2..N saved replicas and a nonnegative seed; prepare more if needed")
        samples = ref["clean_coefficients"][:replicas]
        coefficients, domain = ref["coefficients"], ref["chebyshev_domain"]
        perturbed = load_points(args.perturbed, sizes=args.perturbed_sizes, rmin=R_MIN, rmax=R_MAX)
        if set(ref["clean_paths"]) & {str(point["path"]) for point in perturbed}:
            raise ValueError("clean and perturbed inputs must be disjoint, independent ensembles")
        measured = np.array([point["pair"] for point in perturbed])
        errors = np.sqrt([np.diag(point["covariance"]) for point in perturbed])
        sizes = np.array([point["L"] for point in perturbed])
        if np.any(measured[:, 0] < domain[0]) or np.any(measured[:, 0] > domain[1]):
            raise ValueError("perturbed central points extend beyond the saved clean reference range")
        prefix = Path(args.output_prefix)
        if Path(f"{prefix}_bootstrap.npz").resolve() == Path(args.reference).resolve():
            raise ValueError("output would overwrite the clean reference; choose another prefix")
        chi2, dof = float(ref["chi2"]), int(ref["dof"])
        print(f"loaded clean reference: {args.reference}; chi2/dof = {chi2:.8g}/{dof} = {chi2/dof:.8g}")
        if chi2 / dof > 2:
            print("Warning: assess clean fit quality before interpreting the comparison.", file=sys.stderr)
        pair_samples, scaled_samples = bootstrap_comparison(perturbed, samples, domain, args.seed)
        grid = np.linspace(max(R_MIN, domain[0]), min(R_MAX, domain[1]), 300)
        reference = evaluate(coefficients, grid, DEGREE_MAIN, domain)
        curves = np.array([evaluate(c, grid, DEGREE_MAIN, domain) for c in samples])
        low, high = np.percentile(curves, [16, 84], axis=0)
        reference_at_points = evaluate(coefficients, measured[:, 0], DEGREE_MAIN, domain)
        delta = measured[:, 1] - reference_at_points
        scaled = sizes ** OMEGA * delta
        scaled_errors = scaled_samples.std(axis=0, ddof=1)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(f"{prefix}_points.txt", np.column_stack((
            sizes, [point["beta"] for point in perturbed], measured[:, 0], errors[:, 0],
            measured[:, 1], errors[:, 1], reference_at_points, delta,
            scaled_errors / sizes**OMEGA, scaled, scaled_errors,
        )), header="L beta Rxi err_Rxi U err_U U_clean delta_U err_delta_U scaled_delta_U err_scaled_delta_U")
        np.savetxt(f"{prefix}_clean_curve.txt", np.column_stack((grid, reference, low, high)),
                   header="Rxi U_clean low_68 high_68")
        np.savez_compressed(
            f"{prefix}_bootstrap.npz", coefficients=coefficients, clean_coefficients=samples,
            chebyshev_domain=domain, degree_main=DEGREE_MAIN, degree_correction=DEGREE_CORRECTION,
            omega=OMEGA, correction_size_normalization=16, clean_lmin=CLEAN_LMIN,
            r_min=R_MIN, r_max=R_MAX, chi2=chi2, dof=dof, reference_seed=ref["seed"], seed=args.seed,
            reference_file=str(Path(args.reference).resolve()), perturbed_pairs=pair_samples,
            scaled_delta_U=scaled_samples,
            scaled_delta_U_covariance=np.atleast_2d(np.cov(scaled_samples, rowvar=False, ddof=1)),
            clean_paths=ref["clean_paths"], perturbed_paths=[str(point["path"]) for point in perturbed],
        )
        with open(f"{prefix}_summary.txt", "w", encoding="utf-8") as file:
            file.write("Saved correlated clean reference; no clean fit or clean resampling in this run.\n"
                       "Fixed selections; perturbed Rxi,U paired; one shared clean replica per bootstrap.\n"
                       "Statistical errors conditional on the clean model; full residual covariance in NPZ.\n"
                       f"clean_lmin = {CLEAN_LMIN}; degrees = {DEGREE_MAIN}, {DEGREE_CORRECTION}\n"
                       f"omega = {OMEGA}; perturbed window = {R_MIN}, {R_MAX}\n"
                       f"clean chi2/dof = {chi2:.10g}/{dof} = {chi2/dof:.10g}\n"
                       f"clean points = {len(ref['clean_paths'])}; perturbed points = {len(perturbed)}\n"
                       f"bootstrap replicas = {replicas}; reference seed = {int(ref['seed'])}\n")
            for name, value in vars(args).items():
                file.write(f"{name} = {value}\n")
        save_plots(perturbed, grid, reference, low, high, scaled, scaled_errors, prefix)
    except (OSError, ValueError, np.linalg.LinAlgError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Outputs: {prefix}_u_vs_rxi.png/.pdf, {prefix}_scaled_delta_u.png/.pdf")
    print(f"Data: {prefix}_points.txt, {prefix}_clean_curve.txt, {prefix}_summary.txt, {prefix}_bootstrap.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
