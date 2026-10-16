#!/usr/bin/env python3
"""Confronta fit di Rxi e U variando taglie, intervallo e gradi polinomiali."""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fit_beta_c import fit_points
from fit_scaling import find_block_files, read_block_file

# First row is the reference; subsequent rows change one choice at a time.
DEFAULT_FITS = [(16, .50, .80, 2, 1), (8, .50, .80, 2, 1),
                (24, .50, .80, 2, 1), (16, .55, .75, 2, 1),
                (16, .50, .80, 1, 1), (16, .50, .80, 3, 1),
                (16, .50, .80, 2, 0)]


def scan(points, configurations, replicas, seed, beta_min=None, beta_max=None,
         rxi_star=None, u_star=None):
    rows = []
    for l_min, r_min, r_max, p, q in configurations:
        if (not np.isfinite([l_min, r_min, r_max, p, q]).all()
                or l_min < 2 or p < 1 or q < 0 or r_min >= r_max
                or any(v != int(v) for v in (l_min, p, q))):
            raise ValueError("ogni fit richiede L_min >= 2, r_min < r_max, p >= 1, q >= 0; L_min,p,q interi")
        selected = sorted((point for point in points
                           if point["L"] >= l_min and r_min <= point["Rxi"] <= r_max),
                          key=lambda point: (point["L"], point["beta"]))
        common = dict(L_min=int(l_min), r_min=r_min, r_max=r_max, p=int(p), q=int(q),
                      points=len(selected), sizes=" ".join(map(str, sorted({v["L"] for v in selected}))))
        results = {}
        for observable, star in (("Rxi", rxi_star), ("U", u_star)):
            row = dict(common, observable=observable, critical_value=star, beta_c=np.nan,
                       error=np.nan, chi2=np.nan, dof=0, status="ok")
            try:
                (value, _, chi2, dof), samples = fit_points(
                    selected, observable, int(p), int(q), replicas, seed,
                    beta_min, beta_max, star)
                row.update(beta_c=value, error=np.std(samples[:, 0], ddof=1), chi2=chi2, dof=dof)
                results[observable] = (value, samples[:, 0])
            except (ValueError, np.linalg.LinAlgError) as error:
                row["status"] = str(error)
            rows.append(row)
        if len(results) == 2:
            r, u = results["Rxi"], results["U"]
            rows.append(dict(common, observable="U-Rxi", critical_value=None, beta_c=u[0]-r[0],
                             error=np.std(u[1]-r[1], ddof=1), chi2=np.nan, dof=0, status="ok"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="file o directory con *_blocks.txt")
    parser.add_argument("--fits", help="tabella opzionale: L_min r_min r_max p q, una riga per fit")
    parser.add_argument("--rxi-star", type=float, help="Rxi critico fissato (default: libero)")
    parser.add_argument("--u-star", type=float, help="U critico fissato (default: libero)")
    parser.add_argument("--beta-min", type=float)
    parser.add_argument("--beta-max", type=float)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-prefix", default="beta_c_stability")
    args = parser.parse_args()
    try:
        if args.bootstrap < 2:
            raise ValueError("servono almeno 2 repliche bootstrap")
        configurations = np.loadtxt(args.fits, ndmin=2) if args.fits else np.array(DEFAULT_FITS)
        if configurations.shape[1] != 5 or not len(configurations):
            raise ValueError("servono cinque colonne: L_min r_min r_max p q")
        points = [read_block_file(path) for path in find_block_files(args.inputs)]
        rows = scan(points, configurations, args.bootstrap, args.seed, args.beta_min,
                    args.beta_max, args.rxi_star, args.u_star)
        prefix = Path(args.output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with open(str(prefix) + ".csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) + ["bootstrap", "seed", "beta_min", "beta_max"])
            writer.writeheader()
            writer.writerows(dict(row, bootstrap=args.bootstrap, seed=args.seed,
                                  beta_min=args.beta_min, beta_max=args.beta_max) for row in rows)
        figure, axis = plt.subplots(figsize=(8, max(3, .5 * len(configurations))))
        for observable, offset in (("Rxi", -.12), ("U", .12)):
            series = [row for row in rows if row["observable"] == observable]
            valid = [(i, row) for i, row in enumerate(series) if row["status"] == "ok"]
            if valid:
                axis.errorbar([r["beta_c"] for _, r in valid], [i+offset for i, _ in valid],
                              xerr=[r["error"] for _, r in valid], fmt="o", label=observable)
        axis.set_yticks(range(len(configurations)),
                       [f"L≥{l:g}, R={lo:g}–{hi:g}, p={p:g}, q={q:g}"
                        for l, lo, hi, p, q in configurations])
        axis.invert_yaxis()
        axis.set_xlabel(r"$\beta_c$ (statistical errors)")
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        if axis.lines:
            axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(str(prefix) + ".pdf")
        plt.close(figure)
        failures = [row for row in rows if row["status"] != "ok"]
        for row in failures:
            print(f"Skipped {row['observable']}, L_min={row['L_min']}, "
                  f"R={row['r_min']}:{row['r_max']}, p={row['p']}, q={row['q']}: {row['status']}")
        print(f"{prefix}.csv\n{prefix}.pdf\n{len(failures)} failed fits; inspect before estimating systematics.")
        return 0 if any(row["status"] == "ok" for row in rows) else 1
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
