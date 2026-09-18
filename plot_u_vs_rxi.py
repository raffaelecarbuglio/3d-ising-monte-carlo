#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Grafico di Binder U in funzione di R_xi")
    parser.add_argument("filename", help="riepilogo prodotto con --summary-output")
    parser.add_argument("--output", default="plots/u_vs_rxi.png", help="file del grafico")
    arguments = parser.parse_args()

    try:
        data = np.loadtxt(arguments.filename, comments="#", ndmin=2)
        if data.size == 0 or data.shape[1] != 6:
            raise ValueError("il riepilogo deve contenere 6 colonne: L beta Rxi err_Rxi U err_U")
        if not np.all(np.isfinite(data)):
            raise ValueError("il riepilogo contiene valori non finiti")
        if np.any(data[:, 0] <= 0) or np.any(data[:, 0] != np.floor(data[:, 0])):
            raise ValueError("L deve essere un intero positivo")
        if np.any(data[:, 3] < 0) or np.any(data[:, 5] < 0):
            raise ValueError("gli errori devono essere non negativi")

        figure, axis = plt.subplots(figsize=(6.0, 4.5))
        for lattice_size in np.unique(data[:, 0]):
            points = data[data[:, 0] == lattice_size]
            points = points[np.argsort(points[:, 2])]
            axis.errorbar(
                points[:, 2], points[:, 4],
                xerr=points[:, 3], yerr=points[:, 5],
                marker="x", linestyle="none",
                markersize=5.5, markeredgewidth=1.0,
                elinewidth=0.8, capsize=2.5, capthick=0.8,
                label=f"L={int(lattice_size)}",
            )

        axis.set_xlabel(r"$R_\xi$", fontsize=13)
        axis.set_ylabel(r"$U$", fontsize=13)
        axis.tick_params(direction="in", top=True, right=True, labelsize=11)
        axis.legend(frameon=False, fontsize=10)
        figure.tight_layout()

        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=300, bbox_inches="tight")

        # Mantieni anche una copia vettoriale adatta a tesi e pubblicazioni.
        pdf_path = output_path.with_suffix(".pdf")
        if pdf_path != output_path:
            figure.savefig(pdf_path, bbox_inches="tight")
    except (OSError, ValueError) as error:
        print(f"Errore: {error}", file=sys.stderr)
        return 1

    print(f"Grafico salvato in {output_path}")
    if pdf_path != output_path:
        print(f"Copia vettoriale salvata in {pdf_path}")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
