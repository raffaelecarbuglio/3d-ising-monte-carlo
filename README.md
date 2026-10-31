# 3D Ising Monte Carlo

MSc thesis project in Theoretical Physics at the University of Pisa, studying whether controlled numerical perturbations change the universal critical properties of the 3D Ising model.

- **C11:** Metropolis and Wolff simulations on a periodic cubic lattice.
- **Perturbations:** independent Gaussian noise or deterministic rounding of the Metropolis energy difference.
- **Python:** observables, blocking, jackknife errors, and finite-size-scaling fits with block bootstrap.
- **Z2 gauge extension:** a separate C11 simulator with periodic boundaries and energy measurements, sharing the Ising code's random-number generator.

## Quick start

Requires GCC, Make, and Python 3. From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make
./ising inputs/input_example.dat
python3 analyze.py data/example_data.dat --block-size 2
make test
make test-analysis
```

Tests are kept in [`tests/`](tests/); `make test` runs both C test suites and `make test-analysis` runs the Python analysis tests.

The bundled input is a tiny demonstration. For scientific runs, increase thermalization and sampling, and choose a block size using [blocking_plot.py](blocking_plot.py).

Edit [inputs/input_example.dat](inputs/input_example.dat) to set lattice size, temperature, run length, seed, and algorithm. Optional final fields `sigma` and `lambda` default to zero. Set `sigma > 0` for Gaussian noise, or `lambda > 0` for `round(lambda * delta_E) / lambda`; enable only one. Wolff requires both to be zero. One Wolff “sweep” means one cluster update.

For rounding, try `sigma = 0` and `lambda = 0.55`. The four families at strength `1/m = 1/2` use lambda `0.55, 0.60, 0.40, 0.45`. Keep lambda fixed while scanning beta; locate the shifted transition before production runs. Outputs record lambda and the rounded costs and ratios. A restart into the same data file requires the same sigma and lambda. `run_metropolis.sh` and `cluster/prepare.sh` accept `LAMBDA=...` (default zero), alongside `SIGMA=...`.

## Campaigns from a configuration file

Edit [`campaigns/sigma5.conf`](campaigns/sigma5.conf), then launch from the project directory:

```bash
nohup bash simulate.sh campaigns/sigma5.conf > sigma5.log 2>&1 &
```

The launcher builds `ising`, creates a unique directory under `runs/`, and prints its path.
It saves `campaign.conf`, individual `.in` inputs, simulation logs, and a `runs.csv`
status table there. Each `BETA_GRID` row contains `L` followed by explicit beta values.
Seeds are `BASE_SEED`, `BASE_SEED+1`, etc., in grid order; larger lattices launch first,
in groups of up to `MAX_JOBS`. An unsuccessful group stops the campaign before the next group.

Choose an unused `BASE_SEED` before each campaign. The launcher rejects reuse against
saved `.in` files in `SEED_HISTORY` (by default this project); add other directories if
older inputs live elsewhere. It cannot check missing inputs or reserve seeds against
another launcher preparing a campaign at exactly the same time. Failed preparation
removes its unused batch; once simulations start, logs and inputs are retained.
Existing history/history duplicates do not block an unrelated new seed range.

Inspect representative blocking plateaus for each size before analysis, for example:

```bash
python3 blocking_plot.py runs/YOUR_BATCH/results/L32_b0p383703000_data.dat
```

Set `BLOCK_SIZES` **in that batch's saved `campaign.conf`**, using `L:measurements`
pairs such as `"8:256 16:512 24:1024 32:2048"` (illustrative values, not a plateau check).
Then run:

```bash
bash analyse.sh runs/YOUR_BATCH
```

The wrapper requires every simulation to have status `OK`, runs `analyze_grid.sh`,
and compares with `REFERENCE` if it is nonempty. Set `REFERENCE=""` to skip that comparison.
Analysis uses the saved batch settings; editing the original campaign does not affect old batches.
To switch to Wolff, copy the config and set `ALGORITHM=wolff`, `SIGMA=0`, `LAMBDA=0`,
then choose an appropriate beta grid, run lengths and unused seeds. One Wolff sweep
is one cluster update.

Paths in configurations are relative to the project directory; output paths cannot
contain spaces or commas because of the C input and CSV formats. The `.conf` files
are ordinary Bash code and should be files you wrote or reviewed. The existing
`run_metropolis.sh`, `run_wolff.sh` and direct `analyze_grid.sh` commands still work.
These new launch commands run on the local machine; use the existing `cluster/`
workflow for Slurm submission on INFN.

## Z2 gauge model

The gauge simulator is in [`z2_gauge/`](z2_gauge/). From the repository root:

```bash
make z2
make test-z2
./z2 z2_gauge/input_example.dat
```

Edit [`z2_gauge/input_example.dat`](z2_gauge/input_example.dat) to set the run parameters. The example writes `sweep`, `total_energy`, and `acceptance` to `data/z2_energy.dat` and refuses to overwrite an existing file.
The energy is `E = -sum_p P_p`, without volume normalisation. Each sweep visits all links with a 50% chance of proposing each flip, followed by the usual Metropolis acceptance rule.
This first version supports the unperturbed model and ordered/random starts. Its short example is a demonstration; choose production run lengths after checking thermalisation and autocorrelations. See [`z2_gauge/README.md`](z2_gauge/README.md) for the geometry, sweep convention, input format, and checks.

## Further analysis

- [fit_scaling.py](fit_scaling.py): universal scaling curve and finite-size corrections.
- [compare_universality.py](compare_universality.py): perturbed data against the saved Wolff reference.
- [fit_beta_c.py](fit_beta_c.py): critical inverse temperature from `--observable Rxi` (default) or `U`, assuming Ising exponents. Optional `--critical-value VALUE` fixes the observable's critical value.
- [check_beta_c_stability.py](check_beta_c_stability.py): both observables, varying minimum size, fit window and polynomial degrees; CSV results and a comparison PDF.

For example: `python3 check_beta_c_stability.py runs/sigma5 --output-prefix results/sigma5_stability`.
The seven default fits change one choice at a time from `L_min=16`, `0.50 <= Rxi <= 0.80`, degrees `(2,1)`.
Override them with `--fits FILE`: five whitespace-separated columns `L_min r_min r_max p q`, one fit per row.
Both observables use the same points and paired block resamples; CSV rows `U-Rxi` report the difference between their beta estimates with its correlated statistical error.
Failed fits are recorded, not silently discarded. At least three sizes are required with a free critical value, or two with a fixed value. Inspect chi-squared and stability before assigning systematic errors; the scan does not automatically turn the spread into an uncertainty.
Use `--rxi-star VALUE --u-star VALUE` for fixed critical values; their uncertainties are not included in these conditional fits.

For the clean-curve check, add `--critical-point RXI ERR_RXI U ERR_U` to `fit_scaling.py`, using literature values in the same conventions and geometry.
The summary reports `U_infinity(Rxi_star)` and its difference from the reference, with linear propagation of reference uncertainties and bootstrap fit uncertainty. `--critical-covariance VALUE` optionally supplies the reference covariance (default zero); reference values are assumed independent of the simulations. Fit systematics remain separate.

Use `python3 <script> --help` for analysis options. Parameter scans are provided by [run_metropolis.sh](run_metropolis.sh) and [run_wolff.sh](run_wolff.sh).

For INFN submission scripts, see [cluster/README.md](cluster/README.md).

*Active thesis project; results and analysis are evolving.*
