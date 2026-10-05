# 3D Ising Monte Carlo simulator

This repository contains the simulation and analysis code developed for my MSc thesis in Theoretical Physics at the University of Pisa.

The project studies the **3D Ising model** and the robustness of Monte Carlo methods under controlled numerical perturbations. The simulator is written in **C11**; analysis, finite-size scaling, bootstrap resampling, and plotting are implemented in **Python**.

The code is intentionally compact and readable, with an emphasis on reproducibility, statistical validation, and a clear separation between simulation and analysis.

## Main features

- 3D cubic Ising lattice with periodic boundary conditions and \(J=1\)
- local **Metropolis** updates
- **Wolff cluster** updates
- PCG32 pseudorandom-number generator
- ordered, random, and restart initial conditions
- checkpoint/restart support for long simulations
- reproducible seeds and configurable measurement intervals
- automated parameter scans with Bash
- blocking and jackknife uncertainty estimation
- Binder cumulant, susceptibility, and second-moment correlation length
- finite-size-scaling fits with corrections to scaling
- paired/block bootstrap resampling that preserves correlations between derived observables
- controlled Gaussian perturbations of the Metropolis acceptance step
- automated tests for both the C simulator and Python analysis

## Repository structure

```text
.
├── main.c, ising.c, rng.c, input.c   C simulator
├── analyze.py                        observable and uncertainty analysis
├── blocking_plot.py                  blocking-plateau diagnostics
├── fit_scaling.py                    finite-size-scaling fit and bootstrap
├── fit_beta_c.py                     critical-beta finite-size-scaling fit
├── clean_reference.npz               fixed Wolff fit and 5000 bootstrap fits
├── compare_universality.py           perturbed data vs saved clean reference
├── plot_u_vs_rxi.py                  U vs R_xi visualization
├── run_metropolis.sh                 Metropolis parameter scans
├── run_wolff.sh                      Wolff parameter scans
├── run_and_analyze.sh                combined simulation/analysis workflow
├── analyze_grid.sh                   batch analysis
├── tests.c                           C tests
├── test_analysis.py                  Python analysis tests
├── test_fit_scaling.py               scaling-fit tests
├── test_fit_beta_c.py                critical-beta fit tests
├── test_compare_universality.py       clean-reference and bootstrap tests
├── inputs/                           example input files
├── plots/                            example diagnostic plots
└── Makefile
```

Generated simulation data and checkpoints are excluded from version control.

## Build and test

The C code requires GCC, Make, and the standard math library:

```bash
make
make test
```

The Python analysis requires Python 3, NumPy, and Matplotlib:

```bash
python3 -m pip install -r requirements.txt
make test-analysis
```

To remove compiled objects and temporary files:

```bash
make clean
```

The simulator is currently a serial CPU implementation and does not require external C libraries.

The default build uses `-O3 -flto` (link-time optimization). Run `make clean`
before rebuilding after changing compiler flags. To disable LTO, use
`make CFLAGS='-std=c11 -O3 -Wall -Wextra -Wpedantic'` after cleaning.

Metropolis computes the sum of the six neighboring spins directly for each
proposed flip.

## Running a simulation

The executable takes a text input file:

```bash
./ising inputs/input_example.dat
```

A typical input is

```text
L = 6
beta = 0.22
n_therm = 20
n_sweeps = 40
measure_every = 5
seed = 12345
start = ordered
algorithm = metropolis
config_file = data/example_config.dat
data_file = data/example_data.dat
sigma = 0.0
```

The main parameters are:

- `L`: lattice linear size
- `beta`: inverse temperature
- `n_therm`: thermalization sweeps
- `n_sweeps`: production sweeps
- `measure_every`: interval between stored measurements
- `seed`: PCG32 initialization seed
- `start`: `ordered`, `random`, or `restart`
- `algorithm`: `metropolis` or `wolff`
- `config_file`: checkpoint path
- `data_file`: measurement output path
- `sigma`: Gaussian noise amplitude for the Metropolis perturbation study

For Wolff, one "sweep" in the present implementation means one cluster update.

## Restartable runs

Each run saves the lattice configuration and completed production-sweep count every 100,000 production sweeps, and again at normal completion. The interval is fixed by `save_every` in `main.c`; no input setting is required. Thermalization sweeps are not counted. For Wolff, the interval counts cluster updates.

Each save replaces `config_file` through a temporary file in the same directory, after the write and close succeed. Measurements are flushed before periodic saves. The configuration format is unchanged, so a later run can continue from that configuration:

```bash
./ising inputs/input_restart_example.dat
```

On restart, new measurements are appended to the existing data file and the production-sweep counter continues from the previous segment. The random-number generator is initialized from the new seed supplied in the restart input.

After an interruption, the measurement file may contain rows beyond the saved production-sweep count. Before appending a restart, manually remove those later rows, or use a new `data_file`. The program does not truncate measurements or save the random-number generator state, so restarting does not reproduce the interrupted trajectory exactly.

## Stored measurements

Both Metropolis and Wolff write four numerical columns:

```text
sweep energy magnetization g_min
```

`energy` and `magnetization` are stored as integer totals. `g_min` is the structure-factor estimator at the minimum non-zero momentum, averaged over the three equivalent lattice directions.

Several derived quantities are intentionally reconstructed during analysis instead of being stored redundantly.

With \(V=L^3\),

```text
m      = M / V
g_zero = M^2 / V
e      = -E / (3 V)
```

The second-moment correlation length is computed from ensemble averages,

```text
xi = sqrt(<g_zero>/<g_min> - 1) / (2 sin(pi/L))
```

rather than configuration by configuration.

## Statistical analysis

Run the analysis by specifying a block size:

```bash
python3 analyze.py data/example_data.dat --block-size 100
```

The analysis computes, among other quantities:

- energy and magnetization observables
- Binder cumulant
- magnetic susceptibility
- second-moment correlation length \(\xi\)
- \(R_\xi=\xi/L\)
- block-jackknife uncertainties

Nonlinear observables are recomputed inside each jackknife sample.

The script also writes a `*_blocks.txt` file containing block-level averages of the quantities required to reconstruct \(R_\xi\) and the Binder cumulant. These block data are then used by the bootstrap scaling analysis.

`analyze.py` reads the time series in chunks targeting **100,000 measurements**.
The chunk size is rounded down to a whole number of jackknife blocks; if one
statistical block is larger than the target, one block is read at a time.
With `--block-size 2000`, each chunk contains 50 blocks. Only seven primary
sums per complete block are retained in RAM; raw chunks are discarded after
processing. Reading boundaries do not change statistical blocks, and only the
final incomplete block is excluded, as in the full-array calculation.

The observable and jackknife formulas and all text output formats are retained.
Central values are reconstructed from the block sums; grouping floating-point
additions differently can change their last digits. Tests compare the chunked
calculation with both the full-array and brute-force leave-one-block-out
calculations. Compact four-column and legacy seven-column files, gzip,
restart comments, blank lines, and final incomplete blocks are supported.
No analysis cache is created.

A synthetic test with **10 million measurements and block size 2000** measured
**43 MiB peak RAM** for chunked analysis, compared with **1090 MiB** for the
full-array calculation, about a **25-fold reduction**. The block averages
matched exactly. Actual RAM use depends on the number of statistical blocks,
block size, input format and Python/NumPy environment. RAM scales with one
reading chunk plus the retained block sums and jackknife arrays.

Batch analysis uses **eight concurrent processes by default**, with one numerical
library thread per process. This setting is independent of simulation
`MAX_JOBS`. Each process has private log and summary files; the launcher merges
results in input order and stops on failed analyses.

```bash
./analyze_grid.sh "$RUN_DIR" 8:32 16:32 24:64 32:64
# Override the number of concurrent analyses:
ANALYSIS_JOBS=20 ./analyze_grid.sh "$RUN_DIR" 8:32 16:32 24:64 32:64
```

`ANALYSIS_JOBS` also propagates through `run_and_analyze.sh`. The default remains
eight for the professor's machine showing 125 GiB RAM. Choose a higher count
according to available cores and disk throughput, as well as available RAM.

## Blocking diagnostics

Monte Carlo measurements are correlated. To study whether the uncertainty estimate has reached a stable blocking regime:

```bash
python3 blocking_plot.py data/example_data.dat
```

The resulting plot shows how estimated errors change as the number of measurements per block is increased.
This diagnostic still reads the full time series; chunked reading applies to
`analyze.py`. The full-array functions remain available for this diagnostic and
for validation.

## Finite-size scaling

The global scaling fit uses

```text
U(R_xi, L) =
    sum_k b_k R_xi^k
    + L^(-omega) sum_k c_k R_xi^k
```

with the 3D Ising correction-to-scaling exponent fixed to

```text
omega = 0.8295
```

from the literature.

Example:

```bash
MPLBACKEND=Agg python3 fit_scaling.py beta_runs_L16-24-32 beta_runs_L48-64 \
    --sizes 16 24 32 48 64 \
    --degree-main 6 \
    --degree-correction 3 \
    --bootstrap 2000 \
    --output-prefix scaling_fit
```

For each bootstrap replica, the block rows of each simulation are sampled with replacement. The same resampled blocks are used to reconstruct both \(R_\xi\) and \(U\), so their statistical correlation is preserved naturally.

The script produces:

- fit coefficients and bootstrap uncertainties
- \(\chi^2/mathrm{dof}\)
- the asymptotic scaling curve
- a pointwise 68% bootstrap band
- PNG and PDF plots

The fit can be rerun with different lattice-size cuts and polynomial degrees to test stability.

## Clean-versus-perturbed universality comparison

The bundled `clean_reference.npz` contains the fixed Wolff reference: **all 94
points with L >= 16**, degrees **6 and 3**, **omega = 0.8295**, and **5000 paired
block-bootstrap coefficient sets** (seed 12345). It uses the existing
`fit_scaling.py` vertical-error fit, `U = P(Rxi) + L^(-omega) Q(Rxi)`, with ordinary
powers of Rxi and no clean Rxi cut. Its chi2/dof is **44.21632987/83 = 0.5327268659**.
Source filenames, SHA-256 hashes and the observed Rxi range are stored in the
archive. The source is `fit_analysis_input.tar.gz` supplied on 24 September 2026.

Compare any perturbed batch directly:

```bash
MPLBACKEND=Agg python3 compare_universality.py \
    --perturbed "$ORIG_DIR" "$EXTRA_DIR" --output-prefix sigma05_vs_clean
```

The comparison **never refits or resamples the clean data**. It loads the saved
reference beside the script, checks its model and array dimensions, and resamples
only the perturbed blocks. Central observables and their errors are read directly
from the headers written by `analyze.py`; no jackknife calculation is repeated.
No preparation script or SciPy installation is required.
The same saved clean replica is shared by all perturbed points in each bootstrap,
preserving their residual covariance. Clean block files are no longer needed.

The perturbed selection and plot window are fixed at **0.30 <= Rxi <= 1.00**;
perturbed L=8 points are included. The former `--clean`, `--clean-lmin`,
`--clean-r-min/max`, `--degree-main/correction`, `--omega`, and `--r-min/max`
options have been removed. By default comparison uses all saved replicas;
`--bootstrap 200` uses the first 200 for a quick check, and cannot exceed the
saved count. `--seed` controls the perturbed resampling; `--reference` can select
another saved file with the same model and coefficient format.

Outputs under the comparison prefix:

- `_u_vs_rxi.png/.pdf`: perturbed points and clean curve with pointwise 68% band.
- `_scaled_delta_u.png/.pdf`: `L^omega [U - U_clean_infinity(Rxi)]` with bootstrap errors.
- `_points.txt`: perturbed observables/errors, reference values, differences and
  scaled differences with bootstrap errors.
- `_bootstrap.npz`: coefficient/pair/residual replicas, full residual covariance,
  reference filename, settings and source paths; point order matches `_points.txt`.
  `clean_curve` contains columns `Rxi, U_clean, low_68, high_68` for the plotted band.
  This archive replaces the separate curve and summary text files.

Use adequate block sizes and independent ensembles. Duplicate `(L,beta)` points,
block files identical to a clean source (even if renamed), and central points
beyond the clean domain are rejected.
Errors remain conditional on the clean model: assess its quality before use.
Scaled residuals should approach a common curve, not necessarily zero.
The saved bootstrap sets retain correlations between fit coefficients and give
the reference band without treating coefficient errors as independent. Earlier
Chebyshev/correlated-fit archives are incompatible: use the bundled reference.

## Critical inverse temperature fit

The critical inverse temperature can be estimated from the finite-size scaling of
\(R_\xi(\beta,L)\). The fit uses

```text
x = (beta - beta_c) L^(1/nu)

R_xi(beta, L) =
    sum_k a_k x^k
    + L^(-omega) sum_k b_k x^k
```

with the 3D Ising exponents fixed to

```text
nu = 0.62997097
omega = 0.8295
```

For every trial value of `beta_c`, the polynomial coefficients are found by
weighted linear least squares. A one-dimensional minimization then determines
the `beta_c` that minimizes chi squared. Block bootstrap replicas reconstruct
\(R_\xi\) from the stored `g_zero` and `g_min` block averages and repeat the
complete fit to estimate the statistical uncertainty.

Example:

```bash
MPLBACKEND=Agg python3 fit_beta_c.py metropolis_runs_sigma05 \
    --sizes 8 16 24 32 \
    --degree-main 2 \
    --degree-correction 1 \
    --bootstrap 2000 \
    --output-prefix beta_c_sigma05
```

By default the fit uses a local window, \(0.45 \le R_\xi \le 0.75\), around
the expected crossing region. This is a starting choice, not a universal range:
vary `--r-min` and `--r-max`, as well as the polynomial degrees and size cuts,
to check that the estimate is stable. Point selection uses the central values
and stays fixed in every bootstrap replica. The search
interval for `beta_c` spans all selected beta values, allowing bootstrap
minima beyond the narrower common overlap of the lattice sizes;
it can be overridden with `--beta-min` and `--beta-max`.

At least three distinct lattice sizes must remain after selection. With the
default quadratic main term and linear correction, two sizes cannot determine
`beta_c`: its changes can be absorbed into the polynomial coefficients. Thus
`--sizes 24 32` is not a valid stability check for this script.
The script also rejects flat chi-squared profiles, minima at the search limits,
and search intervals whose endpoints do not both exceed the minimum by
`Delta chi2 > 1`. The last check requires the search range to constrain the
estimate; widen it or add data if necessary. Failed bootstrap fits stop with
the replica number and cause, rather than being silently discarded.

The exponents remain fixed to Ising values: the estimate is conditional on that
scaling model, so it does not independently establish the universality class.

The script writes a text summary and PNG/PDF plots. The summary includes
`beta_c`, its bootstrap uncertainty, the fitted critical value
\(R_\xi^*=a_0\), chi squared per degree of freedom, polynomial degrees, and
the fixed exponents.

## Controlled perturbation study

For Metropolis runs, an optional Gaussian perturbation can be added to the energy difference entering the acceptance probability:

```text
DeltaE -> DeltaE + sigma * G,
G ~ N(0,1)
```

For example:

```text
sigma = 0.5
```

The measured energy and magnetization remain those of the physical spin configuration; the perturbation acts only on the update dynamics.

Setting `sigma = 0` recovers the unperturbed implementation. Wolff runs require `sigma = 0`.

For independent noise at each attempted flip, the code averages the acceptance
over the Gaussian analytically:

```text
P(d) = Phi(-d/sigma) + exp(-beta*d + (beta*sigma)^2/2) * Phi(d/sigma - beta*sigma)
```

Here `Phi` is the standard normal cumulative distribution function. The seven
probabilities for `d = -12,-8,-4,0,4,8,12` are cached (and recomputed if `beta`
or `sigma` changes). A numerically stable form avoids overflow for large noise.
Each noisy flip uses a lookup and a uniform draw, without generating a Gaussian.
This preserves the spin transition probabilities, but changes trajectories for
the same seed when `sigma > 0`. The `sigma = 0` RNG sequence is unchanged.
This averaging does not apply to correlated noise. The cache assumes serial
updates, as used by the batch scripts (independent processes).

This mechanism is used to study the stability of Monte Carlo observables under controlled algorithmic noise.

## Batch workflows

Parameter scans can be launched with the supplied Bash scripts. For example:

```bash
./run_wolff.sh inputs/input_example.dat
```

or

```bash
SIGMA=0.5 ./run_metropolis.sh inputs/input_example.dat
```

The scripts generate independent inputs and seeds and can run multiple simulations concurrently.
Both launch larger lattices first to reduce the time spent waiting for a few
long runs at the end of a batch. Within each size, Wolff launches higher beta
first, where clusters are generally larger. This is a runtime estimate, not a
measured ordering. Launch order does not change the seed assigned to a run.

## Reproducibility and validation

The project includes tests for both simulation and analysis code:

```bash
make test
make test-analysis
```

The tests cover core lattice operations and selected numerical-analysis routines, including the finite-size-scaling design matrix, recovery of known fit coefficients, and reproducibility of bootstrap resampling.

The simulation workflow also supports explicit seeds, restartable runs, deterministic file formats, and automated parameter scans.

## Project status

This is an active MSc thesis project. The code and analysis are still evolving as additional lattice sizes, perturbations, and scaling tests are studied.
