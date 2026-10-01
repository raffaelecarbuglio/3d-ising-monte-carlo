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
├── prepare_clean_reference.py        one-time clean fit and bootstrap
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

The Python analysis requires Python 3, NumPy, and Matplotlib. The correlated
clean-reference preparation also uses SciPy:

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

A completed segment writes a checkpoint containing the lattice configuration and the number of production sweeps already completed. A later run can continue from that configuration:

```bash
./ising inputs/input_restart_example.dat
```

On restart, new measurements are appended to the existing data file and the production-sweep counter continues from the previous segment. The random-number generator is initialized from the new seed supplied in the restart input.

This makes long simulations easier to split into independent execution segments while keeping the output continuous and reproducible.

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

## Blocking diagnostics

Monte Carlo measurements are correlated. To study whether the uncertainty estimate has reached a stable blocking regime:

```bash
python3 blocking_plot.py data/example_data.dat
```

The resulting plot shows how estimated errors change as the number of measurements per block is increased.

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

First prepare the clean Wolff reference **once**:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 python3 prepare_clean_reference.py \
    beta_runs_L8-16-24-32 beta_runs_L48-64 beta_runs_L80 \
    --output clean_reference.npz
```

Preparation uses all clean `*_blocks.txt` with **L >= 16**, degrees **6 and 3**,
and **omega = 0.8295**, fitting `U = P(Rxi) + (L/16)^(-omega) Q(Rxi)` with both
coordinate errors and their block-jackknife covariance. It saves the central
coefficients, 5000 clean bootstrap fits, observed domain, fit quality and source
paths. Chebyshev polynomials on the clean range improve conditioning.

Then compare any perturbed batch with that saved reference:

```bash
MPLBACKEND=Agg python3 compare_universality.py \
    --reference clean_reference.npz \
    --perturbed "$ORIG_DIR" "$EXTRA_DIR" --output-prefix sigma05_vs_clean
```

The comparison **never refits or resamples the clean data**. It loads the saved
reference, checks its fixed settings, and resamples only the perturbed blocks.
The same saved clean replica is shared by all perturbed points in each bootstrap,
preserving their residual covariance. Clean block files are no longer needed.

The perturbed selection and plot window are fixed at **0.30 <= Rxi <= 1.00**;
perturbed L=8 points are included. The former `--clean`, `--clean-lmin`,
`--clean-r-min/max`, `--degree-main/correction`, `--omega`, and `--r-min/max`
options have been removed. By default comparison uses all saved replicas;
`--bootstrap 200` uses the first 200 for a quick check, and cannot exceed the
saved count. Both scripts accept `--seed`; preparation also accepts `--bootstrap`.

Outputs under the comparison prefix:

- `_u_vs_rxi.png/.pdf`: perturbed points and clean curve with pointwise 68% band.
- `_scaled_delta_u.png/.pdf`: `L^omega [U - U_clean_infinity(Rxi)]` with bootstrap errors.
- `_points.txt`, `_clean_curve.txt`, `_summary.txt`: numerical results and settings.
- `_bootstrap.npz`: coefficient/pair/residual replicas, full residual covariance,
  reference filename, basis metadata and source paths; point order matches `_points.txt`.

Use adequate block sizes and independent ensembles. Duplicate `(L,beta)` points,
overlapping source paths and central points beyond the clean domain are rejected.
Errors remain conditional on the clean model: assess its quality before use.
Scaled residuals should approach a common curve, not necessarily zero.
Regenerate the reference only if the clean data change. Old comparison archives
lack the required preparation metadata; generate `clean_reference.npz` with the
preparation command above. The reference is a local analysis result, not supplied
with the code; keep it alongside your production data.

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
interval for `beta_c` is the common beta overlap of the selected lattice sizes;
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
