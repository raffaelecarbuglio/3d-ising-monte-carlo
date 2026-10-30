# Minimal 3D Z2 gauge simulation

This is a separate C11 implementation of the pure Z2 gauge model on a cubic
periodic lattice, with coupling J = 1. It shares the repository's PCG32
implementation in `../rng.c` and `../rng.h` with the Ising simulator.

## Build and run

```bash
make z2
make test-z2
./z2 z2_gauge/input_example.dat
```

Run these commands from the repository root. After the first build, edit
`z2_gauge/input_example.dat` and run the last command.
The example is a short demonstration, not a production run or an assessment
of adequate thermalisation. An existing data file is never overwritten:
choose a different `data_file` for each new run.

## Files and suggested reading order

1. `z2.h`: the lattice and the public functions.
2. `z2.c`: neighbours, plaquettes, energy, and the Metropolis sweep.
3. `main_z2.c`: input, thermalisation, and measurement output.
4. `../tests/test_z2.c`: small checks of the physics and periodic boundaries.

The root `Makefile` builds both simulators with `make` and runs both C test
suites with `make test`. `make test-z2` runs only the gauge checks.
`make clean` removes build files; energy measurements survive.

## Lattice and energy

There are L^3 sites, 3 L^3 links, and 3 L^3 plaquettes.
The site index is `(z * L + y) * L + x`. The three links leaving that site
in the positive x, y, z directions are `links[3 * site + mu]`, for
`mu = 0, 1, 2`. Each physical link is stored once. Its value is +1 or -1,
and traversing it backwards uses the same value.

`z2_plaquette(g, site, mu, nu)` multiplies the four links around the square
starting at `site` in the positive `mu` and `nu` directions:

```text
P_mu,nu(site) = u_mu(site) * u_nu(site + mu)
             * u_mu(site + nu) * u_nu(site).
```

The total energy is `E = -sum_p P_p`; the equilibrium weight is
`exp(-beta * E)`. Each plaquette is counted once. The energy change for
flipping one link is twice the sum of the four plaquettes containing it.
Those plaquettes start at `site` and `site - nu` for each `nu != mu`.

## Meaning of a sweep

One sweep visits all 3 L^3 links in a fixed order. At each visit, a fair coin
decides whether to propose a flip. For a proposed flip, the usual Metropolis
rule applies: accept if dE <= 0; otherwise accept with probability
`exp(-beta * dE)`.

The coin is the only change from the initial sketch. Without it, at beta=0
every link would flip in each sweep, and every plaquette would return to its
old value. The 50% skip preserves the equilibrium distribution: it reduces
both the forward and reverse transition probabilities by the same factor.
It changes the Monte Carlo dynamics, so this sweep is a different time unit
from a sweep that proposes all 3 L^3 flips. On average it makes 1.5 L^3
flip proposals. At beta=0, one pass generates independent random links.

The output acceptance is the number of accepted flips divided by the number
of proposed flips since the previous measurement; skipped visits are
excluded. If that interval contains no proposals, acceptance is recorded as 0.

## Input

Keep these fields in this order, without comments or extra fields:

```text
L = 8
beta = 0.75
n_therm = 1000
n_sweeps = 5000
measure_every = 10
seed = 12345
start = ordered
data_file = data/z2_energy.dat
```

- `L`: integer from 2 to 512. Memory grows as 3 L^3 integers.
- `beta`: finite, nonnegative inverse temperature.
- `n_therm`: nonnegative number of sweeps to discard before production.
- `n_sweeps`: positive number of production sweeps.
- `measure_every`: integer from 1 to `n_sweeps`. If it does not divide
  `n_sweeps`, the final incomplete measurement interval produces no row.
- `seed`: nonnegative integer used by the existing PCG32 implementation.
- `start`: `ordered` (all links +1) or `random` (independent random signs).
- `data_file`: a new output filename, without spaces. Its parent directory
  must already exist. Paths are relative to the directory you run from.

The sweep counts and seed must fit in a C `int`. The lattice-size bound keeps
link indices and total energies within the same integer type.

## Measurements

The header records the parameters and conventions. Numeric columns are:

```text
sweep    total_energy    acceptance
```

Sweep numbering starts at 1 after thermalisation. The energy is an exact
integer: it is not divided by the volume. The energy per plaquette is
`total_energy / (3 * L**3)`, and the mean plaquette in a configuration is its
negative. Saving total energy lets you calculate its moments and cumulants
later, with the normalisation you choose. Python can load the columns with:

```python
import numpy as np

sweep, energy, acceptance = np.loadtxt("data/z2_energy.dat", unpack=True)
L = 8  # Match the input.
energy_per_plaquette = energy / (3 * L**3)
```

Measurements are flushed to disk after every row. As with your Ising runs,
thermalisation and statistical errors require checking autocorrelations and
blocking before production results can be used.

This first version implements the unperturbed model, ordered/random starts,
and energy measurements. It does not yet implement restart files,
perturbations, Wilson loops, or statistical analysis.

## Checks

`make test-z2` checks:

- All links +1 give E = -3 L^3; one isolated flip raises E by 8.
- Local dE equals the difference of full energies for every link of random
  lattices with L = 2, 3, 4, including links across periodic boundaries.
- Flipping the six links touching any site leaves every plaquette unchanged.
- At beta=0 the measured energy moments agree with the independent-link
  expectations, rather than remaining frozen after each sweep.

The implementation was also checked against enumeration of all 2^24 link
assignments on the L=2 lattice. At beta = 0.25, 0.50, and 0.75, simulated
first and second energy moments agreed with the exact results within 1.3
estimated standard errors (using blocks of 500 sweeps). Address and undefined
behaviour checks passed; leak detection could not run in this environment.
