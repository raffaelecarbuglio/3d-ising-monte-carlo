# 3D Ising Monte Carlo

MSc thesis project in Theoretical Physics at the University of Pisa, studying whether controlled numerical perturbations change the universal critical properties of the 3D Ising model.

- **C11:** Metropolis and Wolff simulations on a periodic cubic lattice.
- **Perturbations:** independent Gaussian noise or deterministic rounding of the Metropolis energy difference.
- **Python:** observables, blocking, jackknife errors, and finite-size-scaling fits with block bootstrap.

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

Tests are kept in [`tests/`](tests/); run them with `make test` and `make test-analysis`.

The bundled input is a tiny demonstration. For scientific runs, increase thermalization and sampling, and choose a block size using [blocking_plot.py](blocking_plot.py).

Edit [inputs/input_example.dat](inputs/input_example.dat) to set lattice size, temperature, run length, seed, and algorithm. Optional final fields `sigma` and `lambda` default to zero. Set `sigma > 0` for Gaussian noise, or `lambda > 0` for `round(lambda * delta_E) / lambda`; enable only one. Wolff requires both to be zero. One Wolff “sweep” means one cluster update.

For rounding, try `sigma = 0` and `lambda = 0.55`. The four families at strength `1/m = 1/2` use lambda `0.55, 0.60, 0.40, 0.45`. Keep lambda fixed while scanning beta; locate the shifted transition before production runs. Outputs record lambda and the rounded costs and ratios. A restart into the same data file requires the same sigma and lambda. `run_metropolis.sh` and `cluster/prepare.sh` accept `LAMBDA=...` (default zero), alongside `SIGMA=...`.

## Further analysis

- [fit_scaling.py](fit_scaling.py): universal scaling curve and finite-size corrections.
- [compare_universality.py](compare_universality.py): perturbed data against the saved Wolff reference.
- [fit_beta_c.py](fit_beta_c.py): critical inverse temperature, assuming Ising exponents.

Use `python3 <script> --help` for analysis options. Parameter scans are provided by [run_metropolis.sh](run_metropolis.sh) and [run_wolff.sh](run_wolff.sh).

For INFN submission scripts, see [cluster/README.md](cluster/README.md).

*Active thesis project; results and analysis are evolving.*
