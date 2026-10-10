# 3D Ising Monte Carlo

MSc thesis project in Theoretical Physics at the University of Pisa, studying whether controlled numerical perturbations change the universal critical properties of the 3D Ising model.

- **C11:** Metropolis and Wolff simulations on a periodic cubic lattice.
- **Perturbations:** optional independent Gaussian noise in the Metropolis energy difference.
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

The bundled input is a tiny demonstration. For scientific runs, increase thermalization and sampling, and choose a block size using [blocking_plot.py](blocking_plot.py).

Edit [inputs/input_example.dat](inputs/input_example.dat) to set lattice size, temperature, run length, seed, and algorithm. Set `sigma > 0` for noisy Metropolis; Wolff requires `sigma = 0`. One Wolff “sweep” means one cluster update.

## Further analysis

- [fit_scaling.py](fit_scaling.py): universal scaling curve and finite-size corrections.
- [compare_universality.py](compare_universality.py): perturbed data against the saved Wolff reference.
- [fit_beta_c.py](fit_beta_c.py): critical inverse temperature, assuming Ising exponents.

Use `python3 <script> --help` for analysis options. Parameter scans are provided by [run_metropolis.sh](run_metropolis.sh) and [run_wolff.sh](run_wolff.sh).

## INFN cluster

[cluster/README.md](cluster/README.md) explains preparation, Slurm submission, and continuation of independent simulations. Checkpoints preserve the spin configuration, random generator, thermalization progress, and measurement-file position. `save_every` sets their interval (default: 100000 updates). Existing input files remain valid.

*Active thesis project; results and analysis are evolving.*
