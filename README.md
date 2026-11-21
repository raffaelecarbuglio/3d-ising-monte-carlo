# 3D Ising Monte Carlo

Can numerical errors shift the critical temperature of the 3D Ising model while preserving its universal behaviour? In my MSc thesis at the University of Pisa, I study this question by adding Gaussian noise to the energy difference used in a Metropolis update or by rounding that difference.

I simulate several lattice sizes and plot the Binder ratio against the correlation length divided by the lattice size, $\xi/L$. This lets me compare runs even when their critical temperatures differ. I test whether the perturbed data approach the same curve as unperturbed Wolff simulations as the lattice size grows. The simulations are written in C11 and the analysis in Python.

## Run the example

From the repository root, with GCC, Make and Python 3 installed, run

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
make
./ising inputs/input_example.dat
python3 analyze.py data/example_data.dat --block-size 2
```

The [example input](inputs/input_example.dat) runs a $6^3$ lattice and records eight measurements, enough to try the workflow. Run the C and Python tests with `make test` and `make test-analysis`.

## Run a parameter scan

Edit the lattice sizes, inverse temperatures and seed range in [campaigns/sigma5.conf](campaigns/sigma5.conf), then launch the simulations locally.

```bash
bash simulate.sh campaigns/sigma5.conf
```

The launcher prints the batch directory. After the runs finish, choose block sizes and set `BLOCK_SIZES` in that directory’s saved `campaign.conf`. Then run `bash analyse.sh runs/YOUR_BATCH` with the printed path. The [workflow guide](docs/ising-workflow.md) explains how to choose block sizes, configure perturbations and run independent replicas. On Magellano, use `bash submit_campaign.sh campaigns/sigma5.conf` to submit the campaign as Slurm arrays. See the [cluster instructions](cluster/README.md) for time limits and retries.

## Analysis

- [fit_scaling.py](fit_scaling.py) estimates the large-lattice scaling curve and its finite-size corrections.
- [compare_universality.py](compare_universality.py) compares perturbed data with the Wolff reference curve.
- [fit_beta_c.py](fit_beta_c.py) estimates the critical inverse temperature from the Binder ratio or $\xi/L$.
- [check_beta_c_stability.py](check_beta_c_stability.py) checks how those estimates change with lattice size cuts and fit choices.

A separate [3D Z₂ gauge simulator](z2_gauge/README.md) measures the energy of the unperturbed gauge model.
