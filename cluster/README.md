# INFN runs

Work under `/gpfs/ddn/teorici/carbuglio/3d-ising-monte-carlo`. From Magellano, compile on a compute node:

```bash
srun --ntasks=1 --cpus-per-task=1 --pty /bin/bash
make
make test
exit
```

## Submit a campaign

From the project directory on Magellano, edit your campaign `.conf` and submit it.

```bash
bash submit_campaign.sh campaigns/sigma5.conf
```

Use a current `ising` executable built with `make ising` on a compute node.
The launcher copies the executable and configuration into one campaign folder.
It submits one array per lattice size, with one CPU and 1 GiB of memory per replica.
Each replica writes its own files to the same `results/` directory.

Optional settings in the campaign config:

- `SLURM_PARTITIONS=levante,mefisto,zefiro` sets the eligible partitions (the default).
- `SLURM_TIME=1-00:00:00` sets the default wall-time limit per replica.
  `SLURM_TIME_BY_L="48:12:00:00 64:18:00:00"` overrides it for individual sizes.
- `SLURM_MAX_RUNNING=32` limits each size array to 32 running tasks. Every task
  still counts toward Slurm's submission limits. `MAX_JOBS` controls the local
  launcher.

The launcher prints the campaign path and the job IDs. Use `squeue -u "$USER"`
to check progress. Once the arrays finish, analyse the folder as usual.

```bash
bash analyse.sh runs/YOUR_BATCH
```

If a submission fails or a replica exits early, wait for the campaign's queued
tasks to finish and run `bash submit_campaign.sh --retry runs/YOUR_BATCH`.
This resubmits incomplete replicas from their original inputs. Earlier partial
files are saved under `slurm/attempts/`. The analysis requires a successful exit
and a measurement file for every replica.

## Submit a small fixed-node job

From the repository directory, prepare and submit a short test:

```bash
REPLICAS=2 N_THERM=100 N_SWEEPS=1000 \
    ./cluster/prepare.sh runs/test_L8 8 0.220 0.222
./cluster/submit.sh runs/test_L8
```

Each beta gets `REPLICAS` simulations with distinct seeds, up to 32 per job. Defaults: `REPLICAS=2`, `BASE_SEED=200000`, `N_THERM=50000`, `N_SWEEPS=1000000`, `MEASURE_EVERY=10`, `ALGORITHM=metropolis`, `SIGMA=0`, `LAMBDA=0`. Use a different seed range for each campaign. For energy rounding use `LAMBDA=0.55 SIGMA=0`; choose the beta grid around its shifted transition. Gaussian noise and rounding cannot both be enabled. Wolff requires `SIGMA=0 LAMBDA=0`. The generated inputs are in `inputs/`, outputs in `results/`, and logs in `logs/` inside the run directory.

Check seeds across all batches under `runs/` with `python3 cluster/check_seeds.py runs`. You can also pass several batch directories. The checker reads `.in` files, reports repeated seeds and their file paths, and exits with an error if duplicates are found. It only checks the inputs currently present; keep copies of old inputs if you want to include past runs.

The job requests the professor's `default` account, `zefiro` partition, one core per simulation, 1 GiB per core and 23:59 hours. Edit `cluster/job.batch` before preparation if needed. Use `squeue -u "$USER"` to monitor jobs.

To continue a **completed** job, edit its inputs: set `start = restart`, `n_therm = 0`, and a new distinct seed for each simulation. Then submit the same directory again after the previous job has ended. Each invocation adds `n_sweeps` production updates.

Choose run lengths that finish within the time limit, using a short timing test. Interrupted jobs retain the original program's restart limitations: it saves spins every 100000 production updates, does not save the random generator or thermalization progress, and does not remove measurements beyond the saved configuration. Do not resubmit an interrupted run without checking its files.
