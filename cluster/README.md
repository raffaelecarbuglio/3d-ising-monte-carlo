# INFN runs

Work under `/gpfs/ddn/teorici/carbuglio/3d-ising-monte-carlo`. From Magellano, compile on a compute node:

```bash
srun --ntasks=1 --cpus-per-task=1 --pty /bin/bash
make
make test
exit
```

From the repository directory, prepare and submit a short test:

```bash
REPLICAS=2 N_THERM=100 N_SWEEPS=1000 \
    ./cluster/prepare.sh runs/test_L8 8 0.220 0.222
./cluster/submit.sh runs/test_L8
```

Each beta gets `REPLICAS` simulations with distinct seeds, up to 32 per job. Defaults: `REPLICAS=2`, `BASE_SEED=200000`, `N_THERM=50000`, `N_SWEEPS=1000000`, `MEASURE_EVERY=10`, `ALGORITHM=metropolis`, `SIGMA=0`. Use a different seed range for each campaign. Wolff requires `SIGMA=0`. The generated inputs are in `inputs/`, outputs in `results/`, and logs in `logs/` inside the run directory.

Check seeds across all batches under `runs/` with `python3 cluster/check_seeds.py runs`. You can also pass several batch directories. The checker reads `.in` files, reports repeated seeds and their file paths, and exits with an error if duplicates are found. It only checks the inputs currently present; keep copies of old inputs if you want to include past runs.

The job requests the professor's `default` account, `zefiro` partition, one core per simulation, 1 GiB per core and 23:59 hours. Edit `cluster/job.batch` before preparation if needed. Use `squeue -u "$USER"` to monitor jobs.

To continue a **completed** job, edit its inputs: set `start = restart`, `n_therm = 0`, and a new distinct seed for each simulation. Then submit the same directory again after the previous job has ended. Each invocation adds `n_sweeps` production updates.

Choose run lengths that finish within the time limit, using a short timing test. Interrupted jobs retain the original program's restart limitations: it saves spins every 100000 production updates, does not save the random generator or thermalization progress, and does not remove measurements beyond the saved configuration. Do not resubmit an interrupted run without checking its files.
