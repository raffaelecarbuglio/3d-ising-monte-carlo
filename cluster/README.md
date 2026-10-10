# INFN runs

Work under `/gpfs/ddn/teorici/carbuglio/3d-ising-monte-carlo`. From Magellano, compile and test on a compute node:

```bash
srun --ntasks=1 --cpus-per-task=1 --pty /bin/bash
make
make test
make test-cluster
exit
```

Prepare a short clean test and submit it from the repository directory:

```bash
REPLICAS=2 N_THERM=100 N_SWEEPS=1000 SAVE_EVERY=100 \
    ./cluster/prepare.sh runs/test_L8 8 0.220 0.222
./cluster/submit.sh runs/test_L8
```

Each beta gets `REPLICAS` independent chains with distinct seeds, up to 32 chains per job. Defaults: `ALGORITHM=metropolis`, `SIGMA=0`, `REPLICAS=2`, `BASE_SEED=200000`, `N_THERM=50000`, `N_SWEEPS=1000000`, `MEASURE_EVERY=10`, `SAVE_EVERY=10000`. Use a different `BASE_SEED` for independent campaigns. Wolff requires `SIGMA=0`; one Wolff update is one cluster flip. Choose production betas and run lengths for your noise strength after timing a short job.

The job uses the professor's `default` account, `zefiro` partition, 1 GiB per core, and 23:59 hours. Edit `cluster/job.batch` before preparation if needed. It runs the repository's executable: keep the same compiled code for all segments. The preparation commit is recorded in `revision.txt`.

Resubmit **after the previous job has ended** using the same `submit.sh` command. Existing checkpoints load automatically, preserving the random sequence and remaining thermalization. Each invocation requests `N_SWEEPS` additional production updates per chain. Slurm requests a checkpoint two minutes before the time limit; increase this margin if one update and saving take longer. `SIGTERM` and `SIGINT` also save before stopping. Measurements written after the last checkpoint are removed on restart.

New checkpoints require unchanged `L`, `beta`, `sigma`, `algorithm`, and `measure_every`. Old spin-only configurations use the input seed and thermalization count; change the seed on each restart. A missing data file starts a new measurement segment.

Use `squeue -u "$USER"` to monitor jobs and `scancel JOB_ID` to stop one. Per-chain logs are in `logs/`; measurements and checkpoints are in `results/`. Analyze independent chains separately for blocking and autocorrelation. `make test-cluster` tests restarts and uses local mock Slurm commands.
