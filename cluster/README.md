# INFN runs

Use `/gpfs/ddn/teorici/carbuglio` for the repository and results. Prepare and submit jobs from Magellano; compile, test, and run simulations on compute nodes.

```bash
cd /gpfs/ddn/teorici/carbuglio/3d-ising-monte-carlo
git pull
srun --ntasks=1 --cpus-per-task=1 --pty /bin/bash
make clean
make
make test
make test-cluster
exit
```

If using the compiler module supplied by the professor, load `intel-oneapi-compilers` and use `make CC=icx CFLAGS='-std=c11 -O3 -Wall -Wextra'` instead. The batch jobs use the already compiled executable.

## Prepare and submit

One campaign contains up to 32 serial simulations on one node. Give the lattice size and an explicit list of inverse temperatures. `REPLICAS=2` means two independent chains per beta; 16 betas therefore use 32 cores.

This is a **short clean test**, not a production grid for noisy Ising:

```bash
REPLICAS=2 N_THERM=100 N_SWEEPS=1000 MEASURE_EVERY=10 SAVE_EVERY=100 \
    ./cluster/prepare.sh runs/test_L8 8 0.220 0.222
./cluster/submit.sh runs/test_L8
```

For production, choose betas appropriate to the noise strength and set:

| Variable | Default | Meaning |
|---|---|---|
| `ALGORITHM` | `metropolis` | `metropolis` or `wolff` |
| `SIGMA` | `0` | Noise standard deviation; zero for Wolff |
| `REPLICAS` | `2` | Independent chains per beta |
| `N_THERM` | `50000` | Thermalization updates for each new chain |
| `N_SWEEPS` | `1000000` | Additional production updates per invocation |
| `MEASURE_EVERY` | `10` | Measurement interval |
| `SAVE_EVERY` | `10000` | Checkpoint interval, including thermalization |
| `START` | `random` | `random` or `ordered` |
| `BASE_SEED` | Random | First seed; subsequent chains use distinct successive seeds |

Time a short job on the cluster before choosing production lengths. One Wolff update counts as one cluster flip.

The campaign keeps a copy of the executable, its SHA-256 checksum, the Git commit, any tracked local changes in `code.patch`, and the generated inputs. Compile from a clean checkout for scientific campaigns; changing the repository later does not change the copied executable or inputs.

The supplied Slurm settings are the professor's `default` account, `zefiro` partition, 1 GiB per core, and 23:59 hours. Submission requests one core for each generated simulation. Edit `cluster/job.batch` before preparing a campaign if the cluster settings change.

## Continue

```bash
./cluster/submit.sh runs/test_L8
```

The job loads each existing checkpoint automatically. Each invocation requests `N_SWEEPS` **additional** production updates; if a job stops early, the actual number can be smaller. The per-chain logs report the number completed. Resubmitting also continues chains that already finished their previous segment.

To queue three successive jobs, each starting after successful completion of the preceding one:

```bash
./cluster/submit.sh runs/test_L8 3
```

`SIGUSR1`, `SIGTERM`, and `SIGINT` request a stop after the current update, followed by a checkpoint. Slurm sends `SIGUSR1` two minutes before the time limit. A saved, orderly stop exits successfully; an I/O or simulation error fails the job and blocks its dependants. Simultaneous jobs for the same campaign are rejected by a file lock.

The checkpoint restores the random sequence and unfinished thermalization. After an abrupt interruption, measurements beyond the checkpoint's saved file position are removed before continuation. Measurement spacing uses the total sweep count. Two minutes must be enough to finish one update and save; increase the warning interval for unusually large lattices.

New checkpoints require unchanged `L`, `beta`, `algorithm`, `sigma`, and `measure_every`. The restart seed is ignored because the random generator is restored. Old spin-only configurations remain readable and use the input seed and thermalization count; use a fresh seed when restarting those. A missing or empty data file starts a new measurement segment; it cannot recover missing earlier measurements.

Keep independent chains separate during blocking and autocorrelation analysis. Runs at the same beta can be combined statistically after thermalization; their junction is not a physical time step. Existing analysis tools can read the four-column measurement files.

```bash
squeue -u "$USER"
scancel JOB_ID
make test-cluster
```

The local cluster tests use mock Slurm commands; they do not submit real jobs.
