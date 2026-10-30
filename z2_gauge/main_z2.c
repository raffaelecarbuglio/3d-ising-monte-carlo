#include "z2.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int L;
    double beta;
    int n_therm;
    int n_sweeps;
    int measure_every;
    int seed;
    char start[16];
    char data_file[512];
} Parameters;

/* Keep the same simple, fixed-order input format as the Ising example. */
static int read_input(const char *filename, Parameters *p)
{
    FILE *file = fopen(filename, "r");
    char extra;

    if (file == NULL) {
        fprintf(stderr, "Cannot open input file '%s'.\n", filename);
        return 0;
    }

    int fields = fscanf(file,
                        " L = %d beta = %lf n_therm = %d n_sweeps = %d"
                        " measure_every = %d seed = %d start = %15s"
                        " data_file = %511s",
                        &p->L, &p->beta, &p->n_therm, &p->n_sweeps,
                        &p->measure_every, &p->seed, p->start, p->data_file);
    int trailing = fscanf(file, " %c", &extra);
    int read_error = ferror(file);
    int close_error = fclose(file);

    if (fields != 8 || trailing != EOF || read_error || close_error != 0) {
        fprintf(stderr, "Invalid input format; use z2_gauge/input_example.dat as a model.\n");
        return 0;
    }
    if (p->L < 2 || p->L > Z2_MAX_L) {
        fprintf(stderr, "L must be between 2 and %d.\n", Z2_MAX_L);
        return 0;
    }
    if (!isfinite(p->beta) || p->beta < 0.0) {
        fprintf(stderr, "beta must be finite and nonnegative.\n");
        return 0;
    }
    if (p->n_therm < 0 || p->n_sweeps <= 0 || p->measure_every <= 0
        || p->measure_every > p->n_sweeps || p->seed < 0) {
        fprintf(stderr,
                "Require n_therm >= 0, n_sweeps > 0, seed >= 0, and\n"
                "1 <= measure_every <= n_sweeps.\n");
        return 0;
    }
    if (strcmp(p->start, "ordered") != 0 && strcmp(p->start, "random") != 0) {
        fprintf(stderr, "start must be ordered or random.\n");
        return 0;
    }
    return 1;
}

static int run_simulation(const Parameters *p)
{
    GaugeLattice g;
    Pcg32 rng;
    long long proposed = 0;
    long long accepted = 0;
    int ok = 1;

    if (!z2_create(&g, p->L))
        return 0;

    pcg32_seed(&rng, p->seed);
    if (strcmp(p->start, "ordered") == 0)
        z2_fill_ordered(&g);
    else
        z2_fill_random(&g, &rng);

    /* C11 exclusive creation protects previous measurements. */
    FILE *data = fopen(p->data_file, "wx");
    if (data == NULL) {
        fprintf(stderr,
                "Cannot create '%s'; choose a new name if it already exists.\n",
                p->data_file);
        z2_destroy(&g);
        return 0;
    }

    if (fprintf(data,
                "# Pure Z2 gauge model, 3D, periodic boundaries, coupling J=1\n"
                "# E = -sum_p P_p; probability proportional to exp(-beta*E)\n"
                "# L = %d\n# beta = %.17g\n# seed = %d\n# start = %s\n"
                "# n_therm = %d\n# n_sweeps = %d\n# measure_every = %d\n"
                "# sweep: one pass over 3*L^3 links; 50%% chance to propose a flip\n"
                "# acceptance: accepted flips / proposed flips since last measure\n"
                "# columns: sweep total_energy acceptance\n",
                p->L, p->beta, p->seed, p->start,
                p->n_therm, p->n_sweeps, p->measure_every) < 0
        || fflush(data) != 0) {
        fprintf(stderr, "Cannot write the data header.\n");
        fclose(data);
        z2_destroy(&g);
        return 0;
    }

    printf("L=%d, beta=%.8g, start=%s, seed=%d\n",
           p->L, p->beta, p->start, p->seed);
    printf("Thermalisation: %d sweeps.\n", p->n_therm);
    fflush(stdout);
    for (int sweep = 0; sweep < p->n_therm; sweep++)
        z2_metropolis_sweep(&g, p->beta, &rng);

    printf("Production: %d sweeps, measurement every %d.\n",
           p->n_sweeps, p->measure_every);
    fflush(stdout);
    for (int sweep = 0; sweep < p->n_sweeps; sweep++) {
        SweepResult result = z2_metropolis_sweep(&g, p->beta, &rng);
        proposed += result.proposed;
        accepted += result.accepted;

        if ((sweep + 1) % p->measure_every == 0) {
            int energy = z2_total_energy(&g);
            double acceptance = proposed > 0 ? (double)accepted / proposed : 0.0;

            if (fprintf(data, "%d %d %.17g\n", sweep + 1, energy, acceptance) < 0
                || fflush(data) != 0) {
                fprintf(stderr, "Error writing measurements.\n");
                ok = 0;
                break;
            }
            proposed = 0;
            accepted = 0;
        }
    }

    if (fclose(data) != 0) {
        fprintf(stderr, "Error closing the data file.\n");
        ok = 0;
    }
    z2_destroy(&g);

    if (ok)
        printf("Done. Measurements: %s\n", p->data_file);
    return ok;
}

int main(int argc, char **argv)
{
    Parameters p;

    if (argc != 2) {
        fprintf(stderr, "Usage: %s input.dat\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!read_input(argv[1], &p) || !run_simulation(&p))
        return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
