#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "checkpoint.h"

static volatile sig_atomic_t stop_requested;

static void request_stop(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static FILE *open_data_file(const SimulationParameters *p,
                            const Checkpoint *state, int complete)
{
    const char *columns = "# columns: sweep energy magnetization g_min\n";
    FILE *file;

    if (p->start == START_RESTART) {
        char line[512], old_algorithm[16] = "";
        int same_columns = 0, old_L = -1, old_measure_every = -1;
        double old_beta = -1.0, old_sigma = 0.0;
        long size, offset;

        file = fopen(p->data_file, "r+");
        if (file == NULL && errno == ENOENT) file = fopen(p->data_file, "wx");
        if (file == NULL) goto error;
        if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0) goto invalid;
        rewind(file);
        if (size > 0) {
            while (fgets(line, sizeof(line), file) != NULL) {
                sscanf(line, "# L = %d", &old_L);
                sscanf(line, "# beta = %lf", &old_beta);
                sscanf(line, "# sigma = %lf", &old_sigma);
                sscanf(line, "# algorithm = %15s", old_algorithm);
                sscanf(line, "# measure_every = %d", &old_measure_every);
                if (strcmp(line, columns) == 0) {
                    same_columns = 1;
                    break;
                }
                if (line[0] != '#') break;
            }
            if (ferror(file) || !same_columns || old_L != p->L ||
                old_beta != p->beta || old_sigma != p->sigma ||
                old_measure_every != p->measure_every ||
                strcmp(old_algorithm, algorithm_name(p->algorithm)) != 0) goto invalid;

            if (complete) {
                offset = state->data_offset;
                if (strcmp(state->data_file, p->data_file) != 0 ||
                    offset < ftell(file) || offset > size) goto invalid;
            } else {
                /* Vecchio salvataggio: elimina le misure successive agli spin salvati. */
                offset = ftell(file);
                while (fgets(line, sizeof(line), file) != NULL) {
                    int sweep;
                    if (line[0] != '#') {
                        if (sscanf(line, "%d", &sweep) != 1 ||
                            sweep > state->production_sweeps) break;
                    }
                    offset = ftell(file);
                }
                if (ferror(file)) goto invalid;
            }
            /* Il checkpoint viene scritto dopo il flush: tutto il seguito e' da rifare. */
            if (ftruncate(fileno(file), offset) != 0 ||
                fseek(file, offset, SEEK_SET) != 0) goto invalid;
            if (fprintf(file,
                        "# restart: initial_sweep=%d seed=%d rng_restored=%d\n",
                        state->production_sweeps, p->seed, complete) < 0) goto invalid;
            return file;
        }
    } else {
        file = fopen(p->data_file, "wx");
        if (file == NULL) goto error;
    }
    if (fprintf(file,
                "# Ising 3D\n# L = %d\n# beta = %.17g\n# seed = %d\n"
                "# start = %s\n# algorithm = %s\n# sigma = %.17g\n"
                "# n_therm = %d\n# n_sweeps = %d\n# measure_every = %d\n"
                "# save_every = %d\n# config_file = %s\n%s",
                p->L, p->beta, p->seed, start_mode_name(p->start),
                algorithm_name(p->algorithm), p->sigma, p->n_therm,
                p->n_sweeps, p->measure_every, p->save_every,
                p->config_file, columns) < 0) goto invalid;
    return file;
invalid:
    fclose(file);
error:
    fprintf(stderr, "Errore: dati non accessibili o incompatibili in '%s'.\n", p->data_file);
    return NULL;
}

static int save_checkpoint(FILE *data, const SimulationParameters *p,
                           const IsingLattice *lattice, Checkpoint *state)
{
    if (fflush(data) != 0 || (state->data_offset = ftell(data)) < 0 ||
        !checkpoint_save(p, lattice, state)) {
        fprintf(stderr, "Errore durante il salvataggio del checkpoint.\n");
        return 0;
    }
    return 1;
}

static int write_measurement(FILE *file, const IsingLattice *lattice,
                             int sweep,
                             const double *cos_table, const double *sin_table)
{
    int L = lattice->L;
    int energy = 0;
    double n = (double)lattice->n_spins;
    int magnetization = 0;
    int plane_x[ISING_MAX_L] = {0};
    int plane_y[ISING_MAX_L] = {0};
    int plane_z[ISING_MAX_L] = {0};
    double real_x = 0.0;
    double imaginary_x = 0.0;
    double real_y = 0.0;
    double imaginary_y = 0.0;
    double real_z = 0.0;
    double imaginary_z = 0.0;
    double g_min;
    int x;
    int y;
    int z;
    int coordinate;

    /* Un solo passaggio raccoglie energia, magnetizzazione e somme sui piani. */
    for (z = 0; z < L; z++) {
        int next_z = (z + 1 == L) ? 0 : z + 1;

        for (y = 0; y < L; y++) {
            int next_y = (y + 1 == L) ? 0 : y + 1;

            for (x = 0; x < L; x++) {
                int next_x = (x + 1 == L) ? 0 : x + 1;
                int index = (z * L + y) * L + x;
                int spin = lattice->spins[index];

                /* I tre vicini in avanti contano ogni legame una sola volta. */
                energy -= spin * (lattice->spins[(z * L + y) * L + next_x]
                                + lattice->spins[(z * L + next_y) * L + x]
                                + lattice->spins[(next_z * L + y) * L + x]);
                magnetization += spin;
                plane_x[x] += spin;
                plane_y[y] += spin;
                plane_z[z] += spin;
            }
        }
    }

    /* A momento (2*pi/L,0,0), la fase dipende solo da x: sommiamo prima
       gli spin a x fissato. Lo stesso vale per le direzioni y e z. */
    for (coordinate = 0; coordinate < L; coordinate++) {
        real_x += plane_x[coordinate] * cos_table[coordinate];
        imaginary_x += plane_x[coordinate] * sin_table[coordinate];
        real_y += plane_y[coordinate] * cos_table[coordinate];
        imaginary_y += plane_y[coordinate] * sin_table[coordinate];
        real_z += plane_z[coordinate] * cos_table[coordinate];
        imaginary_z += plane_z[coordinate] * sin_table[coordinate];
    }

    g_min = (real_x * real_x + imaginary_x * imaginary_x
           + real_y * real_y + imaginary_y * imaginary_y
           + real_z * real_z + imaginary_z * imaginary_z) / (3.0 * n);

    if (fprintf(file, "%d %d %d %.8g\n",
                sweep, energy, magnetization, g_min) < 0) {
        return 0;
    }
    return 1;
}

static int run_simulation(const SimulationParameters *p)
{
    IsingLattice lattice;
    Checkpoint state = {0};
    FILE *data;
    int *cluster = NULL, *in_cluster = NULL;
    double wolff_probability = 0.0;
    double cos_table[ISING_MAX_L], sin_table[ISING_MAX_L];
    double minimum_momentum;
    int complete = 0, initial_sweep, target_sweep, updates = 0, ok = 1;

    if (!ising_create(&lattice, p->L)) return 0;
    pcg32_seed(&state.rng, p->seed);
    state.thermal_remaining = p->n_therm;
    if (p->start == START_ORDERED) {
        ising_fill_ordered(&lattice);
    } else if (p->start == START_RANDOM) {
        ising_fill_random(&lattice, &state.rng);
    } else if (!checkpoint_load(p, &lattice, &state, &complete)) {
        ising_destroy(&lattice);
        return 0;
    }
    initial_sweep = state.production_sweeps;
    if (p->n_sweeps > INT_MAX - initial_sweep) {
        fprintf(stderr, "Errore: troppi sweep per il contatore.\n");
        ising_destroy(&lattice);
        return 0;
    }
    target_sweep = initial_sweep + p->n_sweeps;

    minimum_momentum = 2.0 * acos(-1.0) / lattice.L;
    for (int i = 0; i < lattice.L; i++) {
        double angle = minimum_momentum * i;
        cos_table[i] = cos(angle);
        sin_table[i] = sin(angle);
    }
    if (p->algorithm == ALGORITHM_WOLFF) {
        cluster = malloc(lattice.n_spins * sizeof(*cluster));
        in_cluster = calloc(lattice.n_spins, sizeof(*in_cluster));
        if (cluster == NULL || in_cluster == NULL) {
            fprintf(stderr, "Errore: memoria insufficiente per Wolff.\n");
            free(cluster);
            free(in_cluster);
            ising_destroy(&lattice);
            return 0;
        }
        wolff_probability = 1.0 - exp(-2.0 * p->beta);
    }
    data = open_data_file(p, &state, complete);
    if (data == NULL) {
        free(cluster);
        free(in_cluster);
        ising_destroy(&lattice);
        return 0;
    }
    printf("L=%d beta=%.17g algorithm=%s sigma=%.17g seed=%d rng_restored=%d\n",
           p->L, p->beta, algorithm_name(p->algorithm), p->sigma, p->seed, complete);
    fflush(stdout);

    /* Salva anche prima della termalizzazione: ogni job puo' essere ripreso. */
    ok = save_checkpoint(data, p, &lattice, &state);
    while (ok && !stop_requested &&
           (state.thermal_remaining > 0 || state.production_sweeps < target_sweep)) {
        if (p->algorithm == ALGORITHM_METROPOLIS) {
            ising_metropolis_sweep(&lattice, p->beta, p->sigma, &state.rng);
        } else {
            ising_wolff_update(&lattice, wolff_probability, &state.rng, cluster, in_cluster);
        }
        if (state.thermal_remaining > 0) {
            state.thermal_remaining--;
        } else {
            state.production_sweeps++;
            /* La cadenza delle misure non riparte da zero a ogni segmento. */
            if (state.production_sweeps % p->measure_every == 0) {
                ok = write_measurement(data, &lattice, state.production_sweeps,
                                       cos_table, sin_table);
            }
        }
        updates++;
        if (ok && updates == p->save_every) {
            ok = save_checkpoint(data, p, &lattice, &state);
            updates = 0;
        }
    }
    if (ok) ok = save_checkpoint(data, p, &lattice, &state);
    if (fclose(data) != 0) ok = 0;
    if (ok) {
        printf("%s: %d sweep prodotti; totale=%d; termalizzazione restante=%d.\n",
               stop_requested ? "Interrotto e salvato" : "Completato",
               state.production_sweeps - initial_sweep,
               state.production_sweeps, state.thermal_remaining);
    }
    free(cluster);
    free(in_cluster);
    ising_destroy(&lattice);
    return ok;
}

int main(int argc, char **argv)
{
    SimulationParameters parameters;
    struct sigaction action = {0};

    if (argc != 2) {
        fprintf(stderr, "Uso: %s input.dat\n", argv[0]);
        return EXIT_FAILURE;
    }
    action.sa_handler = request_stop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, NULL) != 0 ||
        sigaction(SIGINT, &action, NULL) != 0 ||
        sigaction(SIGUSR1, &action, NULL) != 0) return EXIT_FAILURE;
    if (!input_read(argv[1], &parameters) || !run_simulation(&parameters)) return EXIT_FAILURE;
    return EXIT_SUCCESS;
}
