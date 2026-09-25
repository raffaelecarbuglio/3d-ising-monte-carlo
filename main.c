#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "input.h"
#include "ising.h"
#include "rng.h"

static FILE *open_data_file(const SimulationParameters *p, int initial_sweep)
{
    FILE *file;
    const char *columns = "# columns: sweep energy magnetization g_min\n";

    if (p->start == START_RESTART) {
        /* Un restart aggiunge un segmento breve al file esistente o ne crea uno. */
        char line[512];
        int has_contents = 0;
        int same_columns = 0;
        double old_sigma = 0.0; /* I file precedenti erano senza rumore. */

        file = fopen(p->data_file, "a+");
        if (file == NULL) {
            fprintf(stderr, "Errore: impossibile aggiungere dati a '%s'.\n", p->data_file);
            return NULL;
        }
        rewind(file);
        while (fgets(line, sizeof(line), file) != NULL) {
            has_contents = 1;
            sscanf(line, "# sigma = %lf", &old_sigma);
            if (strcmp(line, columns) == 0) {
                same_columns = 1;
                break;
            }
            if (line[0] != '#') {
                break;
            }
        }
        if (ferror(file) || (has_contents && (!same_columns || old_sigma != p->sigma))) {
            fprintf(stderr,
                    "Errore: formato dati o sigma incompatibile in '%s'; "
                    "usare un nuovo data_file per il restart.\n", p->data_file);
            fclose(file);
            return NULL;
        }
        if (fseek(file, 0, SEEK_END) != 0) {
            fclose(file);
            return NULL;
        }
        if (has_contents) {
            if (fprintf(file,
                        "#\n"
                        "# restart: initial_sweep=%d beta=%.17g "
                        "seed=%d n_therm=%d "
                        "n_sweeps=%d measure_every=%d algorithm=%s sigma=%.17g\n"
                        "# config_file=%s\n",
                        initial_sweep, p->beta, p->seed, p->n_therm,
                        p->n_sweeps, p->measure_every,
                        algorithm_name(p->algorithm), p->sigma, p->config_file) < 0) {
                fclose(file);
                return NULL;
            }
            return file;
        }
    } else {
        file = fopen(p->data_file, "wx");
    }
    if (file == NULL) {
        fprintf(stderr,
                "Errore: impossibile creare il file dati '%s'; "
                "un file esistente non viene sovrascritto.\n",
                p->data_file);
        return NULL;
    }
    if (fprintf(file,
                "# Ising 3D\n"
                "# L = %d\n"
                "# beta = %.17g\n"
                "# seed = %d\n"
                "# start = %s\n"
                "# algorithm = %s\n"
                "# sigma = %.17g\n"
                "# n_therm = %d\n"
                "# n_sweeps = %d\n"
                "# measure_every = %d\n"
                "# config_file = %s\n"
                "%s",
                p->L, p->beta, p->seed, start_mode_name(p->start),
                algorithm_name(p->algorithm), p->sigma,
                p->n_therm, p->n_sweeps, p->measure_every,
                p->config_file, columns) < 0) {
        fclose(file);
        return NULL;
    }
    return file;
}

static int write_measurement(FILE *file, const IsingLattice *lattice,
                             int sweep,
                             const double *cos_table, const double *sin_table)
{
    int energy = ising_total_energy(lattice);
    double n = (double)lattice->n_spins;
    int magnetization = 0;
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

    for (z = 0; z < lattice->L; z++) {
        for (y = 0; y < lattice->L; y++) {
            for (x = 0; x < lattice->L; x++) {
                int index = (z * lattice->L + y) * lattice->L + x;
                int spin = lattice->spins[index];

                magnetization += spin;
                real_x += spin * cos_table[x];
                imaginary_x += spin * sin_table[x];
                real_y += spin * cos_table[y];
                imaginary_y += spin * sin_table[y];
                real_z += spin * cos_table[z];
                imaginary_z += spin * sin_table[z];
            }
        }
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
    Pcg32 rng;
    FILE *data;
    int *cluster = NULL;
    int *in_cluster = NULL;
    double wolff_probability = 0.0;
    double cos_table[ISING_MAX_L];
    double sin_table[ISING_MAX_L];
    double minimum_momentum;
    int previous_sweeps = 0;
    int sweep;
    int coordinate;
    int ok = 1;

    /* 1. Creazione del reticolo. */
    if (!ising_create(&lattice, p->L)) {
        return 0;
    }

    /* 2. Inizializzazione del generatore casuale. */
    pcg32_seed(&rng, p->seed);

    minimum_momentum = 2.0 * acos(-1.0) / lattice.L;
    for (coordinate = 0; coordinate < lattice.L; coordinate++) {
        double angle = minimum_momentum * coordinate;

        cos_table[coordinate] = cos(angle);
        sin_table[coordinate] = sin(angle);
    }

    /* 3. Preparazione iniziale oppure caricamento di un restart. */
    if (p->start == START_ORDERED) {
        ising_fill_ordered(&lattice);
    } else if (p->start == START_RANDOM) {
        ising_fill_random(&lattice, &rng);
    } else {
        if (!ising_load_configuration(p->config_file, &lattice, &previous_sweeps)) {
            ising_destroy(&lattice);
            return 0;
        }
    }

    if (p->algorithm == ALGORITHM_WOLFF) {
        cluster = malloc(lattice.n_spins * sizeof(*cluster));
        in_cluster = calloc(lattice.n_spins, sizeof(*in_cluster));
        if (cluster == NULL || in_cluster == NULL) {
            fprintf(stderr, "Errore: memoria insufficiente per il cluster Wolff.\n");
            free(cluster);
            free(in_cluster);
            ising_destroy(&lattice);
            return 0;
        }
        wolff_probability = 1.0 - exp(-2.0 * p->beta);
    }

    /* 4. Apertura del file delle misure. */
    data = open_data_file(p, previous_sweeps);
    if (data == NULL) {
        free(cluster);
        free(in_cluster);
        ising_destroy(&lattice);
        return 0;
    }

    printf("Reticolo %d x %d x %d, start=%s, algorithm=%s, seed=%d, sigma=%.17g\n",
           p->L, p->L, p->L, start_mode_name(p->start),
           algorithm_name(p->algorithm), p->seed, p->sigma);

    /* 5. Termalizzazione: questi sweep non producono misure. */
    for (sweep = 0; sweep < p->n_therm; sweep++) {
        if (p->algorithm == ALGORITHM_METROPOLIS) {
            ising_metropolis_sweep(&lattice, p->beta, p->sigma, &rng);
        } else {
            ising_wolff_update(&lattice, wolff_probability, &rng,
                               cluster, in_cluster);
        }
    }

    /* 6. Produzione e scrittura periodica delle misure. */
    for (sweep = 1; sweep <= p->n_sweeps; sweep++) {
        if (p->algorithm == ALGORITHM_METROPOLIS) {
            ising_metropolis_sweep(&lattice, p->beta, p->sigma, &rng);
        } else {
            ising_wolff_update(&lattice, wolff_probability,
                               &rng, cluster, in_cluster);
        }

        if (sweep % p->measure_every == 0) {
            if (!write_measurement(data, &lattice, previous_sweeps + sweep,
                                   cos_table, sin_table)) {
                fprintf(stderr, "Errore durante la scrittura delle misure.\n");
                ok = 0;
                break;
            }
        }
    }

    /* 7. Chiusura del file dati. */
    if (fclose(data) != 0) {
        fprintf(stderr, "Errore durante la chiusura del file dati.\n");
        ok = 0;
    }

    /* 8. Salvataggio della configurazione finale. */
    if (ok) {
        if (!ising_save_configuration(p->config_file, &lattice,
                                      previous_sweeps + p->n_sweeps)) {
            ok = 0;
        }
    }

    if (ok) {
        printf("Completati %d sweep; totale salvato: %d.\n",
               p->n_sweeps, previous_sweeps + p->n_sweeps);
        printf("Configurazione finale: %s\n", p->config_file);
    }

    /* 9. Liberazione della memoria del reticolo. */
    free(cluster);
    free(in_cluster);
    ising_destroy(&lattice);
    return ok;
}

int main(int argc, char **argv)
{
    SimulationParameters parameters;

    if (argc != 2) {
        fprintf(stderr, "Uso: %s input.dat\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!input_read(argv[1], &parameters)) {
        return EXIT_FAILURE;
    }
    if (!run_simulation(&parameters)) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
