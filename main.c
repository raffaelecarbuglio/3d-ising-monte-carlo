#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "input.h"
#include "ising.h"
#include "rng.h"

static FILE *open_data_file(const SimulationParameters *p, uint64_t initial_sweep)
{
    FILE *file;

    if (p->start == START_RESTART) {
        /* Un restart aggiunge un segmento breve al file esistente o ne crea uno. */
        file = fopen(p->data_file, "a");
        if (file == NULL) {
            fprintf(stderr, "Errore: impossibile aggiungere dati a '%s'.\n", p->data_file);
            return NULL;
        }
        if (fprintf(file,
                    "#\n"
                    "# restart: initial_sweep=%" PRIu64 " beta=%.17g "
                    "seed=%" PRIu64 " n_therm=%" PRIu64 " "
                    "n_sweeps=%" PRIu64 " measure_every=%" PRIu64 "\n"
                    "# config_file=%s\n",
                    initial_sweep, p->beta, p->seed, p->n_therm,
                    p->n_sweeps, p->measure_every, p->config_file) < 0) {
            fclose(file);
            return NULL;
        }
        return file;
    }

    file = fopen(p->data_file, "wx");
    if (file == NULL) {
        if (errno == EEXIST) {
            fprintf(stderr,
                    "Errore: il file dati '%s' esiste gia'; non verra' sovrascritto.\n",
                    p->data_file);
        } else {
            fprintf(stderr, "Errore: impossibile creare il file dati '%s'.\n",
                    p->data_file);
        }
        return NULL;
    }
    if (fprintf(file,
                "# Ising 3D Metropolis\n"
                "# L = %d\n"
                "# beta = %.17g\n"
                "# seed = %" PRIu64 "\n"
                "# start = %s\n"
                "# n_therm = %" PRIu64 "\n"
                "# n_sweeps = %" PRIu64 "\n"
                "# measure_every = %" PRIu64 "\n"
                "# config_file = %s\n"
                "# columns: sweep energy_per_spin magnetization_per_spin "
                "abs_magnetization_per_spin acceptance\n",
                p->L, p->beta, p->seed, start_mode_name(p->start),
                p->n_therm, p->n_sweeps, p->measure_every,
                p->config_file) < 0) {
        fclose(file);
        return NULL;
    }
    return file;
}

static int write_measurement(FILE *file, const IsingLattice *lattice,
                             uint64_t sweep, uint64_t accepted, uint64_t attempts)
{
    int64_t energy = ising_total_energy(lattice);
    int64_t magnetization = ising_total_magnetization(lattice);
    double n = (double)lattice->n_spins;
    double m = (double)magnetization / n;
    double abs_m = fabs(m);

    if (fprintf(file, "%" PRIu64 " %.12g %.12g %.12g %.12g\n",
                sweep, (double)energy / n, m, abs_m,
                (double)accepted / (double)attempts) < 0) {
        return 0;
    }
    return 1;
}

static int run_simulation(const SimulationParameters *p)
{
    IsingLattice lattice;
    Pcg32 rng;
    FILE *data;
    uint64_t previous_sweeps = 0;
    uint64_t accepted = 0;
    uint64_t attempts = 0;
    uint64_t sweep;
    int ok = 1;

    /* 1. Creazione del reticolo. */
    if (!ising_create(&lattice, p->L)) {
        return 0;
    }

    /* 2. Inizializzazione del generatore casuale. */
    pcg32_seed(&rng, p->seed);

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

    /* 4. Apertura del file delle misure. */
    data = open_data_file(p, previous_sweeps);
    if (data == NULL) {
        ising_destroy(&lattice);
        return 0;
    }

    printf("Reticolo %d x %d x %d, start=%s, seed=%" PRIu64 "\n",
           p->L, p->L, p->L, start_mode_name(p->start), p->seed);

    /* 5. Termalizzazione: questi sweep non producono misure. */
    for (sweep = 0; sweep < p->n_therm; sweep++) {
        ising_metropolis_sweep(&lattice, p->beta, &rng);
    }

    /* 6. Produzione e scrittura periodica delle misure. */
    for (sweep = 1; sweep <= p->n_sweeps; sweep++) {
        accepted += ising_metropolis_sweep(&lattice, p->beta, &rng);
        attempts += (uint64_t)lattice.n_spins;

        if (sweep % p->measure_every == 0) {
            if (!write_measurement(data, &lattice, previous_sweeps + sweep,
                                   accepted, attempts)) {
                fprintf(stderr, "Errore durante la scrittura delle misure.\n");
                ok = 0;
                break;
            }
            accepted = 0;
            attempts = 0;
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
        printf("Completati %" PRIu64 " sweep; totale salvato: %" PRIu64 ".\n",
               p->n_sweeps, previous_sweeps + p->n_sweeps);
        printf("Configurazione finale: %s\n", p->config_file);
    }

    /* 9. Liberazione della memoria del reticolo. */
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
