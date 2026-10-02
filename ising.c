#include "ising.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int wrap_neighbor_coordinate(int coordinate, int L)
{
    if (coordinate == -1) {
        return L - 1;
    }
    if (coordinate == L) {
        return 0;
    }
    return coordinate;
}

int ising_create(IsingLattice *lattice, int L)
{
    if (L < 2) {
        fprintf(stderr, "Errore: L deve essere almeno 2.\n");
        return 0;
    }
    if (L > ISING_MAX_L) {
        fprintf(stderr, "Errore: L non puo' superare %d.\n", ISING_MAX_L);
        return 0;
    }

    lattice->n_spins = L * L * L;

    lattice->spins = malloc(lattice->n_spins * sizeof(*lattice->spins));
    if (lattice->spins == NULL) {
        fprintf(stderr, "Errore: memoria insufficiente per il reticolo.\n");
        ising_destroy(lattice);
        return 0;
    }

    lattice->L = L;
    return 1;
}

void ising_destroy(IsingLattice *lattice)
{
    free(lattice->spins);
    memset(lattice, 0, sizeof(*lattice));
}

void ising_fill_ordered(IsingLattice *lattice)
{
    int i;

    for (i = 0; i < lattice->n_spins; i++) {
        lattice->spins[i] = 1;
    }
}

void ising_fill_random(IsingLattice *lattice, Pcg32 *rng)
{
    int i;

    for (i = 0; i < lattice->n_spins; i++) {
        if (pcg32_bit(rng) == 0) {
            lattice->spins[i] = -1;
        } else {
            lattice->spins[i] = 1;
        }
    }
}

static int ising_index(const IsingLattice *lattice, int x, int y, int z)
{
    /* x varia piu' rapidamente nella disposizione lineare del reticolo. */
    return (z * lattice->L + y) * lattice->L + x;
}

static int ising_index_periodic(const IsingLattice *lattice,
                                int x, int y, int z)
{
    int wx = wrap_neighbor_coordinate(x, lattice->L);
    int wy = wrap_neighbor_coordinate(y, lattice->L);
    int wz = wrap_neighbor_coordinate(z, lattice->L);

    return ising_index(lattice, wx, wy, wz);
}

static int ising_neighbor_sum(const IsingLattice *lattice, int x, int y, int z)
{
    /* Le coordinate sono avvolte per applicare le condizioni periodiche. */
    return lattice->spins[ising_index_periodic(lattice, x - 1, y, z)]
         + lattice->spins[ising_index_periodic(lattice, x + 1, y, z)]
         + lattice->spins[ising_index_periodic(lattice, x, y - 1, z)]
         + lattice->spins[ising_index_periodic(lattice, x, y + 1, z)]
         + lattice->spins[ising_index_periodic(lattice, x, y, z - 1)]
         + lattice->spins[ising_index_periodic(lattice, x, y, z + 1)];
}

int ising_total_energy(const IsingLattice *lattice)
{
    int energy = 0;
    int x;
    int y;
    int z;

    for (z = 0; z < lattice->L; z++) {
        for (y = 0; y < lattice->L; y++) {
            for (x = 0; x < lattice->L; x++) {
                int spin = lattice->spins[ising_index(lattice, x, y, z)];

                energy -= spin
                        * lattice->spins[ising_index_periodic(lattice, x + 1, y, z)];
                energy -= spin
                        * lattice->spins[ising_index_periodic(lattice, x, y + 1, z)];
                energy -= spin
                        * lattice->spins[ising_index_periodic(lattice, x, y, z + 1)];
            }
        }
    }
    return energy;
}

int ising_total_magnetization(const IsingLattice *lattice)
{
    int magnetization = 0;
    int i;

    for (i = 0; i < lattice->n_spins; i++) {
        magnetization += lattice->spins[i];
    }
    return magnetization;
}

/* exp(x*x) erfc(x), per x >= 0, senza overflow nella coda. */
static double scaled_erfc(double x)
{
    if (x < 26.0) {
        return exp(x * x) * erfc(x);
    }
    double term = 1.0, sum = 1.0;
    double inverse_square = (1.0 / x) * (1.0 / x);
    for (int k = 1; k <= 8; k++) {
        term *= -(2.0 * k - 1.0) * 0.5 * inverse_square;
        sum += term;
    }
    return sum / (sqrt(acos(-1.0)) * x);
}

double ising_noisy_acceptance(int delta_energy, double beta, double sigma)
{
    if (beta == 0.0) {
        return 1.0;
    }
    if (sigma == 0.0) {
        return delta_energy <= 0 ? 1.0 : exp(-beta * delta_energy);
    }
    double a = delta_energy / sigma;
    double b = beta * sigma;
    double z = (b - a) / sqrt(2.0);
    double tail;

    /* Media di min(1, exp(-beta*(delta_energy + sigma*G))).
       Per z > 0 riscriviamo il prodotto exp(...) erfc(z), evitando inf*0. */
    if (z > 0.0) {
        tail = 0.5 * exp(-0.5 * a * a) * scaled_erfc(z);
    } else {
        tail = 0.5 * exp(b * (0.5 * b - a)) * erfc(z);
    }
    return fmin(1.0, 0.5 * erfc(a / sqrt(2.0)) + tail);
}

int ising_metropolis_sweep(IsingLattice *lattice, double beta, double sigma, Pcg32 *rng)
{
    double acceptance_probability[4];
    /* Cache per le simulazioni seriali; aggiornata se cambiano i parametri. */
    static double noisy_probability[7];
    static double cached_beta = -1.0, cached_sigma = -1.0;
    int accepted = 0;
    int x;
    int y;
    int z;

    /* I soli delta E positivi possibili sono 4, 8 e 12. */
    acceptance_probability[1] = exp(-4.0 * beta);
    acceptance_probability[2] = exp(-8.0 * beta);
    acceptance_probability[3] = exp(-12.0 * beta);

    if (sigma > 0.0 && (beta != cached_beta || sigma != cached_sigma)) {
        for (int j = 0; j < 7; j++) {
            noisy_probability[j] = ising_noisy_acceptance(4 * (j - 3), beta, sigma);
        }
        cached_beta = beta;
        cached_sigma = sigma;
    }

    for (z = 0; z < lattice->L; z++) {
        for (y = 0; y < lattice->L; y++) {
            for (x = 0; x < lattice->L; x++) {
                int index = ising_index(lattice, x, y, z);
                int delta_energy;
                int accept;

                /* Per un flip, delta E = 2 s_i moltiplicato per i sei vicini. */
                delta_energy = 2 * lattice->spins[index]
                             * ising_neighbor_sum(lattice, x, y, z);

                if (sigma > 0.0) {
                    /* Il rumore agisce su ogni tentativo, anche per delta E <= 0. */
                    double probability = noisy_probability[delta_energy / 4 + 3];
                    accept = probability >= 1.0 || pcg32_uniform(rng) < probability;
                } else {
                    /* Caso pulito: stesse probabilita' e stessa sequenza RNG. */
                    accept = delta_energy <= 0 ||
                             pcg32_uniform(rng) < acceptance_probability[delta_energy / 4];
                }
                if (accept) {
                    lattice->spins[index] = -lattice->spins[index];
                    accepted++;
                }
            }
        }
    }
    return accepted;
}

int ising_wolff_update(IsingLattice *lattice, double probability, Pcg32 *rng,
                       int *cluster, int *in_cluster)
{
    int first = 0;
    int cluster_size = 1;
    int initial_index;
    int cluster_spin;
    int i;

    initial_index = (int)(pcg32_uniform(rng) * lattice->n_spins);
    cluster_spin = lattice->spins[initial_index];
    cluster[0] = initial_index;
    in_cluster[initial_index] = 1;

    while (first < cluster_size) {
        int index = cluster[first];
        int x = index % lattice->L;
        int y = (index / lattice->L) % lattice->L;
        int z = index / (lattice->L * lattice->L);
        int neighbors[6];
        int neighbor_number;

        first++;
        neighbors[0] = ising_index_periodic(lattice, x - 1, y, z);
        neighbors[1] = ising_index_periodic(lattice, x + 1, y, z);
        neighbors[2] = ising_index_periodic(lattice, x, y - 1, z);
        neighbors[3] = ising_index_periodic(lattice, x, y + 1, z);
        neighbors[4] = ising_index_periodic(lattice, x, y, z - 1);
        neighbors[5] = ising_index_periodic(lattice, x, y, z + 1);

        for (neighbor_number = 0; neighbor_number < 6; neighbor_number++) {
            int neighbor = neighbors[neighbor_number];

            if (!in_cluster[neighbor] &&
                lattice->spins[neighbor] == cluster_spin &&
                pcg32_uniform(rng) < probability) {
                in_cluster[neighbor] = 1;
                cluster[cluster_size] = neighbor;
                cluster_size++;
            }
        }
    }

    for (i = 0; i < cluster_size; i++) {
        int index = cluster[i];

        lattice->spins[index] = -lattice->spins[index];
        in_cluster[index] = 0;
    }

    return cluster_size;
}

int ising_save_configuration(const char *filename, const IsingLattice *lattice,
                             int production_sweeps)
{
    FILE *file;
    char *temporary_filename;
    int i;
    int ok = 1;

    /* Scrivi prima un file temporaneo nello stesso percorso del salvataggio. */
    temporary_filename = malloc(strlen(filename) + sizeof(".tmp"));
    if (temporary_filename == NULL) {
        fprintf(stderr, "Errore: memoria insufficiente per il salvataggio.\n");
        return 0;
    }
    sprintf(temporary_filename, "%s.tmp", filename);
    file = fopen(temporary_filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile scrivere la configurazione '%s'.\n", filename);
        free(temporary_filename);
        return 0;
    }

    fprintf(file, "L %d\nproduction_sweeps %d\nspins\n",
            lattice->L, production_sweeps);

    for (i = 0; i < lattice->n_spins; i++) {
        char separator;

        if ((i + 1) % lattice->L == 0) {
            separator = '\n';
        } else {
            separator = ' ';
        }

        fprintf(file, "%2d%c", lattice->spins[i], separator);
    }

    if (ferror(file)) {
        fprintf(stderr, "Errore durante la scrittura di '%s'.\n", filename);
        ok = 0;
    }

    if (fclose(file) != 0) {
        fprintf(stderr, "Errore durante la chiusura di '%s'.\n", filename);
        ok = 0;
    }

    /* Sostituisci il salvataggio precedente solo dopo una scrittura completa. */
    if (ok && rename(temporary_filename, filename) != 0) {
        fprintf(stderr, "Errore durante la sostituzione di '%s'.\n", filename);
        ok = 0;
    }
    if (!ok) {
        remove(temporary_filename);
    }
    free(temporary_filename);
    return ok;
}

int ising_load_configuration(const char *filename, IsingLattice *lattice,
                             int *production_sweeps)
{
    FILE *file;
    char label[64];
    int file_L;
    int saved_sweeps;
    int i;

    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile leggere la configurazione '%s'.\n", filename);
        return 0;
    }

    if (fscanf(file, "%63s %d", label, &file_L) != 2 ||
        strcmp(label, "L") != 0) {
        fprintf(stderr, "Errore: intestazione L non valida in '%s'.\n", filename);
        fclose(file);
        return 0;
    }

    if (file_L != lattice->L) {
        fprintf(stderr, "Errore: L=%d nell'input ma L=%d nella configurazione.\n",
                lattice->L, file_L);
        fclose(file);
        return 0;
    }

    if (fscanf(file, "%63s %d", label, &saved_sweeps) != 2 ||
        strcmp(label, "production_sweeps") != 0 ||
        fscanf(file, "%63s", label) != 1 ||
        strcmp(label, "spins") != 0) {
        fprintf(stderr, "Errore: intestazione non valida in '%s'.\n", filename);
        fclose(file);
        return 0;
    }

    for (i = 0; i < lattice->n_spins; i++) {
        if (fscanf(file, "%d", &lattice->spins[i]) != 1 ||
            (lattice->spins[i] != -1 && lattice->spins[i] != 1)) {
            fprintf(stderr, "Errore: spin %d non valido in '%s'.\n", i, filename);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    *production_sweeps = saved_sweeps;
    return 1;
}

