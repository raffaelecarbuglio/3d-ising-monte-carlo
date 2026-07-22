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

int ising_metropolis_sweep(IsingLattice *lattice, double beta, Pcg32 *rng)
{
    int accepted = 0;
    int x;
    int y;
    int z;

    for (z = 0; z < lattice->L; z++) {
        for (y = 0; y < lattice->L; y++) {
            for (x = 0; x < lattice->L; x++) {
                int index = ising_index(lattice, x, y, z);
                int delta_energy;

                /* Per un flip, delta E = 2 s_i moltiplicato per i sei vicini. */
                delta_energy = 2 * lattice->spins[index]
                             * ising_neighbor_sum(lattice, x, y, z);

                /* Metropolis: i flip favorevoli sono sempre accettati. */
                if (delta_energy <= 0 ||
                    pcg32_uniform(rng) < exp(-beta * delta_energy)) {
                    lattice->spins[index] = -lattice->spins[index];
                    accepted++;
                }
            }
        }
    }
    return accepted;
}

int ising_save_configuration(const char *filename, const IsingLattice *lattice,
                             int production_sweeps)
{
    FILE *file;
    int i;
    int ok = 1;

    file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile scrivere la configurazione '%s'.\n", filename);
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
