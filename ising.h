#ifndef ISING_H
#define ISING_H

#include <stddef.h>
#include <stdint.h>
#include "rng.h"

#define ISING_MAX_L 512

typedef struct {
    int L;
    size_t n_spins;
    int *spins;
} IsingLattice;

/* L deve essere compreso tra 2 e ISING_MAX_L. */
int ising_create(IsingLattice *lattice, int L);
void ising_destroy(IsingLattice *lattice);
void ising_fill_ordered(IsingLattice *lattice);
void ising_fill_random(IsingLattice *lattice, Pcg32 *rng);
int64_t ising_total_energy(const IsingLattice *lattice);
int64_t ising_total_magnetization(const IsingLattice *lattice);
uint64_t ising_metropolis_sweep(IsingLattice *lattice, double beta, Pcg32 *rng);
int ising_save_configuration(const char *filename, const IsingLattice *lattice,
                             uint64_t production_sweeps);
int ising_load_configuration(const char *filename, IsingLattice *lattice,
                             uint64_t *production_sweeps);

#endif
