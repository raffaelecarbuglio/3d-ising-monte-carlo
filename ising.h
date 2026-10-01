#ifndef ISING_H
#define ISING_H

#include "rng.h"

#define ISING_MAX_L 512

typedef struct {
    int L;
    int n_spins;
    int *spins;
} IsingLattice;

/* L deve essere compreso tra 2 e ISING_MAX_L. */
int ising_create(IsingLattice *lattice, int L);
void ising_destroy(IsingLattice *lattice);
void ising_fill_ordered(IsingLattice *lattice);
void ising_fill_random(IsingLattice *lattice, Pcg32 *rng);
int ising_total_energy(const IsingLattice *lattice);
int ising_total_magnetization(const IsingLattice *lattice);
/* beta >= 0, sigma >= 0; delta_energy in {-12,-8,-4,0,4,8,12}. */
double ising_noisy_acceptance(int delta_energy, double beta, double sigma);
int ising_metropolis_sweep(IsingLattice *lattice, double beta, double sigma, Pcg32 *rng);
int ising_wolff_update(IsingLattice *lattice, double probability, Pcg32 *rng,
                       int *cluster, int *in_cluster);
int ising_save_configuration(const char *filename, const IsingLattice *lattice,
                             int production_sweeps);
int ising_load_configuration(const char *filename, IsingLattice *lattice,
                             int *production_sweeps);

#endif

