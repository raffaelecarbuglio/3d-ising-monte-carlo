#ifndef Z2_H
#define Z2_H

#include "rng.h"

/* This bound keeps 3 * L^3 and the total energy inside a signed int. */
#define Z2_MAX_L 512

typedef struct {
    int L;
    int volume;      /* L^3 sites */
    int *links;      /* links[3 * site + mu], mu = 0, 1, 2 */
} GaugeLattice;

typedef struct {
    int proposed;    /* actual flip proposals, after the 50% skip */
    int accepted;
} SweepResult;

int z2_create(GaugeLattice *g, int L);
void z2_destroy(GaugeLattice *g);
void z2_fill_ordered(GaugeLattice *g);
void z2_fill_random(GaugeLattice *g, Pcg32 *rng);

/* site = (z * L + y) * L + x; mu and nu are distinct directions. */
int z2_plaquette(const GaugeLattice *g, int site, int mu, int nu);
int z2_delta_energy(const GaugeLattice *g, int site, int mu);
int z2_total_energy(const GaugeLattice *g);
SweepResult z2_metropolis_sweep(GaugeLattice *g, double beta, Pcg32 *rng);

#endif
