#include "z2.h"

#include <stdio.h>
#include <stdlib.h>

/* Find a neighbor by coordinates while building the table. */
static int neighbor_index(const GaugeLattice *g, int site, int direction, int step)
{
    int x = site % g->L;
    int y = (site / g->L) % g->L;
    int z = site / (g->L * g->L);
    int coordinates[3] = {x, y, z};

    coordinates[direction] += step;
    if (coordinates[direction] < 0)
        coordinates[direction] += g->L;
    if (coordinates[direction] == g->L)
        coordinates[direction] = 0;

    return (coordinates[2] * g->L + coordinates[1]) * g->L
           + coordinates[0];
}

int z2_create(GaugeLattice *g, int L)
{
    *g = (GaugeLattice){0};

    if (L < 2 || L > Z2_MAX_L) {
        fprintf(stderr, "L must be between 2 and %d.\n", Z2_MAX_L);
        return 0;
    }

    g->L = L;
    g->volume = L * L * L;
    g->links = malloc((size_t)(3 * g->volume) * sizeof(*g->links));
    g->neighbors = malloc((size_t)g->volume * sizeof(*g->neighbors));
    if (g->links == NULL || g->neighbors == NULL) {
        fprintf(stderr, "Not enough memory for the lattice.\n");
        z2_destroy(g);
        return 0;
    }

    for (int site = 0; site < g->volume; site++) {
        for (int direction = 0; direction < 3; direction++) {
            g->neighbors[site][2 * direction] = neighbor_index(g, site, direction, -1);
            g->neighbors[site][2 * direction + 1] = neighbor_index(g, site, direction, 1);
        }
    }
    return 1;
}

void z2_destroy(GaugeLattice *g)
{
    free(g->links);
    free(g->neighbors);
    *g = (GaugeLattice){0};
}

void z2_fill_ordered(GaugeLattice *g)
{
    for (int i = 0; i < 3 * g->volume; i++)
        g->links[i] = 1;
}

void z2_fill_random(GaugeLattice *g, Pcg32 *rng)
{
    for (int i = 0; i < 3 * g->volume; i++)
        g->links[i] = pcg32_bit(rng) ? 1 : -1;
}

/* Each direction has a pair of entries, backward first and forward second. */
static int neighbor(const GaugeLattice *g, int site, int direction, int step)
{
    int side = step > 0 ? 1 : 0;
    return g->neighbors[site][2 * direction + side];
}

int z2_plaquette(const GaugeLattice *g, int site, int mu, int nu)
{
    int along_mu = neighbor(g, site, mu, 1);
    int along_nu = neighbor(g, site, nu, 1);

    /* The four links around the elementary square in the mu-nu plane.
       A Z2 link has the same value when traversed backwards. */
    return g->links[3 * site + mu]
         * g->links[3 * along_mu + nu]
         * g->links[3 * along_nu + mu]
         * g->links[3 * site + nu];
}

int z2_delta_energy(const GaugeLattice *g, int site, int mu)
{
    int sum = 0;

    for (int nu = 0; nu < 3; nu++) {
        if (nu == mu)
            continue;

        /* Two squares in this plane contain the chosen link.
           The other square starts at site - nu, not at site - mu. */
        int behind = neighbor(g, site, nu, -1);

        sum += z2_plaquette(g, site, mu, nu);
        sum += z2_plaquette(g, behind, mu, nu);
    }

    /* Each affected contribution changes from -P to +P. */
    return 2 * sum;
}

int z2_total_energy(const GaugeLattice *g)
{
    int energy = 0;

    for (int site = 0; site < g->volume; site++) {
        /* Each plaquette is counted once: xy, xz, yz. */
        energy -= z2_plaquette(g, site, 0, 1);
        energy -= z2_plaquette(g, site, 0, 2);
        energy -= z2_plaquette(g, site, 1, 2);
    }
    return energy;
}

SweepResult z2_metropolis_sweep(GaugeLattice *g, const double probability[3],
                                Pcg32 *rng)
{
    SweepResult result = {0, 0};

    for (int site = 0; site < g->volume; site++) {
        for (int mu = 0; mu < 3; mu++) {
            /* Give every link a 50% chance to stay unchanged.
               Without this, at beta=0 a fixed pass would flip every
               link and leave every plaquette unchanged after a sweep. */
            if (pcg32_bit(rng) == 0)
                continue;

            result.proposed++;
            int dE = z2_delta_energy(g, site, mu);

            if (dE <= 0 || pcg32_uniform(rng) < probability[dE / 4]) {
                g->links[3 * site + mu] *= -1;
                result.accepted++;
            }
        }
    }
    return result;
}
