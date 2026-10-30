#include "z2_gauge/z2.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Independent coordinate arithmetic, used only to check the geometry. */
static int site_at(int L, int x, int y, int z)
{
    return (((z + L) % L) * L + (y + L) % L) * L + (x + L) % L;
}

static void test_geometry(int L)
{
    GaugeLattice g;
    Pcg32 rng;
    int created = z2_create(&g, L);
    assert(created);

    z2_fill_ordered(&g);
    assert(z2_total_energy(&g) == -3 * g.volume);
    g.links[0] = -1;
    assert(z2_total_energy(&g) == -3 * g.volume + 8);

    pcg32_seed(&rng, 1234 + L);
    z2_fill_random(&g, &rng);

    /* Check every link, including all links crossing periodic boundaries. */
    for (int site = 0; site < g.volume; site++) {
        for (int mu = 0; mu < 3; mu++) {
            int before = z2_total_energy(&g);
            int dE = z2_delta_energy(&g, site, mu);
            assert(dE >= -8 && dE <= 8 && dE % 4 == 0);

            g.links[3 * site + mu] *= -1;
            assert(z2_total_energy(&g) - before == dE);
            g.links[3 * site + mu] *= -1;
        }
    }

    /* A gauge transformation flips the six links touching one site.
       Check each plaquette, rather than only the total energy. */
    int planes[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    int *before = malloc((size_t)(3 * g.volume) * sizeof(*before));
    assert(before != NULL);

    for (int site = 0; site < g.volume; site++) {
        for (int plane = 0; plane < 3; plane++)
            before[3 * site + plane] =
                z2_plaquette(&g, site, planes[plane][0], planes[plane][1]);
    }

    for (int z = 0; z < L; z++) {
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                int site = site_at(L, x, y, z);
                int incoming[3] = {site_at(L, x - 1, y, z),
                                   site_at(L, x, y - 1, z),
                                   site_at(L, x, y, z - 1)};
                for (int mu = 0; mu < 3; mu++) {
                    g.links[3 * site + mu] *= -1;
                    g.links[3 * incoming[mu] + mu] *= -1;
                }

                for (int other = 0; other < g.volume; other++) {
                    for (int plane = 0; plane < 3; plane++)
                        assert(z2_plaquette(&g, other, planes[plane][0],
                                            planes[plane][1])
                               == before[3 * other + plane]);
                }
            }
        }
    }
    free(before);
    z2_destroy(&g);
    printf("PASS: ordered energy, local energy changes, gauge symmetry (L=%d).\n", L);
}

static void test_zero_beta(void)
{
    GaugeLattice g;
    Pcg32 rng;
    int created = z2_create(&g, 3);
    assert(created);
    pcg32_seed(&rng, 91234);
    z2_fill_ordered(&g);

    double sum = 0.0;
    double sum_squared = 0.0;
    const int samples = 10000;
    for (int i = 0; i < samples; i++) {
        SweepResult result = z2_metropolis_sweep(&g, 0.0, &rng);
        assert(result.accepted == result.proposed);
        double energy = z2_total_energy(&g);
        sum += energy;
        sum_squared += energy * energy;
    }

    /* At beta=0, <E>=0 and <E^2>=3 L^3 (here L=3). */
    assert(fabs(sum / samples) < 0.5);
    assert(fabs(sum_squared / samples - 3 * g.volume) < 5.0);
    z2_destroy(&g);
    puts("PASS: beta=0 samples the expected energy moments.");
}

int main(void)
{
    test_geometry(2);
    test_geometry(3);
    test_geometry(4);
    test_zero_beta();
    return EXIT_SUCCESS;
}
