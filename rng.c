#include "rng.h"
#include <math.h>

static uint32_t pcg32_next(Pcg32 *rng)
{
    uint64_t old_state = rng->state;
    uint32_t shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    return (shifted >> rotation) | (shifted << ((-rotation) & 31u));
}

void pcg32_seed(Pcg32 *rng, int seed)
{
    rng->has_gaussian = 0;
    rng->saved_gaussian = 0.0;
    rng->state = 0u;
    rng->increment = (UINT64_C(54) << 1u) | 1u;
    pcg32_next(rng);
    rng->state += (uint64_t)seed;
    pcg32_next(rng);
}

uint32_t pcg32_bit(Pcg32 *rng)
{
    return pcg32_next(rng) % 2u;
}

double pcg32_uniform(Pcg32 *rng)
{
    return (double)pcg32_next(rng) / 4294967296.0;
}


/* Metodo polare di Marsaglia: due gaussiane indipendenti N(0,1). */
double pcg32_gaussian(Pcg32 *rng)
{
    double x, y, radius_squared, factor;

    if (rng->has_gaussian) {
        rng->has_gaussian = 0;
        return rng->saved_gaussian;
    }
    do {
        x = 2.0 * pcg32_uniform(rng) - 1.0;
        y = 2.0 * pcg32_uniform(rng) - 1.0;
        radius_squared = x*x + y*y;
    } while (radius_squared == 0.0 || radius_squared >= 1.0);

    factor = sqrt(-2.0 * log(radius_squared) / radius_squared);
    rng->saved_gaussian = y * factor;
    rng->has_gaussian = 1;
    return x * factor;
}