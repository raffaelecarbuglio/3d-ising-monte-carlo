#include "rng.h"

static uint32_t pcg32_next(Pcg32 *rng)
{
    uint64_t old_state = rng->state;
    uint32_t shifted = (uint32_t)(((old_state >> 18u) ^ old_state) >> 27u);
    uint32_t rotation = (uint32_t)(old_state >> 59u);
    rng->state = old_state * UINT64_C(6364136223846793005) + rng->increment;
    return (shifted >> rotation) | (shifted << ((-rotation) & 31u));
}

void pcg32_seed(Pcg32 *rng, uint64_t seed)
{
    rng->state = 0u;
    rng->increment = (UINT64_C(54) << 1u) | 1u;
    pcg32_next(rng);
    rng->state += seed;
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
