#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t increment;
} Pcg32;

void pcg32_seed(Pcg32 *rng, uint64_t seed);
uint32_t pcg32_bit(Pcg32 *rng);
double pcg32_uniform(Pcg32 *rng);

#endif
