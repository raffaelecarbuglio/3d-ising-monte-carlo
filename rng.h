#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t state;
    uint64_t increment;
    int has_gaussian;
    double saved_gaussian;
} Pcg32;

void pcg32_seed(Pcg32 *rng, int seed);
uint32_t pcg32_bit(Pcg32 *rng);
double pcg32_uniform(Pcg32 *rng);
double pcg32_gaussian(Pcg32 *rng);

#endif
