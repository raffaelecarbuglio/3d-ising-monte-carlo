#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "input.h"
#include "ising.h"

typedef struct {
    int production_sweeps;
    int thermal_remaining;
    long data_offset;
    char data_file[INPUT_PATH_SIZE];
    Pcg32 rng;
} Checkpoint;

int checkpoint_save(const SimulationParameters *p, const IsingLattice *lattice,
                    const Checkpoint *state);
/* I vecchi file di soli spin sono leggibili; complete vale 0 in quel caso. */
int checkpoint_load(const SimulationParameters *p, IsingLattice *lattice,
                    Checkpoint *state, int *complete);

#endif
