#ifndef INPUT_H
#define INPUT_H

#include <stdint.h>

#define INPUT_PATH_SIZE 512

typedef enum {
    START_ORDERED,
    START_RANDOM,
    START_RESTART
} StartMode;

typedef struct {
    int L;
    double beta;
    uint64_t n_therm;
    uint64_t n_sweeps;
    uint64_t measure_every;
    uint64_t seed;
    StartMode start;
    char config_file[INPUT_PATH_SIZE];
    char data_file[INPUT_PATH_SIZE];
} SimulationParameters;

int input_read(const char *filename, SimulationParameters *parameters);
const char *start_mode_name(StartMode mode);

#endif
