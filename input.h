#ifndef INPUT_H
#define INPUT_H

#define INPUT_PATH_SIZE 512

typedef enum {
    START_ORDERED,
    START_RANDOM,
    START_RESTART
} StartMode;

typedef enum {
    ALGORITHM_METROPOLIS,
    ALGORITHM_WOLFF
} Algorithm;

typedef struct {
    int L;
    double beta;
    double sigma;
    int n_therm;
    int n_sweeps;
    int measure_every;
    int seed;
    StartMode start;
    Algorithm algorithm;
    char config_file[INPUT_PATH_SIZE];
    char data_file[INPUT_PATH_SIZE];
} SimulationParameters;

int input_read(const char *filename, SimulationParameters *parameters);
const char *start_mode_name(StartMode mode);
const char *algorithm_name(Algorithm algorithm);

#endif
