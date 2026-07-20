#include "input.h"
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int input_read(const char *filename, SimulationParameters *parameters)
{
    FILE *file;
    char start[16];
    int fields_read = 0;

    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile aprire il file di input '%s'.\n", filename);
        return 0;
    }

    fields_read += fscanf(file, "L = %d", &parameters->L);
    fields_read += fscanf(file, "\nbeta = %lf", &parameters->beta);
    fields_read += fscanf(file, "\nn_therm = %" SCNu64, &parameters->n_therm);
    fields_read += fscanf(file, "\nn_sweeps = %" SCNu64, &parameters->n_sweeps);
    fields_read += fscanf(file, "\nmeasure_every = %" SCNu64,
                          &parameters->measure_every);
    fields_read += fscanf(file, "\nseed = %" SCNu64, &parameters->seed);
    fields_read += fscanf(file, "\nstart = %15s", start);

    if (fields_read != 7) {
        fprintf(stderr, "Errore: formato del file di input non valido.\n");
        fclose(file);
        return 0;
    }

    if (strcmp(start, "ordered") == 0) {
        parameters->start = START_ORDERED;
    } else if (strcmp(start, "random") == 0) {
        parameters->start = START_RANDOM;
    } else if (strcmp(start, "restart") == 0) {
        parameters->start = START_RESTART;
    } else {
        fprintf(stderr, "Errore: start deve essere ordered, random oppure restart.\n");
        fclose(file);
        return 0;
    }

    if (fscanf(file, "\nconfig_file = %511s", parameters->config_file) != 1 ||
        fscanf(file, "\ndata_file = %511s", parameters->data_file) != 1) {
        fprintf(stderr, "Errore: mancano config_file o data_file.\n");
        fclose(file);
        return 0;
    }

    if (parameters->L < 2) {
        fprintf(stderr, "Errore: L deve essere almeno 2.\n");
        fclose(file);
        return 0;
    }
    if (!(parameters->beta >= 0.0)) {
        fprintf(stderr, "Errore: beta deve essere non negativa.\n");
        fclose(file);
        return 0;
    }
    if (parameters->n_sweeps == 0) {
        fprintf(stderr, "Errore: n_sweeps deve essere maggiore di zero.\n");
        fclose(file);
        return 0;
    }
    if (parameters->measure_every == 0) {
        fprintf(stderr, "Errore: measure_every deve essere maggiore di zero.\n");
        fclose(file);
        return 0;
    }
    fclose(file);
    return 1;
}

const char *start_mode_name(StartMode mode)
{
    if (mode == START_ORDERED) {
        return "ordered";
    }
    if (mode == START_RANDOM) {
        return "random";
    }
    return "restart";
}
