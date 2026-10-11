#include "input.h"
#include <stdio.h>
#include <math.h>
#include <string.h>

int input_read(const char *filename, SimulationParameters *parameters)
{
    FILE *file;
    char start[16];
    char algorithm[16];
    int fields_read = 0;
    char extra;

    file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile aprire il file di input '%s'.\n", filename);
        return 0;
    }

    fields_read += fscanf(file, "L = %d", &parameters->L);
    fields_read += fscanf(file, "\nbeta = %lf", &parameters->beta);
    fields_read += fscanf(file, "\nn_therm = %d", &parameters->n_therm);
    fields_read += fscanf(file, "\nn_sweeps = %d", &parameters->n_sweeps);
    fields_read += fscanf(file, "\nmeasure_every = %d",
                          &parameters->measure_every);
    fields_read += fscanf(file, "\nseed = %d", &parameters->seed);
    fields_read += fscanf(file, "\nstart = %15s", start);
    fields_read += fscanf(file, "\nalgorithm = %15s", algorithm);

    if (fields_read != 8) {
        fprintf(stderr, "Errore: formato del file di input non valido.\n");
        fclose(file);
        return 0;
    }

    if (strcmp(algorithm, "metropolis") == 0) {
        parameters->algorithm = ALGORITHM_METROPOLIS;
    } else if (strcmp(algorithm, "wolff") == 0) {
        parameters->algorithm = ALGORITHM_WOLFF;
    } else {
        fprintf(stderr, "Errore: algorithm deve essere metropolis oppure wolff.\n");
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

    /* Campo finale opzionale: gli input precedenti restano validi. */
    parameters->sigma = 0.0;
    if (fscanf(file, " %c", &extra) == 1) {
        ungetc(extra, file);
        if (fscanf(file, "sigma = %lf", &parameters->sigma) != 1 ||
            fscanf(file, " %c", &extra) == 1) {
            fprintf(stderr, "Errore: atteso sigma = valore alla fine dell'input.\n");
            fclose(file);
            return 0;
        }
    }
    if (!isfinite(parameters->sigma) || parameters->sigma < 0.0 ||
        (parameters->algorithm == ALGORITHM_WOLFF && parameters->sigma != 0.0)) {
        fprintf(stderr, "Errore: sigma deve essere finito e >= 0; con Wolff deve essere 0.\n");
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
    if (parameters->n_therm < 0) {
        fprintf(stderr, "Errore: n_therm non puo' essere negativo.\n");
        fclose(file);
        return 0;
    }
    if (parameters->n_sweeps <= 0) {
        fprintf(stderr, "Errore: n_sweeps deve essere maggiore di zero.\n");
        fclose(file);
        return 0;
    }
    if (parameters->measure_every <= 0) {
        fprintf(stderr, "Errore: measure_every deve essere maggiore di zero.\n");
        fclose(file);
        return 0;
    }
    if (parameters->seed < 0) {
        fprintf(stderr, "Errore: seed non puo' essere negativo.\n");
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

const char *algorithm_name(Algorithm algorithm)
{
    if (algorithm == ALGORITHM_METROPOLIS) {
        return "metropolis";
    }
    return "wolff";
}
