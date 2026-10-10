#include "checkpoint.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int checkpoint_save(const SimulationParameters *p, const IsingLattice *lattice,
                    const Checkpoint *state)
{
    char temporary[INPUT_PATH_SIZE + 8];
    FILE *file;
    int ok;

    snprintf(temporary, sizeof(temporary), "%s.tmp", p->config_file);
    file = fopen(temporary, "w");
    if (file == NULL) return 0;

    /* Lo stesso prefisso dei vecchi file di configurazione. */
    fprintf(file, "L %d\nproduction_sweeps %d\nspins\n",
            lattice->L, state->production_sweeps);
    for (int i = 0; i < lattice->n_spins; i++) {
        fprintf(file, "%2d%c", lattice->spins[i],
                (i + 1) % lattice->L == 0 ? '\n' : ' ');
    }
    fprintf(file,
            "checkpoint 1\n"
            "beta %.17g\nsigma %.17g\nalgorithm %d\nmeasure_every %d\n"
            "thermal_remaining %d\ndata_offset %ld\ndata_file %s\n"
            "rng %" PRIu64 " %" PRIu64 " %d %.17g\n",
            p->beta, p->sigma, p->algorithm, p->measure_every,
            state->thermal_remaining, state->data_offset, p->data_file,
            state->rng.state, state->rng.increment,
            state->rng.has_gaussian, state->rng.saved_gaussian);
    ok = !ferror(file);
    if (fclose(file) != 0) ok = 0;
    if (ok && rename(temporary, p->config_file) != 0) ok = 0;
    if (!ok) remove(temporary);
    return ok;
}

int checkpoint_load(const SimulationParameters *p, IsingLattice *lattice,
                    Checkpoint *state, int *complete)
{
    FILE *file = fopen(p->config_file, "r");
    int L, version, algorithm, measure_every;
    double beta, sigma;
    char marker[32];
    int ok = 0;

    *complete = 0;
    if (file == NULL) {
        fprintf(stderr, "Errore: impossibile leggere '%s'.\n", p->config_file);
        return 0;
    }
    if (fscanf(file, "L %d\nproduction_sweeps %d\n%31s",
               &L, &state->production_sweeps, marker) != 3 ||
        L != lattice->L || state->production_sweeps < 0 ||
        strcmp(marker, "spins") != 0) goto done;
    for (int i = 0; i < lattice->n_spins; i++) {
        if (fscanf(file, "%d", &lattice->spins[i]) != 1 ||
            (lattice->spins[i] != -1 && lattice->spins[i] != 1)) goto done;
    }
    if (fscanf(file, " %31s", marker) == EOF && !ferror(file)) {
        /* Legacy: il nuovo seed e n_therm vengono dal file di input. */
        ok = 1;
        goto done;
    }
    if (strcmp(marker, "checkpoint") != 0 ||
        fscanf(file,
               "%d\nbeta %lf\nsigma %lf\nalgorithm %d\nmeasure_every %d\n"
               "thermal_remaining %d\ndata_offset %ld\ndata_file %511s\n"
               "rng %" SCNu64 " %" SCNu64 " %d %lf",
               &version, &beta, &sigma, &algorithm, &measure_every,
               &state->thermal_remaining, &state->data_offset, state->data_file,
               &state->rng.state, &state->rng.increment,
               &state->rng.has_gaussian, &state->rng.saved_gaussian) != 12 ||
        version != 1 || beta != p->beta || sigma != p->sigma ||
        algorithm != (int)p->algorithm || measure_every != p->measure_every ||
        state->thermal_remaining < 0 || state->data_offset < 0 ||
        !(state->rng.increment & 1) ||
        (state->rng.has_gaussian != 0 && state->rng.has_gaussian != 1) ||
        !isfinite(state->rng.saved_gaussian)) goto done;
    if (fscanf(file, " %31s", marker) != EOF || ferror(file)) goto done;
    *complete = 1;
    ok = 1;
done:
    fclose(file);
    if (!ok) fprintf(stderr, "Errore: checkpoint non valido o parametri diversi in '%s'.\n",
                     p->config_file);
    return ok;
}
