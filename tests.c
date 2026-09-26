#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "ising.h"
#include "rng.h"

static int tests_run;
static int tests_failed;

static void check(int condition, const char *name)
{
    tests_run++;
    if (condition) {
        printf("[OK]   %s\n", name);
    } else {
        printf("[FAIL] %s\n", name);
        tests_failed++;
    }
}

static int valid_spins(const IsingLattice *lattice)
{
    int i;

    for (i = 0; i < lattice->n_spins; i++) {
        if (lattice->spins[i] != -1 && lattice->spins[i] != 1) {
            return 0;
        }
    }
    return 1;
}

static int same_spins(const IsingLattice *a, const IsingLattice *b)
{
    if (a->n_spins != b->n_spins) {
        return 0;
    }
    if (memcmp(a->spins, b->spins,
               a->n_spins * sizeof(*a->spins)) != 0) {
        return 0;
    }
    return 1;
}

static int count_numeric_lines(const char *filename)
{
    FILE *file;
    char line[512];
    int lines = 0;

    file = fopen(filename, "r");
    if (file == NULL) {
        return -1;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        unsigned long long sweep;
        int energy;
        int magnetization;
        double g_min;
        char extra;

        if (line[0] == '#') {
            continue;
        }
        if (sscanf(line, "%llu %d %d %lf %c", &sweep, &energy,
                   &magnetization, &g_min, &extra) != 4) {
            fclose(file);
            return -1;
        }
        lines++;
    }

    fclose(file);
    return lines;
}

static unsigned long long last_sweep(const char *filename)
{
    FILE *file;
    char line[512];
    unsigned long long value = 0;

    file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        unsigned long long candidate;

        if (line[0] != '#' && sscanf(line, "%llu", &candidate) == 1) {
            value = candidate;
        }
    }

    fclose(file);
    return value;
}

static int first_measurement(const char *filename,
                             int *energy, int *magnetization, double *g_min)
{
    FILE *file;
    char line[512];
    unsigned long long sweep;

    file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] != '#' &&
            sscanf(line, "%llu %d %d %lf",
                   &sweep, energy, magnetization, g_min) == 4) {
            fclose(file);
            return 1;
        }
    }

    fclose(file);
    return 0;
}

static int write_text(const char *filename, const char *contents)
{
    FILE *file;
    int ok;

    file = fopen(filename, "w");
    if (file == NULL) {
        return 0;
    }
    ok = fputs(contents, file) != EOF;
    if (fclose(file) != 0) {
        return 0;
    }
    return ok;
}

static void test_lattice(void)
{
    IsingLattice lattice;
    Pcg32 rng;
    int i;
    int created;

    created = ising_create(&lattice, 3);
    check(created, "creazione del reticolo");
    if (!created) {
        return;
    }

    ising_fill_ordered(&lattice);
    check(valid_spins(&lattice), "ogni spin vale +1 oppure -1");
    check(ising_total_magnetization(&lattice) == 27,
          "magnetizzazione ordinata = L^3");
    check(ising_total_energy(&lattice) == -81,
          "energia ordinata = -3 L^3");

    for (i = 0; i < lattice.n_spins; i++) {
        lattice.spins[i] = -1;
    }
    lattice.spins[0] = 1;
    check(ising_total_energy(&lattice) == -69,
          "sei legami periodici per uno spin in un angolo");

    pcg32_seed(&rng, 123);
    ising_fill_ordered(&lattice);
    check(ising_metropolis_sweep(&lattice, 0.0, 0.0, &rng) == lattice.n_spins,
          "a beta=0 ogni sito e' aggiornato una volta");

    for (i = 0; i < lattice.n_spins; i++) {
        if (lattice.spins[i] != -1) {
            break;
        }
    }
    check(i == lattice.n_spins,
          "uno sweep a beta=0 inverte tutti gli spin ordinati");
    check(valid_spins(&lattice), "gli aggiornamenti mantengono spin validi");

    ising_destroy(&lattice);
}

static void test_reproducibility(double sigma)
{
    IsingLattice a;
    IsingLattice b;
    Pcg32 rng_a;
    Pcg32 rng_b;
    int sweep;

    ising_create(&a, 4);
    ising_create(&b, 4);
    ising_fill_ordered(&a);
    ising_fill_ordered(&b);
    pcg32_seed(&rng_a, 9876);
    pcg32_seed(&rng_b, 9876);

    for (sweep = 0; sweep < 8; sweep++) {
        ising_metropolis_sweep(&a, 0.22, sigma, &rng_a);
        ising_metropolis_sweep(&b, 0.22, sigma, &rng_b);
    }

    check(same_spins(&a, &b),
          "stessa configurazione e seed: stesso risultato");
    ising_destroy(&a);
    ising_destroy(&b);
}

static void test_gaussian(void)
{
    Pcg32 rng, reference;
    double sum = 0.0, sum_squared = 0.0, sum_neighbors = 0.0;
    double previous = 0.0;
    int i;
    const int n = 200000;

    pcg32_seed(&rng, 31415);
    for (i = 0; i < n; i++) {
        double value = pcg32_gaussian(&rng);
        sum += value;
        sum_squared += value * value;
        sum_neighbors += previous * value;
        previous = value;
    }
    check(fabs(sum / n) < 0.015, "gaussiana: media vicina a zero");
    check(fabs(sum_squared / n - 1.0) < 0.025,
          "gaussiana: secondo momento vicino a uno");
    check(fabs(sum_neighbors / n) < 0.015,
          "gaussiana: campioni consecutivi non correlati");

    /* Il seed deve cancellare anche il secondo campione in attesa. */
    pcg32_gaussian(&rng);
    pcg32_seed(&rng, 42);
    pcg32_seed(&reference, 42);
    check(pcg32_gaussian(&rng) == pcg32_gaussian(&reference),
          "nuovo seed azzera la cache gaussiana");
}

/* Quadratura indipendente della regola originale, separata nel punto angoloso. */
static double integrated_acceptance(int d, double beta, double sigma)
{
    double edges[3] = {-12.0, fmax(-12.0, fmin(12.0, -d / sigma)), 12.0};
    double result = 0.0;
    const int n = 8000;
    for (int part = 0; part < 2; part++) {
        double h = (edges[part + 1] - edges[part]) / n;
        for (int i = 0; i <= n; i++) {
            double g = edges[part] + i * h;
            double delta = d + sigma * g;
            double value = exp(-0.5 * g * g) / sqrt(2.0 * acos(-1.0));
            if (delta > 0.0) value *= exp(-beta * delta);
            result += h / 3.0 * (i == 0 || i == n ? 1 : (i % 2 ? 4 : 2)) * value;
        }
    }
    return result;
}

static void test_noisy_acceptance(void)
{
    const double parameters[][2] = {{0.0, 0.5}, {0.221654, 0.5},
        {0.4, 4.0}, {1.0, 0.01}, {1.0, 40.0}};
    int accurate = 1, sweep_ok = 1, limits_ok = 1;
    IsingLattice lattice;
    ising_create(&lattice, 3);
    /* I sei vicini del primo sito per L=3. */
    const int neighbors[6] = {2, 1, 6, 3, 18, 9};
    for (int k = 0; k < 5; k++) {
        double beta = parameters[k][0], sigma = parameters[k][1];
        for (int j = 0; j < 7; j++) {
            int d = 4 * (j - 3);
            double probability = ising_noisy_acceptance(d, beta, sigma);
            double reference = integrated_acceptance(d, beta, sigma);
            accurate &= isfinite(probability) && fabs(probability - reference) < 1e-9;
            /* Il primo flip verifica anche indice della tabella e cambio parametri. */
            for (int seed = 0; seed < 100; seed++) {
                Pcg32 rng, expected_rng;
                ising_fill_ordered(&lattice);
                for (int n = 0; n < 6 - j; n++) lattice.spins[neighbors[n]] = -1;
                pcg32_seed(&rng, seed);
                pcg32_seed(&expected_rng, seed);
                int flip = beta == 0.0 || pcg32_uniform(&expected_rng) < reference;
                ising_metropolis_sweep(&lattice, beta, sigma, &rng);
                sweep_ok &= lattice.spins[0] == (flip ? -1 : 1);
            }
        }
    }
    for (int d = -12; d <= 12; d += 4) {
        double clean = d <= 0 ? 1.0 : exp(-0.22 * d);
        limits_ok &= ising_noisy_acceptance(d, 0.22, 0.0) == clean;
        limits_ok &= fabs(ising_noisy_acceptance(d, 0.22, 1e-300) - clean) < 1e-14;
        limits_ok &= fabs(ising_noisy_acceptance(d, 0.22, 1e300) - 0.5) < 1e-14;
    }
    check(accurate, "accettazione mediata: accordo con quadratura per tutti i delta E");
    check(sweep_ok, "sweep rumoroso: tabella, indici e aggiornamento parametri corretti");
    check(limits_ok, "accettazione mediata: limiti pulito, rumore piccolo e grande");
    ising_destroy(&lattice);
}

static void test_wolff(void)
{
    IsingLattice lattice;
    Pcg32 rng;
    int cluster[27];
    int in_cluster[27] = {0};

    ising_create(&lattice, 3);
    pcg32_seed(&rng, 321);

    ising_fill_ordered(&lattice);
    check(ising_wolff_update(&lattice, 0.0, &rng,
                             cluster, in_cluster) == 1,
          "Wolff con probabilita' zero: cluster di uno spin");
    check(ising_total_magnetization(&lattice) == 25,
          "Wolff inverte il cluster costruito");

    ising_fill_ordered(&lattice);
    check(ising_wolff_update(&lattice, 1.0, &rng,
                             cluster, in_cluster) == lattice.n_spins,
          "Wolff con probabilita' uno: cluster completo");
    check(ising_total_magnetization(&lattice) == -lattice.n_spins,
          "Wolff inverte il reticolo completamente connesso");

    ising_destroy(&lattice);
}

static void test_save_load(void)
{
    IsingLattice original;
    IsingLattice loaded;
    Pcg32 rng;
    int saved_sweeps = 0;
    int load_ok;

    ising_create(&original, 3);
    ising_create(&loaded, 3);
    pcg32_seed(&rng, 55);
    ising_fill_random(&original, &rng);

    check(ising_save_configuration("test_config.tmp", &original, 17),
          "salvataggio configurazione");

    load_ok = ising_load_configuration("test_config.tmp", &loaded,
                                       &saved_sweeps);
    check(load_ok, "rilettura della configurazione");
    check(saved_sweeps == 17,
          "conservazione del numero di sweep salvato");
    check(load_ok && same_spins(&original, &loaded),
          "rilettura senza modificare gli spin");

    remove("test_config.tmp");
    ising_destroy(&original);
    ising_destroy(&loaded);
}

static void test_restart(void)
{
    const char *first =
        "L = 3\n"
        "beta = 0.2\n"
        "n_therm = 1\n"
        "n_sweeps = 4\n"
        "measure_every = 2\n"
        "seed = 101\n"
        "start = ordered\n"
        "algorithm = metropolis\n"
        "config_file = test_restart_config.tmp\n"
        "data_file = test_data.tmp\n";
    const char *restart =
        "L = 3\n"
        "beta = 0.2\n"
        "n_therm = 0\n"
        "n_sweeps = 4\n"
        "measure_every = 2\n"
        "seed = 202\n"
        "start = restart\n"
        "algorithm = metropolis\n"
        "config_file = test_restart_config.tmp\n"
        "data_file = test_data.tmp\n";
    int first_status;
    int restart_status;
    int measures_before;
    int measures_after;

    remove("test_data.tmp");
    write_text("test_first_input.tmp", first);
    write_text("test_restart_input.tmp", restart);

    first_status = system("./ising test_first_input.tmp > test_program_output.tmp");
    measures_before = count_numeric_lines("test_data.tmp");
    restart_status = system("./ising test_restart_input.tmp >> test_program_output.tmp");
    measures_after = count_numeric_lines("test_data.tmp");

    check(first_status == 0 && restart_status == 0,
          "simulazione iniziale e restart completi");
    check(measures_before == 2 && measures_after == 4,
          "restart conserva e aggiunge le misure");
    check(last_sweep("test_data.tmp") == 8,
          "restart continua la numerazione degli sweep");

    write_text("test_data.tmp",
               "# columns: sweep energy_per_spin magnetization_per_spin "
               "abs_magnetization_per_spin acceptance g_zero g_min\n"
               "8 -3 1 1 0 27 0\n");
    restart_status = system("./ising test_restart_input.tmp >> test_program_output.tmp 2>&1");
    check(restart_status != 0 && last_sweep("test_data.tmp") == 8,
          "restart rifiuta il vecchio formato senza aggiungere misure");

    remove("test_data.tmp");
    restart_status = system("./ising test_restart_input.tmp >> test_program_output.tmp");
    check(restart_status == 0 && count_numeric_lines("test_data.tmp") == 2 &&
          last_sweep("test_data.tmp") == 12,
          "restart puo' continuare in un nuovo file dati");

    remove("test_first_input.tmp");
    remove("test_restart_input.tmp");
    remove("test_restart_config.tmp");
    remove("test_data.tmp");
    remove("test_program_output.tmp");
}

static void test_wolff_simulation(void)
{
    const char *input =
        "L = 3\n"
        "beta = 100\n"
        "n_therm = 1\n"
        "n_sweeps = 4\n"
        "measure_every = 2\n"
        "seed = 303\n"
        "start = ordered\n"
        "algorithm = wolff\n"
        "config_file = test_wolff_config.tmp\n"
        "data_file = test_wolff_data.tmp\n";
    int status;
    int energy;
    int magnetization;
    double g_min;

    remove("test_wolff_data.tmp");
    write_text("test_wolff_input.tmp", input);
    status = system("./ising test_wolff_input.tmp > test_wolff_output.tmp");

    check(status == 0, "simulazione Wolff completa");
    check(count_numeric_lines("test_wolff_data.tmp") == 2,
          "Wolff scrive le misure richieste");
    check(last_sweep("test_wolff_data.tmp") == 4,
          "Wolff mantiene la numerazione degli aggiornamenti");
    check(first_measurement("test_wolff_data.tmp", &energy, &magnetization, &g_min) &&
          energy == -81 && abs(magnetization) == 27 && g_min < 1e-20,
          "energia, magnetizzazione e g_min corretti per il reticolo ordinato");

    remove("test_wolff_input.tmp");
    remove("test_wolff_config.tmp");
    remove("test_wolff_data.tmp");
    remove("test_wolff_output.tmp");
}

int main(void)
{
    test_lattice();
    test_reproducibility(0.0);
    test_reproducibility(0.5);
    test_gaussian();
    test_noisy_acceptance();
    test_wolff();
    test_save_load();
    test_restart();
    test_wolff_simulation();

    printf("\nTest eseguiti: %d, falliti: %d\n", tests_run, tests_failed);
    if (tests_failed != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
