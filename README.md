# Simulatore Ising 3D in C11

Il progetto simula il modello di Ising su un reticolo cubico `L x L x L`.
Ogni sito contiene uno spin `+1` o `-1` e interagisce con i suoi sei primi
vicini. Le condizioni al bordo sono periodiche e la costante di accoppiamento
e' `J = 1`.

Gli spin sono aggiornati con l'algoritmo Metropolis locale. Il reticolo e'
memorizzato con `x` come coordinata piu' rapida; ogni sweep percorre quindi
prima `z`, poi `y` e infine `x` nel ciclo piu' interno. Ogni sito viene visitato
una volta e un flip accettato modifica subito il reticolo. PCG32 fornisce i
numeri casuali usati dal criterio di accettazione.

## Compilazione e test

Servono GCC, Make e la libreria matematica standard, normalmente gia'
disponibili su Linux:

    make
    make test

Per eliminare eseguibili, file oggetto e risultati degli esempi:

    make clean

## Avvio

Il programma riceve il nome del file di input:

    ./ising input_example.dat

L'esempio parte da tutti gli spin uguali a `+1`. Dopo la sua conclusione si
puo' continuare la simulazione con:

    ./ising input_restart_example.dat

Con `start = random` la configurazione iniziale e' casuale. Una simulazione
`ordered` o `random` non sovrascrive mai un `data_file` esistente.

Con `start = restart`, gli spin e il numero di sweep gia' completati vengono
letti da `config_file`. Al termine, il checkpoint aggiornato sostituisce lo
stesso file. Il file dati viene aperto in append se esiste e creato se manca.
Il generatore PCG32 viene inizializzato con il nuovo seed indicato nell'input.

## Formato rigido dell'input

Le righe devono comparire esattamente nell'ordine documentato. Il lettore non
supporta commenti nel file di input, parametri riordinati o nomi di file con
spazi.

Per `start = ordered` o `start = random` il formato e':

    L = 6
    beta = 0.22
    n_therm = 20
    n_sweeps = 40
    measure_every = 5
    seed = 12345
    start = ordered
    config_file = example_config.dat
    data_file = example_data.dat

Per un restart si usa lo stesso campo `config_file`:

    L = 6
    beta = 0.22
    n_therm = 0
    n_sweeps = 20
    measure_every = 5
    seed = 67890
    start = restart
    config_file = example_config.dat
    data_file = example_data.dat

Significato dei parametri:

- `L`: lato del reticolo, almeno 2.
- `beta`: inverso della temperatura, non negativo.
- `n_therm`: sweep iniziali di termalizzazione, non misurati e non negativi.
- `n_sweeps`: sweep di produzione del segmento, maggiore di zero.
- `measure_every`: intervallo tra le misure, maggiore di zero.
- `seed`: intero non negativo usato per inizializzare PCG32.
- `start`: `ordered`, `random` oppure `restart`.
- `config_file`: configurazione da salvare; durante un restart viene prima
  caricata e poi sostituita con il checkpoint aggiornato.
- `data_file`: file testuale delle misure.

## File delle misure

Una nuova simulazione crea un'intestazione commentata contenente modello, `L`,
`beta`, seed, modalita' iniziale, numeri di sweep, intervallo di misura,
`config_file` e nomi delle colonne. Le righe numeriche hanno sempre questa
forma:

    sweep energy_per_spin magnetization_per_spin abs_magnetization_per_spin acceptance

`sweep` e' il numero complessivo di sweep di produzione. `acceptance` e' la
frazione dei tentativi accettati dall'ultima misura. Energia e magnetizzazione
sono divise per `L^3`; ogni legame contribuisce una volta all'energia.

Un restart aggiunge soltanto un breve blocco commentato con lo sweep iniziale,
i parametri del nuovo segmento e `config_file`. Non ripete
l'intestazione completa o i nomi delle colonne. Le misure precedenti restano
nel file e la numerazione prosegue. In Python si possono ignorare tutte le
righe commentate, per esempio:

    data = pandas.read_csv("example_data.dat", sep=r"\s+", comment="#",
                           header=None)

## File di configurazione

La configurazione e' un file testuale con questa struttura:

    L 3
    production_sweeps 20
    spins
     1 -1  1
    ...

Contiene `L`, il numero totale di sweep di produzione completati e tutti gli
spin. Durante il caricamento il programma controlla che `L` coincida e che ogni
spin sia `+1` o `-1`.

Il contenuto viene scritto direttamente nel percorso indicato da `config_file`.
Il programma segnala eventuali errori di apertura, scrittura o chiusura del file.

I percorsi sono relativi alla directory dalla quale si avvia il programma. Il
progetto e' seriale e non usa librerie esterne.
