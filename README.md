# Simulatore Ising 3D in C11

Il progetto simula il modello di Ising su un reticolo cubico `L x L x L`.
Ogni sito contiene uno spin `+1` o `-1` e interagisce con i suoi sei primi
vicini. Le condizioni al bordo sono periodiche e la costante di accoppiamento
e' `J = 1`.

Gli spin possono essere aggiornati con l'algoritmo Metropolis locale oppure con
l'algoritmo di cluster Wolff. La scelta resta fissa per tutta l'esecuzione.
Il reticolo e' memorizzato con `x` come coordinata piu' rapida. PCG32 fornisce
i numeri casuali usati dai due algoritmi.

## Compilazione e test

Servono GCC, Make e la libreria matematica standard, normalmente gia'
disponibili su Linux:

    make
    make test

I test dell'analisi richiedono Python 3 e NumPy:

    make test-analysis

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
    algorithm = metropolis
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
    algorithm = metropolis
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
- `algorithm`: `metropolis` oppure `wolff`; resta fisso durante l'esecuzione.
- `config_file`: configurazione da salvare; durante un restart viene prima
  caricata e poi sostituita con il checkpoint aggiornato.
- `data_file`: file testuale delle misure.

## File delle misure

Una nuova simulazione crea un'intestazione commentata contenente modello, `L`,
`beta`, seed, modalita' iniziale, numeri di sweep, intervallo di misura,
`config_file` e nomi delle colonne. Con Metropolis le righe numeriche hanno
questa forma:

    sweep energy_per_spin magnetization_per_spin abs_magnetization_per_spin acceptance g_zero g_min

`sweep` e' il numero complessivo di sweep di produzione. `acceptance` e' la
frazione dei tentativi accettati dall'ultima misura. Energia e magnetizzazione
sono divise per `L^3`; ogni legame contribuisce una volta all'energia.

Con Wolff ogni aggiornamento costruisce e inverte un cluster e viene contato
come uno sweep. La quinta colonna si chiama `cluster_fraction` e contiene la
dimensione media dei cluster dall'ultima misura, divisa per il numero di spin:

    sweep energy_per_spin magnetization_per_spin abs_magnetization_per_spin cluster_fraction g_zero g_min

Le ultime due colonne sono gli stimatori della funzione di correlazione in
spazio degli impulsi. Per ogni configurazione misurata il programma calcola

    g_zero = M^2 / L^3

e `g_min` al minimo impulso non nullo `2 pi / L`. Per ridurre il rumore,
`g_min` e' la media sui tre impulsi equivalenti lungo `x`, `y` e `z`.

La lunghezza di correlazione del secondo momento deve essere calcolata usando
le medie sulle configurazioni:

    xi = sqrt(mean(g_zero) / mean(g_min) - 1) / (2 sin(pi / L))

Non bisogna calcolare `xi` separatamente per ogni riga. Per stimarne l'errore
occorre dividere i dati in blocchi e ricalcolare il rapporto all'interno dei
campioni jackknife. Se l'argomento della radice risulta negativo, la stima va
segnalata come non valida e non sostituita artificialmente con zero.

Un restart aggiunge soltanto un breve blocco commentato con lo sweep iniziale,
i parametri del nuovo segmento e `config_file`. Non ripete
l'intestazione completa o i nomi delle colonne. Le misure precedenti restano
nel file e la numerazione prosegue.

## Analisi Python

Lo script `analyze.py` legge le sette colonne numeriche, ignora tutte le righe
commentate e ricava `L` e `beta` dall'intestazione. Si avvia indicando il
numero di misure per blocco:

    python3 analyze.py example_data.dat --block-size 100

Lo script calcola le medie di energia, magnetizzazione, magnetizzazione
assoluta, `g_zero` e `g_min`. La cumulante di Binder usa la convenzione

    U = mean(m^4) / mean(m^2)^2

La lunghezza `xi` usa il rapporto delle medie riportato sopra e `R_xi = xi/L`.
La suscettivita' magnetica usa la convenzione

    chi' = beta L^3 (mean(m^2) - mean(|m|)^2)

Gli errori di energia, magnetizzazione assoluta, Binder, suscettivita', `xi` e
`R_xi` sono stimati con un jackknife a blocchi. Binder, suscettivita' e `xi`
vengono ricalcolati da zero in ogni campione jackknife, perche' sono funzioni
non lineari delle medie.

Vengono usati soltanto blocchi contigui completi. Le eventuali misure finali
che non formano un blocco completo sono escluse e il loro numero viene
stampato. Servono almeno due blocchi completi. Se l'argomento della radice di
`xi` e' negativo, lo script termina con un messaggio di errore senza alterare
il risultato.

Per osservare come gli errori stimati cambiano al crescere della dimensione
dei blocchi:

    python blocking_plot.py data.dat

Il grafico serve a scegliere una dimensione dei blocchi per la quale gli errori
stimati hanno raggiunto un plateau approssimativamente stabile.

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
