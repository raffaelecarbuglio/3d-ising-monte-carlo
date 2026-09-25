# Simulatore Ising 3D in C11

Il progetto simula il modello di Ising su un reticolo cubico `L x L x L`.
Ogni sito contiene uno spin `+1` o `-1` e interagisce con i suoi sei primi
vicini. Le condizioni al bordo sono periodiche e la costante di accoppiamento
e' `J = 1`.

Gli spin possono essere aggiornati con l'algoritmo Metropolis locale oppure con
l'algoritmo di cluster Wolff. La scelta resta fissa per tutta l'esecuzione.
Il reticolo e' memorizzato con `x` come coordinata piu' rapida. PCG32 fornisce
i numeri casuali usati dai due algoritmi.

## Organizzazione dei file

- `inputs/`: file di input delle simulazioni.
- `data/`: dati e configurazioni generati, esclusi da Git salvo `.gitkeep`.
- `plots/`: grafici generati, che possono essere aggiunti a Git.

Sorgenti C, script di analisi, test e file di configurazione del progetto
restano nella root. Eseguire i comandi seguenti dalla root del repository:
anche i percorsi scritti negli input sono relativi alla directory di lavoro,
non a `inputs/`.

## Compilazione e test

Servono GCC, Make e la libreria matematica standard, normalmente gia'
disponibili su Linux:

    make
    make test

I test dell'analisi richiedono Python 3 e NumPy:

    make test-analysis

Per eliminare eseguibili, file oggetto e file temporanei:

    make clean

## Avvio

Il programma riceve il nome del file di input:

    ./ising inputs/input_example.dat

L'esempio parte da tutti gli spin uguali a `+1`. Dopo la sua conclusione si
puo' continuare la simulazione con:

    ./ising inputs/input_restart_example.dat

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
    config_file = data/example_config.dat
    data_file = data/example_data.dat

Per un restart si usa lo stesso campo `config_file`:

    L = 6
    beta = 0.22
    n_therm = 0
    n_sweeps = 20
    measure_every = 5
    seed = 67890
    start = restart
    algorithm = metropolis
    config_file = data/example_config.dat
    data_file = data/example_data.dat

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
`config_file` e nomi delle colonne. Sia Metropolis sia Wolff scrivono:

    sweep energy magnetization g_min

`energy` e `magnetization` sono i totali interi `E` e `M`: ogni legame
contribuisce una volta all'energia. `sweep` conta gli sweep di produzione;
per Wolff ogni aggiornamento di un cluster conta come uno sweep.
Non vengono piu' salvate acceptance o cluster_fraction.

Solo `g_min` richiede un numero in virgola mobile e viene scritto con `%.8g`
(otto cifre significative). `beta` resta scritto con `%.17g` nell'intestazione.
`g_min` e' lo stimatore al minimo impulso non nullo `2 pi / L`, mediato sui
tre impulsi equivalenti lungo `x`, `y` e `z`.

L'analisi ricostruisce le quantita' ridondanti, con `V = L^3`:

    m = M / V
    abs_m = abs(M) / V
    g_zero = M^2 / V
    e = -E / (3 V)

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
nel file e la numerazione prosegue. Se il file dati non esiste o e' vuoto,
viene scritta l'intestazione completa.

Il nuovo simulatore rifiuta di aggiungere righe a quattro colonne a un vecchio
file a sette colonne. Per continuare una vecchia simulazione, mantenere
`config_file` e scegliere un nuovo `data_file`.

## Compressione dei dati completati

Dopo aver terminato una simulazione e la sua analisi, si puo' comprimere il
file di misure con:

    gzip data/example_data.dat

Questo sostituisce il file con `data/example_data.dat.gz`. La compressione
e' senza perdita: il contenuto testuale resta identico. Per ispezionarlo:

    zless data/example_data.dat.gz

`analyze.py` e `blocking_plot.py` leggono direttamente anche `.dat.gz`;
`analyze_grid.sh` trova sia i file `.dat` sia i file `.dat.gz` di un batch:

    python3 analyze.py data/example_data.dat.gz --block-size 100

La compressione e' un passo esplicito, esterno al simulatore. Lasciare
non compressi i file a cui si vogliono aggiungere altre misure. Per riprendere
un file compresso nel nuovo formato, prima usare:

    gunzip data/example_data.dat.gz

## Analisi Python

Lo script `analyze.py` legge le quattro colonne numeriche (oppure il vecchio
formato a sette colonne), ignora le righe commentate e ricava `L` e `beta`
dall'intestazione. Si avvia indicando il
numero di misure per blocco:

    python3 analyze.py data/example_data.dat --block-size 100

Lo script calcola le medie di energia, magnetizzazione, magnetizzazione
assoluta, `g_zero` e `g_min`. La cumulante di Binder usa la convenzione

    U = mean(m^4) / mean(m^2)^2

La lunghezza `xi` usa il rapporto delle medie riportato sopra e `R_xi = xi/L`.
La suscettivita' magnetica usa la convenzione

    chi = L^3 mean(m^2)

Gli errori di energia, magnetizzazione assoluta, Binder, suscettivita', `xi` e
`R_xi` sono stimati con un jackknife a blocchi. Binder, suscettivita' e `xi`
vengono ricalcolati da zero in ogni campione jackknife, perche' sono funzioni
non lineari delle medie.

Ogni esecuzione di `analyze.py` salva anche, senza opzioni aggiuntive, un file
`*_blocks.txt` accanto al file delle misure. Il file contiene una riga per
blocco completo con le medie di `g_zero`, `g_min`, `m^2` e `m^4`, piu'
nell'intestazione `L`, `beta`, la dimensione del blocco e i valori finali di
`R_xi` e `U` con i loro errori. Questi dati sono sufficienti per ricostruire
insieme `R_xi` e `U` in ogni replica bootstrap, preservando automaticamente
la loro correlazione. Rieseguire l'analisi con una nuova dimensione del blocco
sovrascrive il corrispondente file `*_blocks.txt`.

Il percorso puo' essere cambiato esplicitamente con `--blocks-output`, ma il
salvataggio dei blocchi avviene sempre.

Vengono usati soltanto blocchi contigui completi. Le eventuali misure finali
che non formano un blocco completo sono escluse e il loro numero viene
stampato. Servono almeno due blocchi completi. Se l'argomento della radice di
`xi` e' negativo, lo script termina con un messaggio di errore senza alterare
il risultato.

Per osservare come gli errori stimati cambiano al crescere della dimensione
dei blocchi:

    python3 blocking_plot.py data/example_data.dat

Il grafico serve a scegliere una dimensione dei blocchi per la quale gli errori
stimati hanno raggiunto un plateau approssimativamente stabile.
Lo script crea `plots/` se manca e salva `plots/blocking_plateau.png`,
sovrascrivendo il grafico precedente.

### Grafico U in funzione di R_xi

Per aggiungere al riepilogo i risultati gia' calcolati dall'analisi:

    python3 analyze.py data/L8_beta0215.dat --block-size 100 --summary-output summary.txt
    python3 analyze.py data/L8_beta0220.dat --block-size 100 --summary-output summary.txt
    python3 analyze.py data/L16_beta0220.dat --block-size 200 --summary-output summary.txt

I nomi dei file e le dimensioni dei blocchi sono esempi: usa i tuoi file e
scegli i blocchi sulla base del plateau degli errori. Il riepilogo contiene
`L beta Rxi err_Rxi U err_U`, con intestazione commentata. Ogni esecuzione
con `--summary-output` aggiunge una riga, anche se la stessa simulazione era
gia' stata analizzata. Non vengono ricalcolate osservabili per il salvataggio.
Senza questa opzione, l'output a terminale resta quello consueto.

Per creare il grafico:

    python3 plot_u_vs_rxi.py summary.txt

Lo script raggruppa per `L`, ordina per `R_xi` e mostra errori orizzontali e
verticali, con marcatori distinti e senza curve o fit. Salva
`plots/u_vs_rxi.png` (sovrascrivendolo se esiste) e mostra la figura.
Per scegliere un altro percorso usa `--output plots/mio_grafico.png`.
Per salvare senza interfaccia grafica:

    MPLBACKEND=Agg python3 plot_u_vs_rxi.py summary.txt

### Fit globale U(R_xi, L)

`fit_scaling.py` esegue il fit

    U(R_xi, L) = sum_k b_k R_xi^k + L^(-omega) sum_k c_k R_xi^k

con `omega = 0.8295` fissato al valore di letteratura per Ising 3D. Il fit e'
lineare nei coefficienti e viene risolto con minimi quadrati pesati usando gli
errori di `U`.

Lo script legge direttamente i file `*_blocks.txt` prodotti da `analyze.py`.
Per ogni replica bootstrap ricampiona con rimpiazzo le righe di ciascuna
simulazione e usa gli stessi blocchi per ricostruire sia `R_xi` sia `U`.
La correlazione tra le due osservabili viene quindi mantenuta senza dover
calcolare una matrice di covarianza esplicita.

Un esempio e':

    MPLBACKEND=Agg python3 fit_scaling.py beta_runs_L16-24-32 beta_runs_L48-64 \
        --sizes 16 24 32 48 64 \
        --degree-main 6 \
        --degree-correction 3 \
        --bootstrap 2000 \
        --output-prefix scaling_fit

Se `--sizes` e' omesso vengono usate tutte le taglie trovate. I default per i
gradi sono 6 per la curva asintotica e 3 per la correzione. L'intervallo
predefinito e' `0.30 <= R_xi <= 1.00`. Per studiare la stabilita' del fit basta
rieseguire lo stesso comando cambiando `--sizes`, `--degree-main` oppure
`--degree-correction`: non serve modificare il codice.

Gli output sono:

- `<prefix>_summary.txt`: parametri del fit, errori bootstrap, chi2/dof e
  impostazioni usate;
- `<prefix>_curve.txt`: curva asintotica centrale e banda bootstrap puntuale
  al 68%;
- `<prefix>.png` e `<prefix>.pdf`: grafico dei dati, curva e banda.

Durante lo sviluppo si puo' usare un numero piccolo di repliche, ad esempio
`--bootstrap 200`; per il risultato finale e' opportuno aumentarlo.

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


## Rumore gaussiano in Metropolis

Aggiungere **alla fine dell'input**, dopo `data_file`:

```text
sigma = 0.5
```

Per ogni tentativo di flip si usa `delta_E + sigma * G`, con un nuovo
`G ~ N(0,1)`, nella probabilita' di accettazione. `sigma` e' la deviazione
standard del rumore aggiunto direttamente a delta E (una sola gaussiana).
Energia e magnetizzazione misurate restano quelle degli spin, senza rumore.

Se il campo manca, `sigma = 0`: stessi aggiornamenti e stessa sequenza casuale
del codice pulito. Sono ammessi solo valori finiti e non negativi;
Wolff richiede `sigma = 0`. Il valore viene scritto nel log, nell'intestazione
dati e nel commento di restart. Non si possono aggiungere dati con sigma
diverso allo stesso file; per cambiare sigma usare un nuovo `data_file`.
Come prima, il restart rilegge gli spin ma reinizializza il generatore dal seed.

Per una griglia Metropolis, `SIGMA` imposta lo stesso valore in tutti gli input
(e prevale sul template; se omesso vale 0):

```bash
SIGMA=0.5 ./run_metropolis.sh inputs/input_example.dat
```

Gli intervalli beta dello script restano quelli puliti: prima delle produzioni
con rumore vanno verificati con brevi simulazioni pilota.