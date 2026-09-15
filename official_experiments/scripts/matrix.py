"""LA pipeline sperimentale. Ex ``temporal_matrix.py``, rinominata il 21/08/2026.

Questo e' lo script definitivo: ``final_matrix.py`` e ``suffix_on_t12.py``
restano solo finche' i loro CSV congelati servono, poi vanno tolti. Motivo della
scelta, in breve:

* misura **entrambi** i compiti (next activity e suffisso) nella stessa run,
  mentre ``final_matrix.py`` ne misurava uno e ``suffix_on_t12.py`` recuperava
  l'altro ricaricando i checkpoint;
* proprio per questo non ricostruisce mai gli artefatti a posteriori. E'
  l'unica difesa contro il fatto che ``pm4py.discover_petri_net_inductive`` NON
  e' riproducibile fra processi (su BPIC20 da' 21 posti quasi sempre e 22 ogni
  tanto): qui la rete e' una sola per run, e addestramento e valutazione la
  condividono.

Il protocollo non e' cablato: split, vocabolario e sorgente della conoscenza
arrivano dal config (``split_strategy``, ``vocabulary_scope``,
``knowledge_source``, vedi ``nspm/config.py`` e ``nspm/data/preparation.py``).
Con ``temporal_protocol()`` si ottiene il protocollo B, descritto qui sotto; con
i default di ``ExperimentConfig`` si otterrebbe lo split del protocollo A.
L'unica cosa ancora cablata e' il modello di rumore: qui ``corrupt_traces``
(eventi), in A ``corrupt_targets`` (target). Diventera' il quarto parametro.

Cosa cambia rispetto alla T12 (protocollo A)
--------------------------------------------
1. **Split temporale su tutti i case**: ordinati per timestamp del primo evento
   e tagliati per posizione, senza shuffle e senza seed. E' la semantica di
   ``nirdizati_light.log.common.split_train_val_test`` con ``shuffle=False``
   (quella libreria non ha uno split temporale suo: taglia per indice, e la
   temporalita' viene dal pre-ordinamento). Default 70/15/15, configurabile.
2. **Nessun filtro**: nessuna deduplica di varianti, nessuna traccia scartata,
   duplicati mantenuti. Il vocabolario si costruisce su TUTTE le partizioni,
   che e' cio' che permette di non scartare i case con attivita' che compaiono
   solo tardi nel tempo.
3. **Background knowledge dal TEST**: DFA, Petri net e vincoli di precedenza
   sono estratti dalla partizione di test.
4. **Rumore a livello di EVENTO**: si sostituisce l'etichetta di attivita' di
   una frazione degli eventi di train con un'altra etichetta del vocabolario,
   come in Mezini et al. Sec. 4.1. La T12 corrompeva solo i target degli
   esempi, lasciando le tracce intatte: li' il training set restava conforme al
   100% a ogni livello di rumore. Qui la corruzione danneggia la traccia
   stessa, quindi si propaga ai prefissi successivi, al replay sulla rete di
   Petri e alla conformita' del log. Ogni riga porta ``train_compliance``, la
   frazione di tracce di train ancora conformi dopo l'iniezione (la Table 2 del
   paper), senza la quale l'asse rumore non e' interpretabile.

   Conseguenza pratica: i marking vanno rigiocati a ogni livello di rumore, per
   cui i log si costruiscono una volta per (noise, seed) invece che una volta
   per dataset. Costa ~12-15 min per dataset, ~40 min in totale.

Cosa misura ogni cella
----------------------
Due task sullo STESSO modello addestrato, in sequenza:

1. **next activity** — un passo dal prefisso di ground truth: accuracy,
   macro F1, top-3, violation rate, forbidden mass.
2. **suffix prediction** — il modello prosegue e rigenera la traccia dal
   proprio output, quindi gli errori si accumulano. Metrica principale la
   **Damerau-Levenshtein normalizzata** (``dl_similarity``, 1 = traccia
   identica), piu' accuracy posizionale, exact match e due tassi di
   conformita': violazioni directly-follows per passo e violazioni di
   precedenza a livello di traccia.

Prefissi: meta' della lunghezza mediana delle tracce di test, +1, +2.

Limiti noti, da dichiarare in tesi
----------------------------------
* La generazione e' **condizionata sulla lunghezza**: i modelli non hanno un
  token di fine traccia, quindi si generano esattamente ``len(suffisso vero)``
  passi. La DL risulta piu' alta di quella di un vero suffix predictor e NON e'
  confrontabile con i valori pubblicati; resta valida come confronto fra
  varianti, che vedono tutte lo stesso vantaggio.
* Con la background knowledge dal test, forbidden mass e tassi di violazione
  sono misurati contro artefatti estratti dalle stesse tracce su cui si valuta:
  sono ottimistici per costruzione.

Uso
---
  python official_experiments/scripts/matrix.py
  python official_experiments/scripts/matrix.py --datasets Sepsis_Case --archs gru
  python official_experiments/scripts/matrix.py --variants baseline checker --noises 0.0
  python official_experiments/scripts/matrix.py --val-fraction 0.10 --test-fraction 0.20

I risultati sono appesi riga per riga a ``runs/temporal_matrix/results.csv`` con
RESUME: le celle gia' presenti vengono saltate, quindi lo script si puo'
lanciare a fette.
"""

import argparse
import csv
import random
import sys
import time
import torch
from dataclasses import replace
from pathlib import Path
from statistics import mean, stdev
from typing import get_args

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nspm.config import DataConfig, ExperimentConfig, temporal_protocol
from nspm.data.loader import read_log
from nspm.data.preparation import (
    PrefixLog,
    TraceUtils,
    build_splits,
    build_vocabulary,
    knowledge_traces,
    unseen_activities,
)
from nspm.process import artifact_store
from nspm.process.petrinet import PetriNet
from nspm.process.automaton import ProcessDFA
from nspm.process.reachability import ReachabilityAutomaton
from nspm.process.ltl_constraints import compliance_ratio, mine_precedence_constraints
from nspm.learning.axel_losses import GlobalLogicLoss, LocalLogicLoss, TensorDFA
from nspm.learning.models import GRAPH_SUFFIXES, ModelKind
from nspm.learning.training import train_model, resolve_device
from nspm.learning.logic import build_allowed_mask, build_state_mask
from nspm.learning.evaluation import evaluate_model
from nspm.learning.trace_prediction import evaluate_suffix_prediction

DATASETS = ("Sepsis_Case", "BPIC_2013_incidents", "BPIC_2020_DomesticDeclarations")
#: Aggiunti il 21/08/2026. Ammessi ma NON nel default: la griglia congelata e'
#: 1260 righe sui tre log sopra, e un lancio senza argomenti deve continuare a
#: riprodurre quella. Questi due si chiedono per nome.
EXTRA_DATASETS = ("BPI_Challenge_2012", "BPI_Challenge_2015_Municipality")
ALL_DATASETS = DATASETS + EXTRA_DATASETS
ARCHS = ("gru", "lstm")
VARIANTS = ("baseline", "checker", "marking", "gnn", "seq")
CROSS_VARIANTS = ("checker_marking", "checker_gnn", "checker_seq")
STEP6_VARIANTS = ("grnn",)
#: Checker guidato dall'automa della RETE DI PETRI invece che dal DFA empirico
#: (vedi ``MASK_SPEC``). Fuori dal default: cartella propria, unione in analisi.
NETDFA_VARIANTS = ("checker_net", "checker_net_state")
#: I due metodi di Axel, aggiunti il 24/08/2026. Fuori dal default per la stessa
#: ragione degli altri: un lancio senza argomenti deve continuare a riprodurre
#: la griglia congelata.
AXEL_VARIANTS = ("lll", "gll")
ALL_VARIANTS = (VARIANTS + CROSS_VARIANTS + STEP6_VARIANTS + NETDFA_VARIANTS
                + AXEL_VARIANTS)
NOISES = (0.0, 0.25, 0.5)
SEEDS = tuple(range(10))
LOGIC_WEIGHT = 0.5  # pre-registrato: lo stesso della T12 e del benchmark di corruzione

# variante -> (suffisso kind, use_logic, famiglia di dati, automa della loss).
# Le assi sono indipendenti: il suffisso sceglie l'encoder simbolico (feature),
# il booleano accende il termine di loss, l'ultimo campo sceglie da quale automa
# arriva la maschera che la loss penalizza.
VARIANT_SPEC = {
    "baseline": ("", False, "plain", "dfa"),
    "checker": ("", True, "plain", "dfa"),
    "marking": ("_marking", False, "marked", "dfa"),
    "gnn": ("_gnn", False, "marked", "dfa"),
    "seq": ("_seq", False, "sequences", "dfa"),
    "checker_marking": ("_marking", True, "marked", "dfa"),
    "checker_gnn": ("_gnn", True, "marked", "dfa"),
    "checker_seq": ("_seq", True, "sequences", "dfa"),
    "grnn": ("_grnn", False, "sequences", "dfa"),
    "checker_net": ("", True, "plain", "net"),
    "checker_net_state": ("", True, "stated", "net_state"),
    # I due metodi di Axel (nesy-suffix-prediction-dfa). Stesso tronco della
    # baseline e stessa struttura di ``checker_net``, cioe' l'automa estratto
    # dalla rete di Petri: cambia solo COME il vincolo entra nella loss, che e'
    # l'unica cosa che l'esperimento vuole confrontare. Il termine di paragone
    # diretto e' ``checker_net``, non ``checker``.
    "lll": ("", True, "plain", "axel_local"),
    "gll": ("", True, "plain", "axel_global"),
}

#: Iperparametri delle due loss di Axel. Lui fa uno sweep su dieci valori di
#: alpha per quindici run; qui la griglia ha una seed sola, quindi i due valori
#: si scelgono a mano e si dichiarano.
#:
#: In entrambe le loss ``alpha`` pesa la supervisione e ``1 - alpha`` la logica,
#: quindi i due metodi girano su miscele opposte: la locale al 75% di logica, la
#: globale al 75% di cross-entropy. Un valore per metodo, non uno solo: la
#: penalita' locale e' una massa di probabilita' a un passo, quella globale il
#: ``-log`` dell'accettazione di un rollout intero, e non stanno sulla stessa
#: scala.
#:
#: ``AXEL_SAMPLES`` e' il valore pieno di Axel. Il rollout costa ``horizon``
#: passi sequenziali su un batch replicato altrettante volte, quindi su una
#: scheda da 16 GB dieci campioni non ci stavano; la griglia gira su A100 da
#: 80 GB, dove ci stanno. Va tenuto UGUALE su tutte le celle: con un numero di
#: campioni diverso la stima dell'accettazione ha una varianza diversa, e le
#: celle non sarebbero piu' confrontabili fra loro.
AXEL_ALPHA_LOCAL = 0.25
AXEL_ALPHA_GLOBAL = 0.75
AXEL_TEMPERATURE = 0.5
AXEL_SAMPLES = 10

#: Taglia del tronco ricorrente. Sono i numeri di Mezini et al. (2026): due
#: strati da cento unita', ottimizzatore Adam, batch da 64. La baseline compatta
#: del repo era uno strato da 64, che per una GRU basta e per una LSTM no --
#: l'intero vantaggio della cella di memoria sta nel poterla impilare.
#:
#: Vale per TUTTE E NOVE le varianti, non solo per le due di Axel. Il confronto
#: cella per cella regge solo se il tronco e' lo stesso: se i due metodi di Axel
#: girassero sul loro tronco e i sette nostri sul nostro, la matrice misurerebbe
#: la differenza fra le architetture invece che fra i canali della conoscenza.
RECURRENT_HIDDEN = 100
RECURRENT_LAYERS = 2

#: Tetto delle epoche. Dodici erano poche: sulla griglia GRU un terzo delle
#: celle si fermava a undici o dodici, cioe' per fine budget e non per
#: convergenza, e una rete piu' grande ne chiede di piu', non di meno. Mezini et
#: al. arrivano a 590-1607 epoche.
#:
#: Il tetto non e' il costo: l'early stopping resta quello di prima, quindi le
#: celle che gia' convergevano in quattro epoche continuano a fermarsi li'.
#: Alzarlo paga solo dove serviva.
MAX_EPOCHS = 60

#: I due automi da cui puo' arrivare la maschera della loss. ``dfa`` e' il
#: directly-follows empirico contato sulle tracce di knowledge; ``net`` e'
#: l'automa di raggiungibilita' della rete di Petri gia' scoperta dalle stesse
#: tracce, proiettato su directly-follows. Stessa informazione sorgente, canale
#: diverso: la rete generalizza, quindi la sua maschera e' piu' larga.
#:
#: ``net_state`` e' la stessa rete SENZA proiezione: una riga per stato
#: dell'automa invece che per attivita', indicizzata replaying il prefisso. E'
#: l'unico modo in cui la memoria dei marking arriva fino alla loss.
#:
#: Tutte le varianti si VALUTANO su ``dfa`` -- comprese le metriche del
#: suffisso, che usano ``art["automaton"]``. Se ogni variante misurasse la
#: conformita' contro il proprio automa le colonne non sarebbero piu'
#: confrontabili fra righe.
MASK_SPEC = ("dfa", "net", "net_state")
EVAL_MASK = "dfa"

#: Chi ha bisogno della struttura estratta dalla rete di Petri.
#:
#: Il criterio e' quello del prof: due metodi sono confrontabili solo se vedono
#: LO STESSO oggetto simbolico. Chi usa la rete usa la stessa rete, chi usa il
#: DFA usa lo stesso DFA -- altrimenti la differenza fra due celle misura da
#: quale struttura viene la conoscenza invece di come quella conoscenza entra
#: nella loss, che e' l'unica cosa che l'esperimento vuole isolare.
#:
#: ``lll`` e ``gll`` stanno di qua insieme a marking, gnn, seq e ai due
#: ``checker_net``: la loro conoscenza e' la rete, proiettata su directly-follows
#: per la locale e tensorizzata per la globale. Il prezzo e' che adesso
#: dipendono dall'automa di raggiungibilita', quindi su una rete con troppe
#: transizioni silenti cadono anche loro.
NET_DFA_USERS = frozenset({"net", "axel_local", "axel_global"})
#: Chi fa costruire l'automa. La maschera per stato non passa dalla proiezione,
#: quindi le chiede l'automa e basta.
AUTOMATON_USERS = NET_DFA_USERS | {"net_state"}

CSV_FIELDS = (
    "dataset", "arch", "variant", "noise", "seed",
    # --- task 1: next activity, un passo dal prefisso vero.
    #     Ogni metrica di conformita' esiste in DUE copie: la stessa definizione,
    #     calcolata contro due oggetti diversi. Senza suffisso e' il
    #     directly-follows empirico, minato dalle tracce; con ``_net`` e' l'automa
    #     di raggiungibilita' proiettato dalla rete di Petri. Entrambe valgono per
    #     tutte e nove le varianti, baseline compresa: il metro deve essere lo
    #     stesso su ogni riga, altrimenti le righe non si confrontano.
    #
    #     Non sono ridondanti, ed e' un fatto misurato e non una precauzione: su
    #     Sepsis la rete ammette 199 celle e il DFA empirico 159, ma le due
    #     maschere NON sono annidate -- l'inductive miner con
    #     ``noise_threshold=0.2`` filtra il comportamento infrequente, quindi la
    #     rete vieta anche coppie che nel log ci sono davvero. Un metodo
    #     addestrato contro la rete misurato col metro empirico si vede contare
    #     come violazioni proprio le generalizzazioni che gli abbiamo chiesto di
    #     fare, e viceversa.
    #
    #     Le colonne ``_net`` restano vuote dove l'automa non si costruisce (BPIC15).
    #     Le altre ci sono sempre, ed e' l'unico motivo per cui quel dataset resta
    #     misurabile.
    "accuracy", "macro_f1", "top3", "violation_rate", "forbidden",
    "violation_rate_net", "forbidden_net",
    # --- task 2: suffix prediction, il modello rigenera dal proprio output.
    #     Stessa coppia di metri sulle stesse tracce generate: contarle due volte
    #     e' un giro di confronti su stringhe, non una seconda inferenza.
    "dl_similarity", "suffix_accuracy", "exact_match",
    "suffix_dfa_violation", "suffix_dfa_violation_net",
    "suffix_precedence_violation",
    # Fitness per token replay della traccia COMPLETA (prefisso reale + suffisso
    # generato) contro la rete di Petri: l'unica conformita' misurata sulla
    # traccia intera invece che sulla singola transizione.
    "suffix_net_fitness",
    "n_suffix", "prefix_lengths",
    # --- rumore, costo e provenienza. ``corrupted`` conta gli EVENTI la cui
    #     etichetta e' stata sostituita; ``train_compliance`` e' la frazione di
    #     tracce di train ancora conformi ai vincoli dopo l'iniezione -- la
    #     Table 2 del paper, senza la quale l'asse rumore non e' interpretabile.
    "best_epoch", "corrupted", "train_compliance", "secs_per_epoch",
    # --- costo. ``secs_train`` e' il muro dell'addestramento, ``secs_per_epoch``
    #     lo stesso diviso per le epoche fatte. Gli altri quattro sono la parte
    #     simbolica, che la rete neurale non paga: la scoperta della rete di
    #     Petri, il DFA con la sua tensorizzazione, l'automa di raggiungibilita'
    #     e il token replay delle partizioni held-out. ``secs_artifacts`` li
    #     contiene tutti piu' la lettura del log.
    "secs_train", "secs_petrinet", "secs_dfa", "secs_automaton",
    "secs_markings", "secs_artifacts",
    # Su quale scheda ha girato la cella. Senza questa colonna i secondi non si
    # possono leggere: un job e' un dataset ed e' un nodo, quindi le nove
    # varianti di uno stesso dataset sono sempre confrontabili fra loro, ma due
    # dataset finiti su schede diverse no.
    "gpu",
    # --- riproducibilita'. ``knowledge_key`` e' l'impronta delle tracce da cui
    #     si e' minato, ``net_fingerprint`` quella della rete che ne e' uscita.
    #     Servono a poter VERIFICARE, non solo a sperare: due righe con la stessa
    #     coppia hanno visto lo stesso oggetto simbolico, e una ricostruzione
    #     futura si confronta con quello che c'e' scritto qui. Senza, un cambio
    #     di rete e' invisibile finche' non lo si legge nei risultati.
    #     ``artifacts_cached`` dice se in quella run gli oggetti sono stati
    #     riletti da disco: quando vale 1 i secondi della parte simbolica sono
    #     quelli della costruzione originale, non di questa run.
    "knowledge_key", "net_fingerprint", "artifacts_cached",
    "split", "knowledge", "vocab_scope", "train_frac", "val_frac", "test_frac",
)


def find_log(dataset: str) -> Path:
    folder = ROOT / "datasets" / dataset
    # ``*.xes.gz`` sta in lista perche' i log pubblici arrivano spesso
    # compressi; pm4py li legge senza scompattarli.
    for pattern in ("*.xes", "*.xes.gz", "*.csv", "*.csv.gz"):
        try:
            return next(folder.glob(pattern))
        except StopIteration:
            continue
    raise FileNotFoundError(f"No .xes o .csv log in {folder}")


def build_dataset_artifacts(dataset: str, config, families: set[str],
                            masks_needed: set[str] = frozenset({"dfa"}),
                            cache_dir: Path | None = None) -> dict:
    """Tutto cio' che deriva da un log, costruito una volta sola per dataset.

    Con ``cache_dir`` gli oggetti minati vengono letti da disco se ci sono e
    scritti se mancano (vedi ``nspm.process.artifact_store``). Con ``None`` si
    mina sempre, che e' il comportamento di prima.
    """
    print(f"[{dataset}] caricamento e costruzione artefatti...", flush=True)
    # I tempi della parte simbolica, cronometrati pezzo per pezzo e riportati nel
    # CSV accanto a quelli di addestramento. Servono a dire quanto costa la
    # conoscenza, che e' una domanda diversa da quanto costa la rete neurale: sotto
    # i protocolli a rumore sugli eventi questi secondi si ripagano a ogni livello
    # di rumore, perche' le tracce cambiano e la conoscenza va riscoperta.
    timings = {"petrinet": 0.0, "dfa": 0.0, "automaton": 0.0,
               "markings": 0.0, "artifacts": 0.0}
    build_started = time.perf_counter()

    events = read_log(find_log(dataset))
    splits = build_splits(events, config)

    # Sotto ``vocabulary_scope="train"`` l'alfabeto viene dal solo blocco di
    # training, quindi i case held-out con attivita' mai viste non sono ne'
    # predicibili ne' valutabili e vanno scartati -- e' la convenzione del
    # protocollo A (``drop_unseen`` in final_matrix.py). Sotto ``"all"`` non si
    # scarta niente e l'insieme e' vuoto per costruzione.
    if config.data.vocabulary_scope == "train":
        known = {a for trace in splits.train.values() for a in trace}

        def keep(partition):
            kept = {cid: t for cid, t in partition.items()
                    if all(a in known for a in t)}
            return kept, len(partition) - len(kept)

        validation, dropped_val = keep(splits.validation)
        test, dropped_test = keep(splits.test)
        if dropped_val or dropped_test:
            print(f"[{dataset}] scartati {dropped_val} case di validation / "
                  f"{dropped_test} di test con attivita' mai viste nel train",
                  flush=True)
        splits = replace(splits, validation=validation, test=test)

    vocab = build_vocabulary(splits, config)
    unseen = unseen_activities(splits, vocab)
    if unseen:
        raise ValueError(
            f"[{dataset}] attivita' non rappresentabili nelle partizioni held-out: "
            f"{sorted(unseen)}. Il protocollo temporale non scarta tracce, quindi "
            f"il vocabolario deve coprirle: controlla data.vocabulary_scope."
        )

    # Background knowledge dalla partizione indicata da ``knowledge_source``
    # (test, in questo protocollo).
    knowledge = knowledge_traces(splits, config)

    # Gli oggetti minati vengono da disco quando ci sono gia'. La chiave e'
    # l'impronta delle tracce di knowledge, quindi B e C dello stesso dataset
    # non si toccano: minano da partizioni diverse e hanno chiavi diverse.
    # Serve a poter RICARICARE una run -- rivalutare i checkpoint, rifare le
    # figure, riprodurre un numero -- cosa che senza gli oggetti salvati non e'
    # possibile, perche' l'inductive miner genera nomi nuovi a ogni chiamata e
    # i vettori di marking dipendono dall'ordine dei posti.
    store_key = artifact_store.knowledge_fingerprint(knowledge, noise_threshold=0.2)
    store_file = artifact_store.cache_path(cache_dir, dataset, store_key) \
        if cache_dir is not None else None
    cached = artifact_store.load(store_file) if store_file is not None else {}
    fresh = False

    started = time.perf_counter()
    if "petrinet" in cached:
        petrinet = cached["petrinet"]
        # I secondi sono quelli della costruzione VERA, riletti dalla cache: la
        # colonna deve dire quanto costa scoprire quella rete, non quanto costa
        # aprire un file. Se riportasse zero, il costo della parte simbolica --
        # che e' una delle metriche richieste -- sparirebbe al secondo lancio.
        timings["petrinet"] = cached.get("secs_petrinet", 0.0)
    else:
        petrinet = PetriNet.from_traces(knowledge)
        timings["petrinet"] = time.perf_counter() - started
        fresh = True

    started = time.perf_counter()
    if "automaton" in cached:
        automaton = cached["automaton"]
        timings["dfa"] = cached.get("secs_dfa", 0.0)
        mask = build_allowed_mask(automaton, vocab)
    else:
        automaton = ProcessDFA.from_traces(knowledge.values())
        mask = build_allowed_mask(automaton, vocab)
        timings["dfa"] = time.perf_counter() - started
        fresh = True
    masks = {"dfa": mask}
    reachability = cached.get("reachability")
    net_dfa = state_mask = None
    started = time.perf_counter()
    if reachability is not None and masks_needed & AUTOMATON_USERS:
        timings["automaton"] = cached.get("secs_automaton", 0.0)
        print(f"[{dataset}] automa della rete (da cache): "
              f"{reachability.state_count} stati "
              f"({len(reachability.accepting)} accettanti)", flush=True)
    elif masks_needed & AUTOMATON_USERS:
        # Stessa rete delle varianti marking/gnn/seq -- stesso oggetto, non una
        # ricostruzione. ``alphabet`` tiene dentro le attivita' potate
        # dall'inductive miner: sulla loro posizione la rete non dice niente, e
        # non dire niente non significa vietarle.
        reachability = ReachabilityAutomaton.from_process_net(
            petrinet, alphabet=vocab.activities
        )
        timings["automaton"] = time.perf_counter() - started
        fresh = True
        print(f"[{dataset}] automa della rete: {reachability.state_count} stati "
              f"({len(reachability.accepting)} accettanti)", flush=True)
    if masks_needed & NET_DFA_USERS:
        # Proiezione su directly-follows: perde la memoria dei marking, ma e'
        # l'unica forma che una maschera sull'ultimo token accetta.
        net_dfa = reachability.to_process_dfa()
        masks["net"] = build_allowed_mask(net_dfa, vocab)
        print(f"[{dataset}] proiettato: {net_dfa.transition_count} transizioni | "
              f"celle ammesse: net {int(masks['net'].sum())} vs empirico "
              f"{int(mask.sum())} su {mask.numel()}", flush=True)
    if "net_state" in masks_needed:
        # Nessuna proiezione: una riga per stato dell'automa, la memoria resta.
        state_mask = build_state_mask(reachability, vocab)
        print(f"[{dataset}] maschera per stato: {tuple(state_mask.shape)} "
              f"({int(state_mask.sum())} celle ammesse)", flush=True)
    # I vincoli di precedenza danno la conformita' a livello di TRACCIA: il
    # dfa_violation_rate conta i singoli passi directly-follows, questo dice se
    # la traccia generata nel suo insieme rompe una regola di ordinamento.
    #
    # Anche questi vanno salvati: ``mine_precedence_constraints`` ha una
    # non-determinismo noto fra processi, quindi due run possono minare vincoli
    # diversi dalle stesse tracce e la colonna della violazione non sarebbe
    # confrontabile fra loro.
    if "constraints" in cached:
        constraints = cached["constraints"]
    else:
        constraints = mine_precedence_constraints(
            knowledge.values(),
            min_support=max(3, len(knowledge) // 10),
            min_confidence=0.95,
            max_constraints=15,
        )
        fresh = True

    # Si salva solo se qualcosa e' stato minato adesso. La cache cresce per
    # aggiunta: un lancio senza automa scrive rete, DFA e vincoli, e il lancio
    # successivo che l'automa lo chiede lo aggiunge senza rifare il resto.
    if store_file is not None and fresh:
        artifact_store.save(store_file, {
            "dataset": dataset, "knowledge_key": store_key,
            "petrinet": petrinet, "automaton": automaton,
            "reachability": reachability, "constraints": constraints,
            "secs_petrinet": timings["petrinet"], "secs_dfa": timings["dfa"],
            "secs_automaton": timings["automaton"],
        })

    # Prefissi: meta' della lunghezza mediana del test, +1, +2 -- calcolata sul
    # test perche' e' li' che la suffix prediction viene valutata.
    lengths = sorted(len(t) for t in splits.test.values())
    median_length = lengths[len(lengths) // 2] if lengths else 0
    prefix_lengths = tuple(
        k for k in (median_length // 2, median_length // 2 + 1, median_length // 2 + 2)
        if k >= 1
    )

    # La loss globale di Axel ha bisogno dell'automa in forma tensoriale, per
    # poterlo attraversare in modo differenziabile. E' la STESSA proiezione che
    # legge ``checker_net``, solo scritta come matrici di transizione: stessa
    # rete, stesso automa, stessa proiezione su directly-follows.
    tensor_dfa = gll_horizon = None
    started = time.perf_counter()
    if "axel_global" in masks_needed:
        tensor_dfa = TensorDFA.from_process_dfa(
            net_dfa, vocab.activities, resolve_device(config.training.device)
        )
        # Quanto lontano prosegue il rollout. Axel arriva alla fine della traccia
        # piu' lunga piu' un margine; qui i prefissi di addestramento partono
        # intorno a meta' della lunghezza mediana, quindi la coda da generare e'
        # dell'ordine dell'altra meta'. Tenerlo legato alla stessa scala del
        # compito suffisso evita di pagare passi che nessuna traccia userebbe.
        gll_horizon = max(1, median_length // 2 + 2)
        print(f"[{dataset}] automa tensoriale: {tensor_dfa.n_states} stati x "
              f"{tensor_dfa.n_actions} azioni | rollout {gll_horizon} passi x "
              f"{AXEL_SAMPLES} campioni", flush=True)
    # La tensorizzazione e' lo stesso automa del checker riscritto in matrici,
    # quindi il suo costo sta con quello del DFA e non con quello della rete.
    timings["dfa"] += time.perf_counter() - started

    def logs(traces_map):
        base = PrefixLog.from_traces(traces_map, vocab)
        out = {"plain": base}
        if "marked" in families:
            out["marked"] = base.with_markings(petrinet)
        if "sequences" in families:
            out["sequences"] = base.with_markings(petrinet).with_marking_sequences(petrinet)
        if "stated" in families:
            # Stesso modello della baseline: lo stato non e' un input, serve
            # solo a indicizzare la maschera della loss.
            out["stated"] = base.with_automaton_states(reachability)
        return out

    batch = config.data.batch_size
    started = time.perf_counter()
    eval_loaders = {
        split_name: {family: log.data_loader(batch, shuffle=False)
                     for family, log in logs(traces_map).items()}
        for split_name, traces_map in (("val", splits.validation), ("test", splits.test))
    }
    # Il token replay di validation e test. Quello del train non e' qui: ``logs``
    # e' una chiusura richiamata a ogni cella, perche' il rumore cambia le tracce
    # di addestramento e i marking vanno rigiocati. Questa colonna misura quindi
    # il replay della parte held-out, che e' l'unica costruita una volta sola.
    timings["markings"] = time.perf_counter() - started

    variants_train = len({tuple(t) for t in splits.train.values()})
    clean_ratio = compliance_ratio(constraints, splits.train.values())
    print(f"[{dataset}] {len(vocab.activities)} attivita, {len(petrinet.places)} posti\n"
          f"[{dataset}] split {config.data.split_strategy} "
          f"{len(splits.train)}/{len(splits.validation)}/"
          f"{len(splits.test)} case (duplicati tenuti: {variants_train} varianti distinte "
          f"su {len(splits.train)} tracce di train)\n"
          f"[{dataset}] knowledge da {config.data.knowledge_source}: "
          f"{len(knowledge)} tracce -> DFA {len(automaton.states)} stati/"
          f"{automaton.transition_count} transizioni, {len(constraints)} vincoli\n"
          f"[{dataset}] conformita' del train pulito: {clean_ratio:.3f}\n"
          f"[{dataset}] suffix: lunghezza mediana test {median_length}, "
          f"prefissi {prefix_lengths}", flush=True)
    timings["artifacts"] = time.perf_counter() - build_started
    print(f"[{dataset}] costruzione: rete {timings['petrinet']:.1f}s | "
          f"dfa {timings['dfa']:.1f}s | automa {timings['automaton']:.1f}s | "
          f"marking held-out {timings['markings']:.1f}s | "
          f"totale {timings['artifacts']:.1f}s", flush=True)
    return {
        "timings": timings,
        "vocab": vocab, "petrinet": petrinet, "mask": mask, "masks": masks,
        "reachability": reachability, "state_mask": state_mask,
        # Le due impronte che finiscono nel CSV. ``knowledge_key`` dice da quali
        # tracce si e' minato, ``net_fingerprint`` che rete ne e' uscita: due
        # righe con le stesse impronte hanno visto lo stesso oggetto, due righe
        # con impronte diverse no. E' cio' che permette di accorgersi che una
        # rete e' cambiata, invece di scoprirlo dai numeri.
        "knowledge_key": store_key,
        "net_fingerprint": artifact_store.net_fingerprint(petrinet),
        "cached": not fresh,
        # L'automa della rete proiettato su directly-follows: e' la struttura
        # contro cui si misurano le colonne ``_net``, ed e' ``None`` solo dove
        # la rete non da' un automa finito.
        "net_dfa": net_dfa,
        "tensor_dfa": tensor_dfa, "gll_horizon": gll_horizon,
        "build_logs": logs, "eval_loaders": eval_loaders,
        "train_traces": splits.train,
        "automaton": automaton, "constraints": constraints,
        "test_traces": splits.test, "prefix_lengths": prefix_lengths,
    }


def loss_arguments(mask_key: str, art: dict) -> dict:
    """Come la variante inietta la conoscenza nella loss.

    Restituisce sempre ``allowed_mask`` = la maschera che ``train_model``
    registra come conformita' (empirica, cosi' la history resta leggibile), piu'
    il modo e l'eventuale maschera indicizzata per stato dell'automa.
    """
    if mask_key == "net_state":
        return {"logic_mode": "checker_state",
                "allowed_mask": art["masks"][EVAL_MASK],
                "logic_mask": art["state_mask"]}
    if mask_key == "axel_local":
        # Stessa maschera di ``checker_net``: la conoscenza che entra nella loss
        # e' la rete di Petri proiettata, la stessa che vedono marking, gnn e
        # seq. ``allowed_mask`` resta quella empirica perche' e' la conformita'
        # che ``train_model`` scrive nella history, non il vincolo -- come per
        # ``checker_state``.
        return {"logic_mode": "axel_local",
                "allowed_mask": art["masks"][EVAL_MASK],
                "axel_loss": LocalLogicLoss(art["masks"]["net"],
                                            alpha=AXEL_ALPHA_LOCAL)}
    if mask_key == "axel_global":
        return {"logic_mode": "axel_global",
                "allowed_mask": art["masks"][EVAL_MASK],
                "axel_loss": GlobalLogicLoss(
                    art["tensor_dfa"], horizon=art["gll_horizon"],
                    alpha=AXEL_ALPHA_GLOBAL, temperature=AXEL_TEMPERATURE,
                    num_samples=AXEL_SAMPLES,
                )}
    return {"logic_mode": "checker", "allowed_mask": art["masks"][mask_key]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", choices=("A", "B", "C"), default="B",
                        help="B: split temporale, rumore sugli eventi, knowledge "
                             "dal test. A: split random, rumore sui target, "
                             "knowledge dal train. Cambia solo il config.")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=ALL_DATASETS)
    parser.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=ALL_VARIANTS)
    parser.add_argument("--noises", nargs="+", type=float, default=list(NOISES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--out-dir", default="temporal_matrix",
                        help="sottocartella di runs/ per CSV e checkpoint")
    parser.add_argument("--no-suffix", action="store_true",
                        help="salta la suffix prediction (celle vuote nelle sue colonne)")
    parser.add_argument("--no-artifact-cache", action="store_true",
                        help="mina sempre rete, DFA, automa e vincoli invece di "
                             "riusare quelli salvati in runs/_artifacts")
    parser.add_argument("--no-net-eval", action="store_true",
                        help="non costruire l'automa della rete per la sola "
                             "valutazione: le colonne _net restano vuote. Serve "
                             "dove l'automa non e' finito (BPIC15), e va usato "
                             "insieme alle varianti che non lo richiedono.")
    args = parser.parse_args()

    # gate: seq resta fuori finche' i kind non esistono in models.py
    variants = list(args.variants)
    if "seq" in variants and "gru_seq" not in get_args(ModelKind):
        print("NOTA: kind 'gru_seq' non ancora implementato -> variante 'seq' saltata.")
        variants.remove("seq")

    out_dir = ROOT / "runs" / args.out_dir
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "results.csv"
    # Fuori dalla cartella del protocollo, e apposta: la chiave e' l'impronta
    # delle tracce di knowledge, quindi due esperimenti che minano dalle stesse
    # tracce condividono gli oggetti anche se scrivono CSV diversi. Cancellare
    # runs/noise_curve_* non butta via l'automa di BPIC15 da un'ora.
    artifact_cache = None if args.no_artifact_cache else ROOT / "runs" / "_artifacts"

    done: set[tuple] = set()
    if results_csv.exists():
        with results_csv.open() as handle:
            reader = csv.DictReader(handle)
            header = tuple(reader.fieldnames or ())
            for row in reader:
                done.add((row["dataset"], row["arch"], row["variant"],
                          float(row["noise"]), int(row["seed"])))
        # Un CSV scritto con altre colonne non si puo' estendere in append: le
        # righe nuove sarebbero disallineate rispetto all'header.
        if header != CSV_FIELDS:
            print(f"ERRORE: {results_csv} ha un header diverso da quello atteso.\n"
                  f"  mancanti: {[f for f in CSV_FIELDS if f not in header]}\n"
                  f"  Usa --out-dir con una cartella nuova.")
            raise SystemExit(1)
        print(f"resume: {len(done)} celle gia' nel CSV, verranno saltate")
    else:
        with results_csv.open("w", newline="") as handle:
            csv.writer(handle).writerow(CSV_FIELDS)

    # Il protocollo e' SOLO una scelta di config: quattro campi di DataConfig.
    # Il resto della pipeline non sa quale dei due sta girando.
    if args.protocol == "B":
        base_config = temporal_protocol(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
    elif args.protocol == "C":
        # B con la conoscenza presa dal train invece che dal test, e nient'altro
        # di diverso: stesso split temporale, stesso vocabolario, stesso rumore
        # sugli eventi. Serve a isolare la sorgente della conoscenza, che nel
        # paper di Mezini et al. e' dichiarata come il test set -- quasi
        # certamente un refuso, e questo protocollo lo misura invece di
        # discuterlo.
        # ``temporal_protocol`` accetta come override solo batch_size e
        # num_workers: gli altri campi li scarta in silenzio, ed e' giusto cosi'
        # perche' e' la funzione che DEFINISCE il protocollo. Quindi il cambio si
        # fa dopo, in chiaro, su un campo solo.
        temporal = temporal_protocol(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction
        )
        base_config = replace(
            temporal, data=replace(temporal.data, knowledge_source="train"))
    else:
        base_config = ExperimentConfig(data=DataConfig(
            validation_fraction=args.val_fraction, test_fraction=args.test_fraction,
            split_strategy="random", vocabulary_scope="train",
            knowledge_source="train", noise_model="target",
        ))
    base_config = replace(
        base_config,
        training=replace(base_config.training, logic_weight=LOGIC_WEIGHT,
                         epochs=MAX_EPOCHS),
        model=replace(base_config.model, hidden_dim=RECURRENT_HIDDEN,
                      recurrent_layers=RECURRENT_LAYERS))
    data = base_config.data
    device = resolve_device(base_config.training.device)
    # ``cpu`` quando non c'e' scheda: e' un valore come un altro, e dice da solo
    # perche' quella riga ha dei secondi fuori scala rispetto alle altre.
    gpu_name = (torch.cuda.get_device_name(device)
                if getattr(device, "type", str(device)) == "cuda" else "cpu")
    families = {VARIANT_SPEC[v][2] for v in variants}
    # La maschera di valutazione serve sempre: e' quella su cui si misurano
    # forbidden e violation_rate di OGNI variante.
    masks_needed = {VARIANT_SPEC[v][3] for v in variants} | {EVAL_MASK}
    # La maschera della rete serve alla VALUTAZIONE di tutte e nove le varianti,
    # non solo a chi la usa nella loss: e' il metro primario, e deve essere lo
    # stesso oggetto per ogni riga. Quindi la si chiede sempre, anche quando
    # nessuna variante la userebbe -- per la baseline il numero e' comunque
    # informativo, dice quanto comportamento fuori dalla rete produce un modello
    # a cui la rete non e' mai stata mostrata.
    if not args.no_net_eval:
        masks_needed |= {"net"}

    total = len(args.datasets) * len(args.archs) * len(variants) * len(args.noises) * len(args.seeds)
    print(f"protocollo {args.protocol}: split={data.split_strategy} "
          f"{data.train_fraction:.2f}/{data.validation_fraction:.2f}/{data.test_fraction:.2f}, "
          f"vocabolario={data.vocabulary_scope}, knowledge={data.knowledge_source}, "
          f"rumore={data.noise_model}\n"
          f"suffix prediction: {'OFF' if args.no_suffix else 'ON'} | output: {out_dir}\n"
          f"matrice: {len(args.datasets)} dataset x {len(args.archs)} arch x "
          f"{len(variants)} varianti x {len(args.noises)} noise x {len(args.seeds)} seed "
          f"= {total} run ({len(done)} gia' fatte)", flush=True)
    completed = 0

    for dataset in args.datasets:
        pending = [c for c in (
            (dataset, arch, variant, noise, seed)
            for arch in args.archs for variant in variants
            for noise in args.noises for seed in args.seeds
        ) if c not in done]
        if not pending:
            print(f"[{dataset}] completo, salto")
            continue
        # Sotto rumore sugli eventi, preparare un livello costa quanto
        # addestrarne tutte le varianti (i marking si rigiocano da capo). Chi
        # riprende una matrice interrotta non deve ripagare quel costo per i
        # livelli gia' chiusi, quindi le coppie (noise, seed) ancora da fare si
        # decidono qui, prima di toccare i log.
        pending_levels = {(cell[3], cell[4]) for cell in pending}

        art = build_dataset_artifacts(dataset, base_config, families, masks_needed,
                                      cache_dir=artifact_cache)
        vocab, petrinet = art["vocab"], art["petrinet"]
        n_places = len(petrinet.places)

        # Due regimi di rumore, e cambiano anche il COSTO, non solo l'effetto.
        #
        # ``event`` corrompe le TRACCE, quindi i marking vanno rigiocati a ogni
        # livello: i log si costruiscono una volta per (noise, seed) e si
        # riusano su tutte le arch x varianti della cella -- la proprieta'
        # twin-run. A rumore 0 le tracce sono quelle pulite, uguali per ogni
        # seed, quindi una sola volta.
        #
        # ``target`` lascia le tracce intatte e sostituisce solo l'etichetta da
        # predire: i marking non cambiano mai, quindi i log si costruiscono UNA
        # volta per dataset e si corrompono i target sulla copia. La conformita'
        # del train resta quella pulita a ogni livello -- ed e' esattamente il
        # motivo per cui questo rumore morde molto meno.
        event_noise = base_config.data.noise_model == "event"
        noisy_cache: dict = {}
        clean_logs = None if event_noise else art["build_logs"](art["train_traces"])
        clean_ratio_train = compliance_ratio(art["constraints"],
                                             art["train_traces"].values())

        def noisy_logs(noise: float, seed: int):
            if not event_noise:
                rng = random.Random(seed + int(noise * 1000))
                logs, changed = {}, 0
                for family, log in clean_logs.items():
                    logs[family], changed = log.corrupt_targets(noise, rng)
                return logs, changed, clean_ratio_train

            key = ("clean",) if noise == 0.0 else (noise, seed)
            if key not in noisy_cache:
                rng = random.Random(seed + int(noise * 1000))
                traces, changed = TraceUtils.corrupt_traces(
                    art["train_traces"], noise, rng, art["vocab"].activities
                )
                ratio = compliance_ratio(art["constraints"], traces.values())
                print(f"[{dataset}] noise {noise:.2f} seed {seed}: {changed} eventi corrotti, "
                      f"conformita' del train {ratio:.3f} -- ricostruzione marking...", flush=True)
                noisy_cache.clear()  # una chiave alla volta: i log sono grossi
                noisy_cache[key] = (art["build_logs"](traces), changed, ratio)
            return noisy_cache[key]

        for seed in args.seeds:
            seed_config = replace(base_config, seed=seed)
            for noise in args.noises:
                if (noise, seed) not in pending_levels:
                    print(f"[{dataset}] noise {noise:.2f} seed {seed}: gia' completo, "
                          f"salto la ricostruzione", flush=True)
                    continue
                # twin-run: dentro una cella (dataset, seed, noise) tutte le
                # varianti vedono LO STESSO log corrotto
                train_logs, corrupted, ratio = noisy_logs(noise, seed)
                for arch in args.archs:
                    for variant in variants:
                        cell = (dataset, arch, variant, noise, seed)
                        if cell in done:
                            continue
                        suffix, use_logic, family, mask_key = VARIANT_SPEC[variant]
                        kind = arch + suffix
                        train_loader = train_logs[family].data_loader(data.batch_size, shuffle=True)

                        started = time.perf_counter()
                        train_res = train_model(
                            model_kind=kind,
                            vocabulary=vocab,
                            train_loader=train_loader,
                            validation_loader=art["eval_loaders"]["val"][family],
                            config=seed_config,
                            checkpoint_path=out_dir / "ckpt" /
                                f"{dataset}_{kind}_{variant}_n{int(noise*100)}_s{seed}.pt",
                            use_logic=use_logic,
                            # Quale conoscenza entra nella loss, e come.
                            **loss_arguments(mask_key, art),
                            # Il marking e' una FEATURE: dipende dall'encoder
                            # (il suffisso del kind), non dalla famiglia di dati
                            # -- la famiglia "stated" porta gli stati per la
                            # loss ma il modello resta quello della baseline.
                            marking_dim=n_places if suffix else 0,
                            adjacency=petrinet.adjacency_matrices if suffix in GRAPH_SUFFIXES else None,
                        )
                        secs_train = time.perf_counter() - started
                        secs_per_epoch = secs_train / max(len(train_res.history), 1)

                        # --- task 1: next activity
                        eval_res = evaluate_model(
                            model=train_res.model,
                            data_loader=art["eval_loaders"]["test"][family],
                            # La maschera empirica, uguale per tutte le righe:
                            # e' il metro che esiste sempre, anche dove l'automa
                            # della rete non si costruisce.
                            allowed_mask=art["masks"][EVAL_MASK].to(device),
                            device=device,
                        )
                        # Lo stesso modello contro l'automa della rete. E' un
                        # secondo passaggio sul test set, non un secondo
                        # addestramento: rispetto ai minuti del training e'
                        # rumore. Assente solo quando la rete non da' un automa.
                        if "net" in art["masks"]:
                            eval_net = evaluate_model(
                                model=train_res.model,
                                data_loader=art["eval_loaders"]["test"][family],
                                allowed_mask=art["masks"]["net"].to(device),
                                device=device,
                            )
                            net_cells = [f"{eval_net.violation_rate:.6f}",
                                         f"{eval_net.forbidden_mass:.6f}"]
                        else:
                            net_cells = ["", ""]

                        # --- task 2: suffix prediction. Decoding LIBERO: la
                        # allowed_mask azzererebbe per costruzione il
                        # dfa_violation, rendendo la conformita' non
                        # informativa. Le varianti con marking hanno bisogno
                        # della rete per rigiocare le proprie predizioni.
                        if args.no_suffix:
                            suffix_cells = ["", "", "", "", "", "", "", "", ""]
                        else:
                            sfx = evaluate_suffix_prediction(
                                train_res.model, art["test_traces"], vocab,
                                art["automaton"], device,
                                prefix_lengths=art["prefix_lengths"],
                                constraints=art["constraints"],
                                petrinet=petrinet if suffix else None,
                                # La fitness si misura per OGNI variante: e' una
                                # proprieta' della traccia prodotta, non del
                                # modello che l'ha prodotta.
                                fitness_net=petrinet,
                                # Il secondo metro sulle stesse tracce generate.
                                automaton_net=art["net_dfa"],
                            )
                            net_violation = ("" if art["net_dfa"] is None
                                             else f"{sfx.dfa_violation_rate_net:.6f}")
                            suffix_cells = [
                                f"{sfx.dl_similarity:.6f}", f"{sfx.activity_accuracy:.6f}",
                                f"{sfx.exact_match_rate:.6f}", f"{sfx.dfa_violation_rate:.6f}",
                                net_violation,
                                f"{sfx.precedence_violation_rate:.6f}",
                                f"{sfx.net_fitness:.6f}", sfx.n_examples,
                                "|".join(str(k) for k in art["prefix_lengths"]),
                            ]

                        with results_csv.open("a", newline="") as handle:
                            csv.writer(handle).writerow([
                                dataset, arch, variant, noise, seed,
                                f"{eval_res.accuracy:.6f}", f"{eval_res.macro_f1:.6f}",
                                f"{eval_res.top_k_accuracy:.6f}", f"{eval_res.violation_rate:.6f}",
                                f"{eval_res.forbidden_mass:.6f}",
                                *net_cells,
                                *suffix_cells,
                                train_res.best_epoch, corrupted, f"{ratio:.6f}",
                                f"{secs_per_epoch:.2f}",
                                f"{secs_train:.2f}",
                                f"{art['timings']['petrinet']:.2f}",
                                f"{art['timings']['dfa']:.2f}",
                                f"{art['timings']['automaton']:.2f}",
                                f"{art['timings']['markings']:.2f}",
                                f"{art['timings']['artifacts']:.2f}",
                                gpu_name,
                                art["knowledge_key"], art["net_fingerprint"],
                                int(art["cached"]),
                                data.split_strategy, data.knowledge_source, data.vocabulary_scope,
                                f"{data.train_fraction:.2f}", f"{data.validation_fraction:.2f}",
                                f"{data.test_fraction:.2f}",
                            ])
                        done.add(cell)
                        completed += 1
                        dl_note = "" if args.no_suffix else f" DL {sfx.dl_similarity:.4f}"
                        print(f"[{dataset} | {arch} {variant:<15} | noise {noise:.2f} | seed {seed}] "
                              f"acc {eval_res.accuracy:.4f} forbidden {eval_res.forbidden_mass:.4f}"
                              f"{dl_note} ({completed}/{total})", flush=True)

    # ---- riepilogo dal CSV completo (incluse run di lanci precedenti) -------
    with results_csv.open() as handle:
        rows = list(csv.DictReader(handle))

    def summarise(column: str, title: str) -> None:
        print(f"\n================ {title} (medie su seed) ================")
        for dataset in args.datasets:
            for noise in args.noises:
                for arch in args.archs:
                    cells = [r for r in rows if r["dataset"] == dataset
                             and r["arch"] == arch and float(r["noise"]) == noise]
                    parts = []
                    for variant in variants:
                        vals = [float(r[column]) for r in cells
                                if r["variant"] == variant and r.get(column)]
                        if vals:
                            spread = stdev(vals) if len(vals) > 1 else 0.0
                            parts.append(f"{variant} {mean(vals):.4f}±{spread:.4f} (n={len(vals)})")
                    if parts:
                        print(f"{dataset} | {arch} | noise {noise:.2f}: " + " | ".join(parts))

    summarise("accuracy", "NEXT ACTIVITY — accuracy")
    summarise("dl_similarity", "SUFFIX PREDICTION — Damerau-Levenshtein normalizzata")


if __name__ == "__main__":
    main()
