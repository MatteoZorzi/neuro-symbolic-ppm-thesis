"""Gli oggetti minati, salvati su disco e richiamabili identici.

Il problema
-----------
``pm4py.discover_petri_net_inductive`` non e' riproducibile fra processi. Non
per l'algoritmo -- quello e' deterministico -- ma perche' genera **nomi nuovi**
per posti e transizioni a ogni chiamata, e ``PetriNet.from_traces`` ordina
``places`` proprio per nome. Cambia l'ordine, cambiano i vettori di marking, e
due run sullo stesso log non sono piu' la stessa cosa. Su BPIC20 la rete esce
con 21 posti quasi sempre e 22 ogni tanto.

Finche' tutto girava dentro un processo solo il danno era invisibile: la rete
veniva minata una volta e usata per l'intera griglia. Diventa visibile appena
si prova a **ricaricare** qualcosa -- rivalutare i checkpoint con una metrica
nuova, rifare le figure, riprodurre un numero della tesi. E' il motivo per cui
la pipeline a due stadi (``final_matrix.py`` + ``suffix_on_t12.py``) e' stata
abbandonata il 21 agosto 2026, ed e' il motivo per cui il 27 agosto una
modifica alla sola valutazione ha costretto a ributtare via giorni di
addestramento.

La cura
-------
Si salvano gli oggetti, non si prova a rendere deterministico il miner. La rete
di pm4py e' gia' serializzabile -- ``PetriNet.prefix_markings`` la spedisce ai
processi worker via ``multiprocessing``, quindi passa per il pickle a ogni run
-- e con lei si salvano l'automa empirico, l'automa di raggiungibilita' e i
vincoli di precedenza.

La chiave e' l'impronta delle **tracce di knowledge**, non il nome del dataset:
sotto B la conoscenza viene dal test e sotto C dal train, quindi le due reti
devono essere diverse, e una chiave che non lo sapesse riuserebbe in silenzio
la rete dell'esperimento sbagliato. Con l'impronta del contenuto la domanda
"sono le stesse tracce?" ha una risposta esatta e non c'e' niente da ricordarsi
di aggiornare.

Il pickle e l'impronta
----------------------
Sono due cose diverse e servono a due scopi diversi.

Il **pickle** e' una cache di lavoro: legata alla versione di pm4py e di
Python, ottima dentro un ambiente, inutile come archivio. Se non si apre non
succede niente di grave, si rimina.

L'**impronta** (:func:`net_fingerprint`) e' una stringa corta che va nel CSV,
accanto ai risultati. Non invecchia, perche' e' costruita solo su cose portabili
-- conteggi ed etichette, mai i nomi generati -- e serve a **verificare**: due
righe con la stessa impronta hanno visto la stessa rete, due righe con impronte
diverse no, e una ricostruzione futura si puo' confrontare con quella scritta.
E' la cosa che finora mancava del tutto: senza, non era nemmeno possibile
accorgersi che la rete era cambiata.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Cambiare questo numero invalida tutte le cache esistenti. Va alzato quando
#: cambia la FORMA di cio' che si salva, o quando cambia il modo in cui gli
#: oggetti vengono costruiti -- altrimenti si riusa una rete minata con
#: parametri diversi credendo che sia quella giusta.
FORMAT_VERSION = 1


def knowledge_fingerprint(traces: Mapping[str, Sequence[str]],
                          **parameters: Any) -> str:
    """Impronta delle tracce da cui si mina, piu' i parametri del miner.

    Le tracce entrano ordinate per case id, e ogni traccia come sequenza di
    attivita': e' il contenuto che decide, non l'ordine in cui il dizionario e'
    stato costruito. I ``parameters`` sono le cose che cambiano il risultato a
    parita' di tracce -- oggi solo ``noise_threshold``.
    """

    digest = hashlib.sha256()
    digest.update(f"v{FORMAT_VERSION}\n".encode())
    for name in sorted(parameters):
        digest.update(f"{name}={parameters[name]!r}\n".encode())
    for case_id in sorted(traces):
        digest.update(case_id.encode("utf-8", "replace"))
        digest.update(b"\x00")
        digest.update("\x01".join(traces[case_id]).encode("utf-8", "replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def net_fingerprint(net) -> str:
    """Impronta portabile della rete, da scrivere nel CSV.

    Dentro ci sono i conteggi (posti, transizioni, transizioni silenti) e le
    etichette ordinate. **Non** ci sono i nomi di posti e transizioni: sono
    proprio quelli che pm4py genera da capo a ogni chiamata, quindi metterli
    renderebbe l'impronta diversa a ogni run e non direbbe piu' niente.

    Cosa cattura: la divergenza tipica del miner, cioe' una rete con un posto o
    una transizione in piu' o in meno, e qualunque cambio nell'insieme delle
    attivita' modellate. Cosa non cattura: due reti con gli stessi conteggi e le
    stesse etichette ma archi diversi. E' un compromesso deliberato -- la
    topologia si puo' confrontare solo attraverso i nomi, che non sono stabili.
    """

    silent = sum(1 for t in net.transitions if t.label is None)
    labels = sorted(t.label for t in net.transitions if t.label is not None)
    digest = hashlib.sha256()
    digest.update(f"{len(net.places)}/{len(net.transitions)}/{silent}\n".encode())
    for label in labels:
        digest.update(label.encode("utf-8", "replace"))
        digest.update(b"\n")
    return digest.hexdigest()[:12]


def cache_path(directory: Path, dataset: str, key: str) -> Path:
    """Il file di una combinazione ``(dataset, tracce di knowledge)``.

    Il nome del dataset c'e' solo per rendere la cartella leggibile a occhio:
    l'identita' sta tutta nella chiave.
    """

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in dataset)
    return directory / f"{safe}.{key}.pkl"


def load(path: Path) -> dict:
    """Il contenuto salvato, o un dizionario vuoto.

    Non solleva mai: una cache che non si apre e' una cache che non c'e'. Il
    caso normale in cui succede e' un aggiornamento di pm4py, e la risposta
    giusta e' riminare, non fermare la griglia.
    """

    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:  # noqa: BLE001 - vedi docstring
        print(f"cache degli artefatti illeggibile ({path.name}): {error} "
              f"-- si rimina", flush=True)
        return {}
    if payload.get("format_version") != FORMAT_VERSION:
        return {}
    return payload


def save(path: Path, payload: dict) -> None:
    """Scrittura atomica: temporaneo nella stessa cartella, poi ``replace``.

    Sul cluster piu' job possono minare la stessa combinazione insieme -- B e C
    di dataset diversi partono in parallelo, e un rilancio dopo uno ``scancel``
    ricomincia da capo. Con la rinomina atomica l'ultimo che finisce vince e
    nessuno legge mai un file scritto a meta'; i due contenuti sono comunque
    equivalenti, perche' la chiave e' il contenuto delle tracce.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "format_version": FORMAT_VERSION}
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    except Exception as error:  # noqa: BLE001
        print(f"cache degli artefatti non salvata ({path.name}): {error}",
              flush=True)
        temporary.unlink(missing_ok=True)
