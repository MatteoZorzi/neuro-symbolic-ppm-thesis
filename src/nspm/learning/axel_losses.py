"""Le due loss di Axel (``nesy-suffix-prediction-dfa``), portate sul nostro setup.

Codice di riferimento: ``src/loss/local_loss.py`` e ``src/loss/global_loss.py``
del suo repo. La sostanza delle due loss e' sua; qui cambia solo cio' che DEVE
cambiare perche' l'esperimento e' diverso.

Cosa cambia, e perche'
----------------------
1. **L'automa.** Da lui nasce da una formula LTLf compilata con MONA
   (``ltlf2dfa``); qui e' l'automa di raggiungibilita' della rete di Petri
   scoperta dal log, lo stesso che legge ``checker_net``. Cosi' i due metodi
   vedono **la stessa conoscenza** di tutte le altre varianti e il confronto e'
   fra canali, non fra fonti.

   CORRETTO il 02/09/2026: questa nota diceva "il directly-follows empirico gia'
   usato dal ``checker``". Era vero alla nascita del modulo (24/08) e superato
   dal **27/08**, quando le due loss sono passate alla rete perche' due metodi
   sono confrontabili solo se vedono lo STESSO oggetto simbolico. Il cablaggio
   sta in ``official_experiments/scripts/matrix.py``, dove ``axel_local`` e ``axel_global`` sono
   dentro ``NET_DFA_USERS`` insieme a ``net``. Il termine di paragone diretto e'
   ``checker_net``, non ``checker`` -- come gia' documentato in ``MODELS.md``
   §10 e ``CODE_DOCUMENTATION.md`` §4.9, rimasto sbagliato solo qui.
   Conta perche' col ribaltamento della tesi il DFG empirico esce dal testo:
   se queste due loss ci girassero sopra, meta' dei metodi riportati leggerebbe
   un oggetto che la tesi non descrive.
2. **La forma del batch.** I suoi esempi sono tracce intere con un target per
   passo, i nostri sono prefissi con un solo target. La loss locale collassa
   quindi da "per passo" a "per esempio": e' la stessa quantita', misurata dove
   il nostro modello fa la sua unica predizione.
3. **Gli indici.** Da lui input e output vivono nello stesso spazio; qui no --
   i token di input sono ``(PAD, START, *attivita')`` e le classi di output solo
   le attivita'. Lo scarto e' un offset costante, ricavato invece che cablato.

Cosa NON cambia: la struttura delle due penalita', il ruolo di ``alpha`` come
miscelatore fra supervisione e logica, il Gumbel-Softmax con la sua temperatura,
e il fatto che la globale legga l'accettazione dell'automa alla fine di un
rollout invece che la massa proibita a un passo.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..process.automaton import END, START, ProcessDFA


@dataclass(frozen=True)
class TensorDFA:
    """Il nostro ``ProcessDFA`` in forma tensoriale, derivabile.

    Serve solo alla loss globale: la locale legge una maschera booleana e non ha
    bisogno di transizioni differenziabili. Gli stati sono
    ``0 = START``, ``1..n = attivita'``, ``n+1 = END``, ``n+2 = trap``; le azioni
    sono le attivita' piu' un simbolo ``end`` finale. ``END`` e trap sono
    assorbenti, come nel DeepDFA di Axel.
    """

    transitions: torch.Tensor      # (n_actions, n_states, n_states)
    accepting: torch.Tensor        # (n_states,) 1.0 sugli stati accettanti
    n_states: int
    n_actions: int
    end_action: int
    trap_state: int

    @classmethod
    def from_process_dfa(cls, automaton: ProcessDFA, activities: tuple[str, ...],
                         device) -> "TensorDFA":
        n = len(activities)
        state_of = {START: 0}
        for index, activity in enumerate(activities):
            state_of[activity] = index + 1
        end_state, trap_state = n + 1, n + 2
        n_states, n_actions = n + 3, n + 1
        end_action = n

        transitions = torch.zeros((n_actions, n_states, n_states), device=device)

        def wire(source_name: str, source_index: int) -> None:
            allowed = automaton.allowed_next.get(source_name, frozenset())
            for action, activity in enumerate(activities):
                target = state_of[activity] if activity in allowed else trap_state
                transitions[action, source_index, target] = 1.0
            transitions[end_action, source_index,
                        end_state if END in allowed else trap_state] = 1.0

        wire(START, 0)
        for activity in activities:
            wire(activity, state_of[activity])
        # Assorbenti: una volta accettata o finita nel trap, la traccia non si
        # muove piu'. Senza questo il rollout uscirebbe dalla matrice.
        for absorbing in (end_state, trap_state):
            transitions[:, absorbing, absorbing] = 1.0

        accepting = torch.zeros(n_states, device=device)
        accepting[end_state] = 1.0
        return cls(transitions, accepting, n_states, n_actions, end_action, trap_state)

    def step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Un passo su distribuzioni: ``state`` (B, S), ``action`` (B, A) -> (B, S).

        Rilassamento esatto della transizione dura: con ``state`` e ``action``
        one-hot il risultato e' one-hot, e resta differenziabile rispetto a
        entrambi. E' il ``step_pi`` di Axel, scritto con un einsum al posto della
        catena di ``matmul`` e ``squeeze``, che sulle nostre dimensioni
        (fino a 384 azioni x 193 stati) era il punto in cui la memoria esplodeva.
        """

        return torch.einsum("ba,bs,ast->bt", action, state, self.transitions)

    def initial_state(self, state_indices: torch.Tensor) -> torch.Tensor:
        return F.one_hot(state_indices, num_classes=self.n_states).float()


class LocalLogicLoss(nn.Module):
    """LLL: cross-entropy pesata piu' penalita' sulla massa che l'automa rifiuta.

    Due differenze rispetto al nostro ``checker``, ed e' li' che sta l'idea di
    Axel. La prima: la cross-entropy viene **spenta** sugli esempi il cui target
    e' esso stesso vietato dall'automa. Sotto rumore quel target e' un'etichetta
    corrotta, quindi la loss smette di insegnare al modello a riprodurre errori
    riconoscibili come tali. La seconda: la penalita' e' ``-log(1 - massa)``
    invece della massa, quindi cresce senza limite man mano che la probabilita'
    vietata si avvicina a uno.
    """

    def __init__(self, allowed_mask: torch.Tensor, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve stare in [0, 1], ricevuto {alpha}")
        self.register_buffer("allowed_mask", allowed_mask)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                state_ids: torch.Tensor) -> torch.Tensor:
        rejects = ~self.allowed_mask[state_ids]           # (B, C)

        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        target_rejected = rejects.gather(1, targets.unsqueeze(1)).squeeze(1)
        weights = (~target_rejected).float()
        # Se in un batch OGNI target e' vietato non resta supervisione: l'epsilon
        # di Axel evita la divisione per zero e il termine si annulla da solo.
        weighted = (cross_entropy * weights).sum() / (weights.sum() + 1e-6)

        invalid_mass = (torch.softmax(logits, dim=1) * rejects).sum(dim=1)
        penalty = -torch.log(1.0 - invalid_mass + 1e-6).mean()

        return self.alpha * weighted + (1.0 - self.alpha) * penalty


class GlobalLogicLoss(nn.Module):
    """GLL: il modello prosegue da solo e l'automa giudica la traccia intera.

    Dal prefisso il modello genera ``horizon`` attivita' campionandole con
    Gumbel-Softmax, cosi' la scelta resta differenziabile; l'automa le consuma
    nella sua forma rilassata e alla fine dice quanta probabilita' di
    accettazione e' rimasta. La penalita' e' ``-log`` di quella quantita', media
    su ``num_samples`` rollout indipendenti dallo stesso prefisso.

    E' il contrario della locale: li' si guarda un passo e la distribuzione, qui
    la traccia e il suo esito. E' anche il motivo per cui costa: un rollout e'
    ``horizon`` passi sequenziali su un batch replicato ``num_samples`` volte.
    """

    def __init__(self, dfa: TensorDFA, *, horizon: int, alpha: float = 0.5,
                 temperature: float = 0.5, num_samples: int = 4) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve stare in [0, 1], ricevuto {alpha}")
        self.dfa = dfa
        self.horizon = horizon
        #: Miscelato dal chiamante, non qui: nel runner di Axel la GLL torna la
        #: sola penalita' logica e la combinazione ``alpha * CE + (1-alpha) * GLL``
        #: avviene nel ciclo di training. Tenerlo qui e' solo un modo di far
        #: viaggiare il valore insieme alla loss a cui appartiene.
        self.alpha = alpha
        self.temperature = temperature
        self.num_samples = num_samples

    @staticmethod
    def _gumbel_softmax(logits: torch.Tensor, temperature: float,
                        eps: float = 1e-10) -> torch.Tensor:
        uniform = torch.rand_like(logits)
        noise = -torch.log(-torch.log(uniform + eps) + eps)
        return torch.softmax((F.log_softmax(logits, dim=-1) + noise) / temperature,
                             dim=-1)

    def forward(self, model, tokens: torch.Tensor, lengths: torch.Tensor,
                state_ids: torch.Tensor) -> torch.Tensor:
        samples = self.num_samples
        batch = tokens.size(0)

        tokens = tokens.repeat_interleave(samples, dim=0)
        lengths = lengths.repeat_interleave(samples, dim=0)
        # Lo stato dell'automa dopo il prefisso e' noto in forma dura: nel
        # directly-follows lo stato E' l'ultima attivita'. Non serve rigiocare
        # il prefisso in forma rilassata, e non conviene -- sarebbe gradiente
        # speso su una parte del testo che il modello non ha generato.
        state = self.dfa.initial_state(state_ids.repeat_interleave(samples, dim=0))

        logits, hidden = model.encode(tokens, lengths)
        for _ in range(self.horizon):
            soft_action = self._gumbel_softmax(logits, self.temperature)
            padded = F.pad(soft_action, (0, self.dfa.n_actions - soft_action.size(1)))
            state = self.dfa.step(state, padded)
            logits, hidden = model.forward_from_state(soft_action, hidden)

        end_action = F.one_hot(
            torch.full((tokens.size(0),), self.dfa.end_action, device=tokens.device),
            num_classes=self.dfa.n_actions,
        ).float()
        state = self.dfa.step(state, end_action)

        acceptance = (state * self.dfa.accepting).sum(dim=1).view(batch, samples)
        return -torch.log(acceptance.mean(dim=1).clamp(min=1e-10)).mean()
