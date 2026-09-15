# nspm — the library the benchmark is built from

This package turns an XES event log into a next-activity and suffix-prediction
experiment. It is a library and has no command line: the experiment lives beside
the results it produced, in `official_experiments/scripts/`, where `matrix.py` is
the worked example of how these pieces fit together.

## Reading order

1. `data/loader.py` parses the XES (or CSV) file into one row per event.
2. `data/preparation.py` splits the cases and creates one example per trace
   prefix: `TraceSplits`, `ActivityVocabulary`, `PrefixLog`, and the two noise
   models — on the events or on the targets.
3. `process/petrinet.py` discovers a Petri net with the pm4py inductive miner,
   computes the per-prefix markings by token replay, and exposes the bipartite
   place/transition adjacency matrices.
4. `process/reachability.py` turns that same net into a deterministic automaton
   by traversing every reachable marking.
5. `process/automaton.py` learns the directly-follows automaton from the traces,
   which is the empirical yardstick every variant is evaluated against.
6. `learning/models.py` provides the GRU and LSTM trunks and the three marking
   encoders.
7. `learning/logic.py` and `learning/axel_losses.py` provide the logic losses.
8. `learning/training.py` trains and restores the best validation checkpoint.
9. `learning/trace_prediction.py` generates the suffix and scores it.

The split is performed by case, never by event or prefix: prefixes belonging to
one case cannot appear in two partitions.

```text
nspm/
|-- data/       XES/CSV parsing, traces, splits and prefix datasets
|-- process/    Petri net, reachability automaton, empirical DFA, constraints
|-- learning/   GRU/LSTM, marking encoders, logic losses, training, evaluation
`-- config.py   typed experiment configuration
```

## The two channels

The whole benchmark is one net reaching the model in different ways.

**As a feature.** `PrefixLog.with_markings` attaches each prefix's marking to its
examples, and the recurrent models take an injected `marking_encoder` whose
contract is `forward(markings) -> (B, output_dim)`. `FlatMarkingEncoder` passes
the raw vector through (`*_marking`); `HeteroGraphEncoder` does two-hop message
passing over the bipartite graph (`*_gnn`); `MarkingSequenceEncoder` reads the
whole sequence of markings recurrently (`*_seq`). `build_model` picks the encoder
from the kind, and checkpoints carry the adjacency matrices.

**As a loss.** `to_process_dfa()` projects the reachability automaton onto
directly-follows form, so the last-token `build_allowed_mask` accepts it
(`checker_net`). Or `logic.build_state_mask` keeps one row per automaton state
and `PrefixLog.with_automaton_states` supplies the per-prefix state, so the net's
memory survives into the loss (`checker_net_state`). The projection is not free:
on Sepsis it discards 31.9% of the decisions where the stateful automaton forbids
and the directly-follows view does not.

Neither loss variant changes the network: the automaton state is not a model
input, it only selects a row of the constraint mask. Every variant is evaluated
against the same empirical DFA, so the conformance columns stay comparable across
rows.

## The objectives

```text
checker        cross_entropy + logic_weight * forbidden_probability_mass
LLL            alpha * weighted_cross_entropy + (1 - alpha) * -log(1 - mass)
GLL            alpha * cross_entropy          + (1 - alpha) * -log(acceptance)
```

The automaton is never used to overwrite predictions. It supplies a
differentiable training signal, which is what lets predictive accuracy and
conformance be measured independently.

The two in `learning/axel_losses.py` are the work of Mezini et al.; unlike the
checker they replace the loss rather than adding a weighted term, because both
blend task and logic through their own `alpha`. `weighted_cross_entropy` drops
the examples whose ground-truth target is itself rejected by the automaton: under
label noise those targets are corrupted, so the objective stops teaching
recognisable mistakes. GLL instead rolls the model forward with Gumbel-Softmax
and scores the whole generated trace against a tensorised copy of the automaton.

## Also here, and out of the thesis

Code kept because it was part of the work, not because the benchmark uses it:
the T-LEAF embedder (`learning/embedder.py`, `learning/embedder_training.py`,
`process/graph_encoding.py`, and `EmbeddingLogicLoss` in `learning/logic.py`),
and the graph-recurrent encoder in `learning/models.py`.

## Determinism

PyTorch deterministic algorithms are enabled during training, so repeated runs
with the same environment, seed and device use reproducible kernels. Training
uses validation-loss early stopping: five epochs of patience, a minimum
improvement of `1e-4`, and at least five completed epochs.

The Petri net is the one thing that is **not** reproducible across processes:
pm4py generates fresh names for places and transitions at every call, and the
marking vectors depend on their order. `process/artifact_store.py` exists for
that reason — it saves the mined objects and recalls them identical, keyed by the
fingerprint of the traces they were mined from.
