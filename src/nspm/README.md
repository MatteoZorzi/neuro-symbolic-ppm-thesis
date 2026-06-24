# nspm — neuro-symbolic PPM pipeline

This package turns an XES event log into a reproducible next-activity prediction
experiment. It replaces the old notebook-only implementation with small modules
that can be tested, imported and reused. The Sepsis Cases log is the default
example; other logs under `datasets/` (e.g. BPIC_2013_incidents,
BPIC_2020_DomesticDeclarations) work via `--dataset-name`.

## Data flow

1. `data/xes.py` parses the XES file into one row per event.
2. `pipeline/analysis.py` builds case, activity, variant and transition tables.
3. `data/prefixes.py` splits cases and creates one example per trace prefix.
4. `process/automaton.py` learns a DFA from the training partition only.
5. `learning/models.py` provides GRU, LSTM and causal Transformer models.
6. `learning/logic.py` provides both logic losses (checker and embedder).
7. `learning/training.py` trains and restores the best validation checkpoint.
8. `pipeline/experiment.py` runs and compares baseline and logic-aware models.

The learned-embedder branch of T-LEAF adds:

- `process/ltl_constraints.py` mines LTLf precedence constraints from the
  training traces and builds a DFA (with propositional edge guards) per rule.
- `process/graph_encoding.py` encodes those DFAs and traces as graphs, with the
  paper's edge-to-node lifting, and a differentiable soft feature for the
  predicted step.
- `learning/embedder.py` is the hierarchical embedder (`qe` edge embedder +
  `qm` meta embedder with random-walk aggregation).
- `learning/embedder_training.py` trains it with the triplet hinge loss.
- `learning/logic.py` (`EmbeddingLogicLoss`) is the T-LEAF logic loss
  `||q(A) - q(w_pred)||^2`, differentiable w.r.t. the task model.

## Package structure

```text
nspm/
|-- data/             XES parsing, traces, splits and prefix datasets
|-- process/          empirical DFA and process-conformance rules
|-- learning/         GRU/LSTM, logic loss, training and evaluation
|-- visualization/    EDA, DFA and comparison plots
|-- pipeline/         high-level analysis and experiment orchestration
|-- config.py         typed experiment configuration
`-- cli.py            command-line interface
```

The split is performed by case, never by event or prefix. Consequently, prefixes
belonging to one patient cannot appear in multiple partitions.

## Models

Both recurrent architectures have the same public interface:

```python
logits = model(tokens, lengths)
```

`NextActivityGRU` is compact and fast. `NextActivityLSTM` adds an explicit cell
state and can retain longer dependencies, at the cost of more parameters.
`NextActivityTransformer` uses causal self-attention, sinusoidal positional
encoding and the final real prefix token for classification.

For an equal task/logic objective, run:

```text
python -m src.nspm experiment --model transformer \
  --task-loss-weight 0.5 --logic-weight 0.5
```

This computes exactly `0.5 * cross_entropy + 0.5 * forbidden_mass` for the
logic-aware Transformer. The baseline is trained alongside it only as an
architecture-matched experimental control.

The logic-aware objective is:

```text
cross_entropy + logic_weight * forbidden_probability_mass
```

The DFA is not used to overwrite predictions. It supplies a differentiable
training signal, allowing predictive accuracy and conformance to be measured
independently.

## Commands

From the repository root:

```powershell
python -m src.nspm analyze
python -m src.nspm experiment --model both --epochs 12
python -m src.nspm experiment --model lstm --device cpu --max-cases 200
python -m src.nspm experiment --model transformer --task-loss-weight 0.5 --logic-weight 0.5
python -m src.nspm experiment --model both --logic-weights 0.05 0.1 0.25 0.5 1.0
```

The executed modular notebook is available at:

```text
notebooks/Sepsis_Case_Modular_TLEAF.ipynb
```

The experiment writes:

- `dataset_summary.json`, split sizes and DFA coverage;
- `dfa.json` and `vocabulary.json`, preprocessing metadata;
- one history CSV and checkpoint for every model variant;
- `test_results.csv` and `test_results.json`;
- EDA, DFA, learning-curve and final-comparison figures.

## Run directories

Experiment outputs are not written under `datasets/`. The CLI scans `runs/` and
creates the next available directory automatically:

```text
runs/
|-- Sepsis_Case_run_1/
|   |-- models/
|   |-- test_results.csv
|   |-- validation_logic_weight_search.csv
|   `-- JSON, CSV and PNG artifacts
`-- Sepsis_Case_run_2/
```

Use `--runs-dir` for another root or `--dataset-name` for another prefix.
Providing `--output-dir` explicitly disables automatic numbering.

## Main API

```python
from pathlib import Path
from src.nspm import ExperimentConfig, run_experiment
from src.nspm.pipeline import create_next_run

paths = create_next_run(Path("runs"), "Sepsis_Case")
run = run_experiment(
    input_path=Path("datasets/Sepsis_Case/Sepsis_Cases_Event_Log.xes"),
    output_dir=paths.root,
    model_dir=paths.models,
    config=ExperimentConfig(),
    model_kinds=("gru", "lstm"),
)
print(run.results)
```

Keep `logic_weight` as a hyperparameter selected on validation data. A lower
forbidden mass is useful only when it does not hide a material loss in predictive
quality.

When `--logic-weights` is used, the pipeline selects the lowest validation
forbidden mass among candidates whose validation accuracy is no more than one
percentage point below the corresponding baseline. The test set is evaluated
only after that choice. Adjust the constraint with
`--max-validation-accuracy-drop`.

PyTorch deterministic algorithms are enabled during training, so repeated runs
with the same environment, seed and device use reproducible kernels.

Training also uses validation-loss early stopping. Defaults are five epochs of
patience, a minimum improvement of `1e-4`, and at least five completed epochs.
They can be changed with `--early-stopping-patience`,
`--early-stopping-min-delta`, and `--early-stopping-min-epochs`.

## Relation to the original T-LEAF workflow

The original Action Recognition experiment performs preprocessing, builds a
logical automaton representation, trains a baseline target model, trains
checker-loss and embedder-loss variants, and compares their results. The Sepsis
pipeline preserves preprocessing, DFA construction, target-model training,
checker-style differentiable logic loss, evaluation and visualization.

The XES dataset does not include external LTL formulas or an edge/meta-embedder
training corpus. Therefore its process model is an empirical training-only DFA,
and the logic-aware model corresponds to the original checker-loss branch. It
must not be described as a reproduction of the original learned embedder loss.
