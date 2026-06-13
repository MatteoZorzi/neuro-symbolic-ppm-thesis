# Neuro-Symbolic Predictive Process Monitoring with Procedural Background Knowledge

Master's thesis project. It studies how **procedural background knowledge**
(temporal-logic constraints over process activities) can be injected into deep
sequential models for **predictive process monitoring**, using the
neuro-symbolic *logic loss* of T-LEAF as the regularisation mechanism.

The case study is the **Sepsis Cases** event log: a next-activity prediction
task where GRU / LSTM / Transformer models are trained both as plain baselines
and with two differentiable logic penalties.

> This repository builds on the T-LEAF method by Xie, Zhou & Soh. The original
> code (synthetic, action-recognition and imitation-learning experiments) has
> been removed; this project keeps and extends only the process-monitoring
> branch. See [Credits & citation](#credits--citation).

## What it does

For each prefix of a process trace the model predicts the next activity. Two
neuro-symbolic regularisers steer it towards process-conformant predictions:

- **checker** — penalises the probability mass the model assigns to next
  activities forbidden by the empirical directly-follows automaton learned from
  the training log.
- **embedder** — the actual T-LEAF logic loss `||q(A) - q(w_pred)||^2`: the
  squared distance, in a learned graph-embedding space, between a relevant LTLf
  constraint's DFA and the model's predicted continuation. Mined LTLf
  *precedence* constraints supply the procedural background knowledge.

No `spot` / `ltlf2dfa` dependency: the (small) constraint automata are built
directly, so the pipeline runs on native Windows.

## Environment

```bash
conda env create -f environment_windows.yml
conda activate tleaf
```

## Usage

Run all commands from the repository root (the package is imported as
`src.Sepsis_Case`).

Descriptive process analysis on the XES log:

```bash
python -m src.Sepsis_Case analyze
```

Train baselines and logic-aware variants (checker branch) and compare them:

```bash
python -m src.Sepsis_Case experiment --model both
```

The learned-embedder branch (full T-LEAF logic loss) is driven from the
notebook / Python API; see `src/Sepsis_Case/README.md`.

Place the event log at `./datasets/Sepsis_Case/Sepsis_Cases_Event_Log.xes`;
results, tables and plots are written under `./datasets/Sepsis_Case/` and
`./runs/` (both git-ignored).

## Repository layout

```text
src/Sepsis_Case/      neuro-symbolic predictive-monitoring pipeline (the project)
  data/               XES parsing, traces, splits, prefix datasets
  process/            empirical DFA + LTLf constraints + graph encoding
  learning/           GRU/LSTM/Transformer, logic losses, embedder, training
  visualization/      EDA, DFA and model-comparison plots
  pipeline/           analysis and experiment orchestration
notebooks/            end-to-end and modular T-LEAF notebooks
docs/                 ARCHITECTURE.md (reading map) and review notes
```

## Documentation

- `docs/ARCHITECTURE.md` — module dependency graph and recommended reading order.
- `src/Sepsis_Case/README.md` — detailed package documentation and data flow.
- `docs/CODE_REVIEW_NOTES.md` — design/cleanup notes.

## Credits & citation

This work builds directly on **T-LEAF**:

> Yaqi Xie, Fan Zhou, Harold Soh. *Embedding Symbolic Temporal Knowledge into
> Deep Sequential Models.* arXiv:2101.11981 (NUS).

Original implementation: <https://github.com/clear-nus/T-LEAF>. The logic-loss
formulation, hierarchical embedder and graph-encoding ideas are theirs; this
repository adapts them to predictive process monitoring on the Sepsis log and
adds the LTLf-precedence procedural background knowledge.
