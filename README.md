# Benchmarking Neuro-Symbolic Predictive Process Monitoring

Master's thesis project. It benchmarks ways of injecting **procedural background
knowledge** — a Petri net discovered from an event log — into a recurrent
**predictive process monitoring** model, and asks which channel carries the
effect and what each one costs.

The reference work being benchmarked is Axel Mezini's *Neuro-Symbolic Predictive
Process Monitoring* (MSc thesis, 2024/25; supervisors Maggi and Donadello) and
the article derived from it. Its two logic losses are reimplemented here in
`src/nspm/learning/axel_losses.py` and are **his contribution, not ours**.

## What it does

Eight models share one recurrent trunk, one training procedure and **one
discovered Petri net**, and differ only in how that net reaches them. Holding the
knowledge fixed while varying the channel is what makes the comparison a
statement about mechanisms rather than about process models.

| variant | channel | how the knowledge enters | whose |
|---|---|---|---|
| `baseline` | — | it does not | — |
| `checker_net` | loss | reachability automaton, projected onto activity pairs | ours |
| `checker_net_state` | loss | the same automaton, indexed by state | ours |
| `lll` | loss | rejected mass as `−log(1−r)`, plus gated cross-entropy | **Mezini** |
| `gll` | loss | acceptance of a differentiable Gumbel-Softmax rollout | **Mezini** |
| `marking` | feature | the flat marking from token replay | ours |
| `gnn` | feature | the marking over the net's bipartite graph | ours |
| `seq` | feature | the sequence of markings, read recurrently | ours |

Each model is measured on next-activity prediction **and** on suffix generation,
on four event logs, at nine noise levels from a clean log to 80% corrupted, under
two protocols that differ only in whether the knowledge is mined from the test
partition or from the training one.

Headline result: both logic losses reduce forbidden probability mass in every
cell of both grids, the feature channel does not move it at all, and the cheapest
method is the most effective one.

## Repository layout

```text
official_experiments/   the results of the thesis and the code that produced them
  protocol-test/        one grid per log, knowledge mined from the test split
  protocol-train/       the same, knowledge mined from the training split
  all_grids.csv         the eight grids concatenated: what every script reads
  figures/              the figures the thesis includes
  scripts/              matrix.py and noise_curve.py (the experiment), plus one
                        script per table and one per figure
  run_slurm.sh          the cluster job that produced the grids
src/nspm/               the library the experiment is built from
  data/                 XES parsing, splits, prefix logs, noise injection
  process/              Petri net + token replay, reachability automaton,
                        directly-follows automaton, precedence constraints
  learning/             recurrent models, logic losses, training, evaluation,
                        suffix generation
scripts/                the probes that led to the design, and the checks on the
                        symbolic layer
datasets/               the four event logs
```

`official_experiments/README.md` explains the results directory and lists every
command that regenerates a table or a figure. `src/nspm/README.md` is the reading
order of the library.

## Environment

```bash
conda env create -f environment.yml
conda activate tleaf
```

The system Python will not do: `pandas` and `pm4py` are pinned in this
environment (`numpy>=1.26,<2`).

The runs themselves were executed on the university HPC cluster, not on a
workstation. `environment.yml` describes the local environment and does **not**
match the one that produced the numbers; the versions stated in §5.2.4 of the
thesis are the authoritative ones.

## Usage

Run all commands from the repository root.

Regenerate every table and figure of the thesis:

```bash
python official_experiments/scripts/summary_table.py
python official_experiments/scripts/metric_figures.py --metric forbidden_net --paired
```

The full list is in `official_experiments/README.md`.

Run a noise curve (one protocol, one log, nine noise levels):

```bash
python official_experiments/scripts/noise_curve.py --protocol test --dataset Sepsis_Case
```

Training writes under `runs/`, which is git-ignored, and resumes: a cell already
in the CSV is skipped, so a grid can be run in slices.

Check that the symbolic layer does what it claims:

```bash
python scripts/verify_symbolic.py
```

## Credits & citation

The two logic losses, local and global, are the work of **Axel Mezini**:

> Axel Mezini. *Neuro-Symbolic Predictive Process Monitoring.* MSc thesis, Free
> University of Bozen-Bolzano, 2024/25. Repositories:
> <https://github.com/axelmezini/suffix-prediction> (global) and
> <https://github.com/axelmezini/nesy-suffix-prediction-dfa> (local).

They are reimplemented here against a different symbolic object so that every
variant reads the same knowledge; the structure of both penalties is his.

The project began from **T-LEAF**, whose logic-loss formulation and graph
encoding shaped its early design:

> Yaqi Xie, Fan Zhou, Harold Soh. *Embedding Symbolic Temporal Knowledge into
> Deep Sequential Models.* arXiv:2101.11981 (NUS).
> <https://github.com/clear-nus/T-LEAF>

The feature channel follows Theis and Darabi (decay replay) and TACO
(Rama-Maneiro, Vidal and Lama). The PPM protocol conventions come from Di
Francescomarino, Donadello and Maggi, *Predictive Process Monitoring* (Springer,
2026).
