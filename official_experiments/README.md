# Official experiments

The results Chapter 5 reports, and the code that turns them into its tables and figures.

## The tree

```
protocol-test/   knowledge mined from the TEST partition   (test protocol, letter B in the grids)
protocol-train/  knowledge mined from the TRAIN partition  (train protocol, letter C in the grids)
  <log>/results.csv                                        one grid
all_grids.csv                                              the eight grids in one file
figures/                                                   the figures of the thesis
scripts/                                                   the experiment, the tables and the figures
run_slurm.sh                                               the job that produced the grids
```

`scripts/` holds two kinds of file. `matrix.py` and `noise_curve.py` **are** the
experiment: one cell of a grid is one call to the first, and the second drives
the nine noise levels. Everything else reads the grids they produced and turns
them into a table or a figure.

A **grid** is a pair (log, knowledge source): 9 methods × 9 noise levels × 10 seeds = **810 cells**. 
Temporal split over all cases, no filtering by variant or by trace, noise injected on the events.

Directory names follow the source of the knowledge: `knowledge` column, `test` or `train`.

## Experiments

Eight grids four logs by two protocols, 810 cells each, seeds 0 to 9 for 6480 rows in `all_grids.csv`.

The eight files under `protocol-test/` and `protocol-train/` are the source of the experiments.
`all_grids.csv` is the same rows concatenated, and exists because it unifies all the experiment data.

## The hardware

**Every cell ran on the same accelerator**, an NVIDIA A100-SXM4-80GB, with the  same stack (torch 2.13.0+cu130). 
The `secs_` columns are therefore comparable across methods, noise levels, protocols and logs.

## Rebuilding

From the repository root, in the `tleaf` environment.

Rebuild `all_grids.csv` from the eight grids:

```
python official_experiments/scripts/publish_experiments.py
```

The tables of Chapter 5, each one named in the caption of the table it produces:

```
python official_experiments/scripts/dataset_table.py
python official_experiments/scripts/artifacts_table.py
python official_experiments/scripts/compliance_table.py
python official_experiments/scripts/summary_table.py
python official_experiments/scripts/cost_table.py
```

The figures. Each one is written here and mirrored into `docs/Thesis/images/`, so that a figure regenerated in one place cannot disagree with the text in the other.

```
python official_experiments/scripts/metric_figures.py --metric forbidden_net --paired
python official_experiments/scripts/metric_figures.py --metric suffix_dfa_violation_net --paired
python official_experiments/scripts/metric_figures.py --metric accuracy --paired
python official_experiments/scripts/metric_figures.py --metric dl_similarity --paired
python official_experiments/scripts/cd_diagram.py --metric forbidden_net
python official_experiments/scripts/cost_figure.py
python official_experiments/scripts/architecture_figures.py
```

Without `--paired`, `metric_figures.py` draws the two protocols as two separate figures instead of one. 
The thesis uses the paired form only.

`scripts/common.py` holds what the scripts share: the palette, the model names, the paths. 
Three scripts import from `src/`: `matrix.py` is the experiment, `dataset_table.py` reads the event logs through `nspm.data.loader`, and `artifacts_table.py` re-mines the symbolic artifacts through the same code the runs used.

Everything here runs from a clone. `artifacts_table.py` re-mines each net and compares its fingerprint against the one recorded in `all_grids.csv`; it uses the cache under `runs/_artifacts` when there is one, and mines from the logs when there is not.

## Running the experiment

One cell is one call to `matrix.py`; a grid is the nine noise levels, which `noise_curve.py` drives. Both write under `runs/`, and both resume: a cell already in the CSV is skipped, so a grid can be run in slices and survives interruption.

```
python official_experiments/scripts/noise_curve.py --protocol test --dataset Sepsis_Case --variants baseline checker checker_net checker_net_state marking gnn seq lll gll
```

## Provenance

The grids were produced on the university HPC cluster by `run_slurm.sh`.
One job per log, since the logs are independent, and the per-cell resume makes a job repeatable without losing work if SLURM preempts it. 
It is submitted from the repository root.
```
sbatch official_experiments/run_slurm.sh test BPI_Challenge_2012
sbatch official_experiments/run_slurm.sh train Sepsis_Case all
```

Without variants the job runs `lll gll` only, the two methods added last to the grids; `all` expands to the nine methods, and any explicit list is passed through as is.

The job writes under `runs/`. Each log of the thesis ran in three launches (seed 0, seeds 1-4, seeds 5-9), and `merge_runs.py` joins them into the file under `protocol-test/` or `protocol-train/`, adding the `protocol` and `source_dir` columns:
```
python official_experiments/scripts/merge_runs.py --protocol test --dataset Sepsis_Case --sources noise_curve_b noise_curve_b_s14_Sepsis_Case noise_curve_b_s59_Sepsis_Case
```
