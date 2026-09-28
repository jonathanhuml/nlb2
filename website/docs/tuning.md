# Hyperparameter tuning

`Experiment` executes one concrete configuration. `Study` uses Ray Tune to
choose configurations, run them as ordinary experiments, and select by a
declared validation metric. Ray is optional. In an existing NLB2 environment:

```bash
pip install "ray[tune]>=2.43,<3" "optuna>=3,<5"
nlb2 tune -c configs/tuning/smoothing_lorenz.yaml
```

These dependencies are also declared in NLB2's `tuning` installation extra.

The smoothing recipe is a small CPU workflow example. It samples four
bandwidths, including the existing setting, and selects the lowest validation
rate MSE. `configs/tuning/stndt_mc_maze.yaml` shows a full-budget GPU TPE study;
it requires prepared MC_Maze validation data and runs 20 candidates across
three training seeds, so it is a substantial training job.

## Python API

```python
from nlb2 import Study

study = Study.from_config_path("configs/tuning/smoothing_lorenz.yaml")
result = study.run()
print(result.best_score)
print(result.best_config_path)
print(result.best_checkpoints)
```

You can also construct `StudyConfig(base=experiment_config, objective=...,
search_space=...)` and pass it to `Study`. The base accepts an
`ExperimentConfig` or a resolved experiment dictionary. `StudyResult.trials`
contains successful, failed, and incomplete candidates. If no candidate
finishes successfully, `best_config` and `best_score` are `None`; the CLI
returns a nonzero status. `StudyResult.from_path(directory)` reads results
without starting Ray.

## A study recipe

```yaml
base: ../experiment/real/mc_maze/stndt/stndt_mc_maze_nlb_5ms.yaml
objective:
  metric: co_bps
  mode: max
  checkpoint: best
evaluation_interval: 50
search_space:
  model.optimization.lr:
    distribution: loguniform
    low: 0.0001
    high: 0.01
  model.dropout:
    distribution: uniform
    low: 0.0
    high: 0.6
algorithm: tpe
scheduler: fifo
num_samples: 20
include_base: true
training_seeds: [0, 1, 2]
search_seed: 7
evaluation_seed: 10000
resources: {cpu: 4, gpu: 1}
max_concurrent_trials: 1
output_dir: studies
name: stndt_mc_maze
```

`base` can instead contain an inline experiment mapping. A referenced base
file resolves relative to the study recipe. Data and model input paths resolve
relative to the working directory at launch, matching `nlb2 run`; the study
records absolute paths before Ray workers change directories.

Supported distributions are `choice` with a `choices` list, `uniform`,
`loguniform`, and `randint` with `low` and `high`. The integer upper bound is
exclusive. Search paths use experiment YAML names, including
`model.optimization.lr`, `trainer.batch_size`, and
`preprocessing.observations.0.kern_sd_ms`. Preprocessing steps must already
exist in the base. Unknown paths are rejected before launch.

Data, data splits, input files, model identity, optimization strategy, seeds,
and training budgets are fixed study settings. They cannot be search paths.
Model and preprocessing fields retain their normal config validation, and
every sampled candidate is revalidated before constructing its model.

Use `parameter_sets` for coupled choices, such as compatible widths and head
counts. Every set must specify the same fields; these fields cannot also
appear in `search_space`:

```yaml
parameter_sets:
  - {model.hidden_size: 128, model.num_heads: 2}
  - {model.hidden_size: 192, model.num_heads: 3}
```

The base must lie inside the declared search space, including one of the
parameter sets, when `include_base: true`. `points_to_evaluate` accepts complete
parameter mappings for additional initial candidates. `num_samples` includes
the base and these initial points. General conditional expressions and
executable code in YAML are not supported.

## Search, selection, and budgets

`algorithm: random` uses Ray's seeded random generator. `algorithm: tpe` uses
Optuna's seeded TPE sampler through Ray. Both currently use FIFO scheduling:
each candidate receives the base experiment's complete training budget.
ASHA is deferred until model-specific learning curves and grace periods have
been established.

One Ray trial runs a candidate sequentially across every `training_seeds`
entry. The search objective is the arithmetic mean of the completed seed
scores. `score_std` is the sample standard deviation, or zero for one seed.
Partial and failed candidates cannot win or supply TPE with an objective.
Epoch reports remain available as progress and recovery checkpoints.

`checkpoint: last` scores the final model. `checkpoint: best` selects the
best fresh validation evaluation within each seed run; it requires a positive
evaluation interval for nonzero-epoch training. `evaluation_interval` overrides
the base's `trainer.live_eval_interval`; otherwise the base setting is used.
The model, metrics, predictions, and selected epoch refer to the same
checkpoint. A missing or nonfinite objective fails that candidate.

Zero-epoch methods still execute `fit_training_data` and final evaluation.
CPU/GPU reservations apply to each candidate; this API supports a single CPU
or single-GPU training process per seed run, with the requested CPU thread
count. Concurrency limits work for both search algorithms. `time_budget_s`
optionally sets a study-wide wall-clock limit; candidates cut off before all
seed runs finish remain incomplete. Training budgets must be chosen for each
model family: equal epochs do not imply equal compute. Recorded elapsed time
includes preparation, fitting, validation, and artifact generation, while
`history.csv` retains the existing optimizer-epoch timing definition.

## Reproduction and recovery

A study directory contains:

- `study.json`: the fully resolved study recipe.
- `provenance.json`: source and input-file hashes, dataset configuration
  identity, Git commit, and software versions.
- `trials.csv` / `trials.json`: candidate parameters, status, scores, seed
  results, and errors.
- `selection.json`: the objective, winning candidate, mean/std, and per-seed
  checkpoint paths.
- `best_config.yaml`: an ordinary experiment recipe for the winning
  hyperparameters and first declared training seed.
- `trials/`: exported per-seed experiment artifacts and training states.
- `ray/`: Ray's logs, search state, and checkpoints.

Reproduce the first seed with `nlb2 run -c STUDY_DIR/best_config.yaml`.
When multiple seeds are used, the mean score describes several fitted models;
`best_checkpoints` therefore maps each seed to its model. There is no single
checkpoint with the aggregate score. Exported configs record each seed and
can be loaded by the usual experiment API.

Resume an interrupted study with its saved recipe:

```bash
nlb2 tune --resume-from studies/my_study
```

Ray restores its search state and errored/unfinished trials from their latest
reported checkpoints. Completed seed runs are retained. Each seed checkpoint
contains model, optimizer/scheduler, epoch, history, selection, and Python,
NumPy, and PyTorch RNG state. Restoration checks that the study, data, source,
and recorded software versions match. Changing the search or adding candidates
creates a new study. A wall-clock-stopped trial that Ray has already marked
terminated is not extended by restoration.

For long experiments, configure periodic evaluation so recovery checkpoints
exist before the final epoch. Evaluation randomness is isolated from training.
`search_seed`, `training_seeds`, and the base dataset seed have separate roles;
model-specific initialization generators are also set from each training seed.
Parallel TPE suggestions can depend on completion order, and numerical
reproducibility across devices is not guaranteed. Use serial trials for the
most repeatable adaptive search.

Local execution works without explicit Ray setup. To use an existing cluster,
initialize Ray before `Study.run()`. All workers need the same installed code
and access to the recorded input paths; the study's local storage path must
be a shared filesystem on a multi-node cluster. Cloud storage URI support and
automatic data distribution are not part of this API.

## Experiment reporting

The underlying experiment API also exposes the new facilities without Ray:

```python
from nlb2 import Experiment

experiment = Experiment.from_config_path("experiment.yaml")
result = experiment.run(callback=lambda report: print(report.epoch, report.metrics))
```

Callbacks receive `ExperimentReport` after fresh task evaluations and once
after final artifacts are written. `final` distinguishes the last report.
Returning `False` from an intermediate callback stops training and retains
artifacts with status `stopped`. `ExperimentResult.completed_epochs` records
the actual budget, and `selected_epoch` identifies the exported model.

Experiment YAML supports a `selection` block with the same metric, mode,
and checkpoint fields, and `experiment.training_seed`, `evaluation_seed`,
`save_training_state`, and `save_plots`. Ordinary experiments retain final
model behavior when no selection is requested. With training-state saving
enabled, `Experiment.run(resume_from="training_state.pt")` or
`nlb2 run -c experiment.yaml --resume-from training_state.pt` continues into
a new run directory. The training configuration and total budget must match.

Select using validation only. For expensive studies, search with a small seed
set, then compare a shortlist at full budget over a larger fixed seed set.
Evaluate the frozen choice on independent test data afterwards. The API
rejects declared test splits during tuning; it relies on prepared datasets
being correctly labeled and partitioned.
