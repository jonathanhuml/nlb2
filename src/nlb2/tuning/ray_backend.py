"""Ray-specific execution. Public configs, models, and trainers stay Ray-free."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from nlb2.experiment import Experiment, ExperimentReport, _write_json
from nlb2.tuning.config import StudyConfig, canonical_config


def require_ray():
    try:
        from ray import tune
        return tune
    except ImportError as exc:
        raise ImportError('Tuning requires optional dependencies: pip install "ray[tune]>=2.43,<3" "optuna>=3,<5"') from exc


def parameter_space(config: StudyConfig):
    tune = require_ray()
    space = {}
    for key, domain in config.search_space.items():
        if domain.distribution == "choice":
            space[key] = tune.choice(domain.choices)
        elif domain.distribution == "randint":
            space[key] = tune.randint(int(domain.low), int(domain.high))
        else:
            space[key] = getattr(tune, domain.distribution)(domain.low, domain.high)
    if config.parameter_sets:
        space["_parameter_set"] = tune.choice(list(range(len(config.parameter_sets))))
    return space


def decode_parameters(config: StudyConfig, sampled: dict) -> dict:
    values = dict(sampled)
    if config.parameter_sets:
        values.update(config.parameter_sets[int(values.pop("_parameter_set"))])
    return values


def _initial_points(config: StudyConfig) -> list[dict]:
    points = []
    for point in config.initial_points():
        encoded = {key: point[key] for key in config.search_space}
        if config.parameter_sets:
            encoded["_parameter_set"] = next(
                index for index, values in enumerate(config.parameter_sets)
                if all(point[key] == value for key, value in values.items())
            )
        points.append(encoded)
    return points


def _search_algorithm(config: StudyConfig):
    if config.algorithm == "random":
        from ray.tune.search.basic_variant import BasicVariantGenerator
        return BasicVariantGenerator(points_to_evaluate=_initial_points(config), random_state=config.search_seed,
                                     max_concurrent=config.max_concurrent_trials)
    try:
        from ray.tune.search.optuna import OptunaSearch
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError('TPE requires Optuna: pip install "optuna>=3,<5"') from exc

    class CompletedOptunaSearch(OptunaSearch):
        # Epoch/seed checkpoints are progress reports, not candidate objectives.
        def on_trial_result(self, trial_id, result):
            pass

        def on_trial_complete(self, trial_id, result=None, error=False):
            if error or not result or not result.get("candidate_complete"):
                result = {"score": None}
                error = True
            return super().on_trial_complete(trial_id, result=result, error=error)

    return CompletedOptunaSearch(
        metric="score", mode=config.objective.mode,
        sampler=TPESampler(seed=config.search_seed), points_to_evaluate=_initial_points(config),
    )


def _train_candidate(sampled: dict, *, study: dict) -> None:
    tune = require_ray()
    config = StudyConfig.model_validate(study)
    values = decode_parameters(config, sampled)
    config.validate_candidate(values)
    torch.set_num_threads(config.resources.cpu)
    with tempfile.TemporaryDirectory(prefix="nlb2_trial_") as temporary:
        work = Path(temporary)
        checkpoint = tune.get_checkpoint()
        if checkpoint is not None:
            with checkpoint.as_directory() as restored:
                shutil.copytree(restored, work, dirs_exist_ok=True)
        state_path = work / "candidate.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            "parameters": values, "seeds": [], "resume_from": None, "active_seed": None,
            "candidate_complete": False,
        }

        def publish(report: ExperimentReport | None = None):
            _write_json(state_path, state)
            metrics = {
                "score": state.get("score"), "candidate_complete": state["candidate_complete"],
                "completed_seeds": len(state["seeds"]),
            }
            if report is not None:
                metrics.update(epoch=report.epoch, seed=state["active_seed"],
                               validation_score=report.metrics[config.objective.metric],
                               elapsed_seconds=report.elapsed_seconds)
            tune.report(metrics, checkpoint=tune.Checkpoint.from_directory(str(work)))

        def observe(report: ExperimentReport):
            if report.checkpoint_path is not None:
                state["resume_from"] = str(report.checkpoint_path.relative_to(work))
            if not report.final:
                publish(report)

        try:
            for seed in config.training_seeds:
                if any(row["seed"] == seed for row in state["seeds"]):
                    continue
                candidate = config.materialize(values, seed)
                candidate.output_dir = str(work)
                candidate.run_name = f"seed_{seed}"
                candidate.save_plots = False
                candidate.save_training_state = True
                candidate.trainer.device = "cuda" if config.resources.gpu else "cpu"
                resume_from = work / state["resume_from"] if state["resume_from"] and state["active_seed"] == seed else None
                state["active_seed"] = seed
                # Keep the config even if model construction fails before reporting.
                _write_json(work / f"seed_{seed}_config.json", canonical_config(candidate))
                result = Experiment(candidate).run(callback=observe, resume_from=resume_from)
                score = result.metrics[config.objective.metric]
                if not math.isfinite(score) or result.completed_epochs != candidate.trainer.epochs:
                    raise ValueError("Only finite, full-budget seed runs can complete a candidate.")
                state["seeds"].append({
                    "seed": seed, "score": score, "metrics": result.metrics,
                    "selected_epoch": result.selected_epoch, "completed_epochs": result.completed_epochs,
                    "elapsed_seconds": result.elapsed_seconds,
                    "run_dir": str(result.run_dir.relative_to(work)),
                    "model_path": str(result.model_path.relative_to(work)),
                })
                state["resume_from"] = None
                if len(state["seeds"]) < len(config.training_seeds):
                    publish()
            scores = [row["score"] for row in state["seeds"]]
            state.update(candidate_complete=True, score=float(np.mean(scores)),
                         score_std=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                         elapsed_seconds=sum(row["elapsed_seconds"] for row in state["seeds"]))
            publish()
        except Exception as exc:
            state["error"] = repr(exc)
            publish()
            raise


def run_study(config: StudyConfig, directory: Path, *, resume: bool):
    from nlb2.tuning.study import write_results

    tune = require_ray()
    trainable = tune.with_resources(
        tune.with_parameters(_train_candidate, study=config.model_dump(mode="json")),
        {"cpu": config.resources.cpu, "gpu": config.resources.gpu},
    )
    storage = directory / "ray"
    if resume:
        tuner = tune.Tuner.restore(str(storage), trainable=trainable, resume_errored=True)
    else:
        tuner = tune.Tuner(
            trainable, param_space=parameter_space(config),
            tune_config=tune.TuneConfig(
                metric="score", mode=config.objective.mode,
                search_alg=_search_algorithm(config), num_samples=config.num_samples,
                max_concurrent_trials=config.max_concurrent_trials if config.algorithm == "tpe" else None,
                time_budget_s=config.time_budget_s,
            ),
            run_config=tune.RunConfig(
                name="ray", storage_path=str(directory), verbose=0,
                checkpoint_config=tune.CheckpointConfig(num_to_keep=1),
                failure_config=tune.FailureConfig(max_failures=0),
            ),
        )
    results = tuner.fit()
    trials = []
    for index, result in enumerate(results):
        trial_id = result.metrics.get("trial_id", f"trial_{index:04d}")
        trial = {"trial_id": trial_id, "parameters": decode_parameters(config, result.config),
                 "status": "failed" if result.error else "incomplete", "score": None,
                 "score_std": None, "elapsed_seconds": result.metrics.get("time_total_s"),
                 "error": str(result.error) if result.error else None, "seeds": []}
        if result.checkpoint is not None:
            target = directory / "trials" / trial_id
            with result.checkpoint.as_directory() as source:
                shutil.copytree(source, target, dirs_exist_ok=True)
            # Worker scratch paths are not useful in the portable experiment exports.
            for path in [*target.glob("seed_*_config.json"), *target.glob("*/config.json")]:
                payload = json.loads(path.read_text())
                payload["experiment"]["output_dir"] = str(directory / "replays")
                _write_json(path, payload)
            for path in target.glob("*/status.json"):
                payload = json.loads(path.read_text())
                payload["run_dir"] = str(path.parent)
                _write_json(path, payload)
            state = json.loads((target / "candidate.json").read_text())
            for row in state["seeds"]:
                row["model_path"] = str((target / row["model_path"]).relative_to(directory))
                row["run_dir"] = str((target / row["run_dir"]).relative_to(directory))
            trial["seeds"] = state["seeds"]
            if not result.error and state["candidate_complete"] and len(state["seeds"]) == len(config.training_seeds) and math.isfinite(state["score"]):
                trial.update(status="complete", score=state["score"], score_std=state["score_std"],
                             elapsed_seconds=state["elapsed_seconds"])
        trials.append(trial)
    return write_results(config, directory, trials)
