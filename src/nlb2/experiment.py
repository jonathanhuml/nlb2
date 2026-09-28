"""Public experiment orchestration API."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable

import numpy as np
import torch

from nlb2.config import ExperimentConfig, load_experiment_config
from nlb2.data import DataModule
from nlb2.datasets.nlb import NLBDatasetConfig
from nlb2.metrics import EvaluationResult, evaluate_model
from nlb2.models.base import BaseDynamicsModel
from nlb2.nlb_eval import evaluate_model_nlb_submission
from nlb2.training import EpochReport, Trainer
from nlb2.training.strategies import build_strategy
from nlb2.training.checkpoint import (
    capture_rng, evaluation_rng, restore_rng, save_training_checkpoint,
    seed_training, validate_resume_config,
)
from nlb2.types import StepResult


@dataclass
class ExperimentReport:
    """A fresh task evaluation. Returning False from a callback stops training."""

    epoch: int
    metrics: dict[str, float]
    elapsed_seconds: float
    run_dir: Path
    checkpoint_path: Path | None = None
    final: bool = False


@dataclass
class ExperimentResult:
    """Artifacts produced by a completed experiment run."""

    run_dir: Path
    metrics: dict[str, float]
    history: list[EpochReport]
    config_path: Path
    history_path: Path
    metrics_path: Path
    model_path: Path
    report_path: Path
    predictions_path: Path | None = None
    plot_paths: dict[str, Path] = field(default_factory=dict)
    selected_epoch: int | None = None
    completed_epochs: int = 0
    elapsed_seconds: float = 0.0


class Experiment:
    """Build data, model, training, metrics, and run artifacts from one config."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.data = DataModule(
            config=config.dataset,
            batch_size=config.batch_size,
            preprocessing=config.preprocessing,
        )
        self.model: BaseDynamicsModel | None = None
        self.trainer = Trainer(config.trainer)
        self.result: ExperimentResult | None = None

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Experiment":
        """Create an experiment from a YAML recipe or saved JSON configuration."""

        return cls(load_experiment_config(str(path)))

    def build_model(self) -> BaseDynamicsModel:
        """Instantiate the configured model for the prepared data dimensions."""

        if self.data.train_dataset is None:
            self.data.setup()
        build_from_data = getattr(self.config.model, "build_from_data", None)
        if callable(build_from_data):
            self.model = build_from_data(self.data)
        else:
            self.model = self.config.model.build(
                n_neurons=self.data.n_neurons,
                n_time=self.data.n_time,
            )
        return self.model

    def run(
        self,
        *,
        callback: Callable[[ExperimentReport], bool | None] | None = None,
        resume_from: str | Path | None = None,
    ) -> ExperimentResult:
        """Train the model, evaluate it, and write a self-contained run folder."""

        is_test_split = isinstance(self.config.dataset, NLBDatasetConfig) and self.config.dataset.split == "test"
        if is_test_split and (self.config.trainer.live_eval_interval > 0 or self.config.selection is not None):
            raise ValueError(
                "NLB checkpoint selection requires split='val'. Test targets may only "
                "be scored after training, with live_eval_interval=0."
            )
        if self.config.trainer.epochs < 0 or self.config.trainer.live_eval_interval < 0:
            raise ValueError("Epochs and live_eval_interval must be nonnegative.")
        if self.config.selection is not None and self.config.selection.checkpoint == "best":
            if self.config.trainer.epochs > 0 and self.config.trainer.live_eval_interval == 0:
                raise ValueError("Best checkpoint selection requires live_eval_interval > 0.")
        started = time.monotonic()
        payload = experiment_config_to_dict(self.config)
        saved = None
        if resume_from is not None:
            saved = torch.load(resume_from, map_location="cpu", weights_only=True)
            validate_resume_config(saved["config"], payload)
            if not 0 <= saved["epoch"] <= self.config.trainer.epochs:
                raise ValueError("Checkpoint epoch is outside the training budget.")
            if not saved["resumable"]:
                raise ValueError("This optimization strategy does not support training-state restoration.")
        self._set_seeds()
        self.trainer.history = []
        self.data.setup()
        model = self.build_model()
        if saved is not None:
            model.load_state_dict(saved["model"])
        strategy = build_strategy(self.config.model.optimization)
        train_loader = self.data.train_loader(
            shuffle=not bool(getattr(strategy, "requires_ordered_training_data", False))
        )
        valid_loader = self.data.valid_loader()

        run_dir = self._make_run_dir()
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.csv"
        live_metrics_path = run_dir / "live_metrics.json"
        status_path = run_dir / "status.json"
        best_model_path = run_dir / "best_model.pt"
        best_metrics_path = run_dir / "best_metrics.json"
        checkpoint_path = run_dir / "training_state.pt"
        _write_json(config_path, experiment_config_to_dict(self.config))
        _write_json(
            status_path,
            {
                "status": "running",
                "epochs": self.config.trainer.epochs,
                "run_dir": run_dir,
            },
        )
        best_eval: dict[str, Any] | None = None if saved is None else saved["best_eval"]
        best_evaluation = None if saved is None else _unpack_evaluation(saved["best_evaluation"])
        if saved is not None:
            self.trainer.history = [_unpack_epoch(item) for item in saved["history"]]
            if saved["best_model"] is not None:
                torch.save(saved["best_model"], best_model_path)
                _write_json(best_metrics_path, best_eval)
        last_eval: dict[str, Any] | None = None
        stopped = False
        elapsed_before = 0.0 if saved is None else saved["elapsed_seconds"]

        def elapsed() -> float:
            return elapsed_before + time.monotonic() - started

        def evaluate() -> EvaluationResult:
            with evaluation_rng(self.config.evaluation_seed, self.config.trainer.device):
                result = evaluate_model(
                    model=model, loader=valid_loader, device=self.config.trainer.device,
                    train_loader=self.data.train_loader(shuffle=False),
                )
            self._selection_score(result.metrics)
            return result

        def checkpoint(epoch: int) -> None:
            if not self.config.save_training_state:
                return
            strategy_state = strategy.state_dict()
            save_training_checkpoint(checkpoint_path, {
                "version": 1, "epoch": epoch, "config": payload,
                "model": model.state_dict(), "strategy": strategy_state,
                "resumable": bool(strategy_state) or strategy.name in {"em", "library_fit", "inference_only"},
                "rng": capture_rng(self.config.trainer.device),
                "history": [asdict(item) for item in self.trainer.history],
                "best_eval": best_eval, "best_evaluation": _pack_evaluation(best_evaluation),
                "best_model": torch.load(best_model_path, map_location="cpu", weights_only=True)
                    if best_model_path.exists() else None,
                "elapsed_seconds": elapsed(),
            })

        def epoch_callback(report: EpochReport) -> bool | None:
            nonlocal best_eval, best_evaluation, last_eval, stopped
            evaluation_metrics: dict[str, float] | None = None
            interval = int(getattr(self.config.trainer, "live_eval_interval", 0))
            should_evaluate = interval > 0 and (
                (report.epoch + 1) % interval == 0
                or report.epoch + 1 == self.config.trainer.epochs
            )
            if should_evaluate:
                evaluation = evaluate()
                evaluation_metrics = evaluation.metrics
                last_eval = {
                    "epoch": report.epoch + 1,
                    "metrics": evaluation_metrics,
                }
                report.metrics.update(
                    {f"eval/{key}": value for key, value in evaluation_metrics.items()}
                )
                live_score = self._selection_score(evaluation_metrics)
                if live_score is not None and (
                    best_eval is None or _is_better_live_score(live_score, best_eval["score"])
                ):
                    best_eval = {
                        "epoch": report.epoch + 1,
                        "metrics": evaluation_metrics,
                        "score": live_score,
                    }
                    torch.save(model.state_dict(), best_model_path)
                    _write_json(best_metrics_path, best_eval)
                    if self.config.selection is not None and self.config.selection.checkpoint == "best":
                        best_evaluation = evaluation
            _write_history(history_path, self.trainer.history)
            _write_live_metrics(
                live_metrics_path,
                report=report,
                epochs=self.config.trainer.epochs,
                evaluation_metrics=evaluation_metrics,
                last_eval=last_eval,
                best_eval=best_eval,
            )
            _write_json(
                status_path,
                {
                    "status": "running",
                    "epoch": report.epoch + 1,
                    "epochs": self.config.trainer.epochs,
                    "run_dir": run_dir,
                    "last_eval": last_eval,
                    "best_eval": best_eval,
                },
            )
            print(_format_epoch_progress(report, evaluation_metrics), flush=True)
            if should_evaluate or report.epoch + 1 == self.config.trainer.epochs:
                checkpoint(report.epoch + 1)
            if callback is not None and evaluation_metrics is not None:
                stopped = callback(ExperimentReport(
                    epoch=report.epoch + 1, metrics=dict(evaluation_metrics),
                    elapsed_seconds=elapsed(), run_dir=run_dir,
                    checkpoint_path=checkpoint_path if checkpoint_path.exists() else None,
                )) is False
                if stopped:
                    return False
            return None

        try:
            history = self.trainer.fit(
                model=model,
                strategy=strategy,
                train_loader=train_loader,
                valid_loader=None if is_test_split else valid_loader,
                epoch_callback=epoch_callback,
                start_epoch=0 if saved is None else saved["epoch"],
                strategy_state=None if saved is None else saved["strategy"],
                on_ready=None if saved is None else lambda: restore_rng(saved["rng"], self.config.trainer.device),
            )
            completed_epochs = history[-1].epoch + 1 if history else 0
            checkpoint(completed_epochs)
            selected_epoch = completed_epochs
            if self.config.selection is not None and self.config.selection.checkpoint == "best" and best_eval:
                model.load_state_dict(torch.load(best_model_path, map_location=self.config.trainer.device, weights_only=True))
                evaluation = best_evaluation
                selected_epoch = best_eval["epoch"]
            else:
                evaluation = evaluate()
            result = self._write_artifacts(run_dir, model, history, evaluation)
            result.selected_epoch = selected_epoch
            result.completed_epochs = completed_epochs
            result.elapsed_seconds = elapsed()
            _write_json(run_dir / "selection.json", {
                "epoch": selected_epoch, "completed_epochs": completed_epochs,
                "selection": None if self.config.selection is None else self.config.selection.model_dump(),
                "metrics": result.metrics, "model_path": "model.pt", "elapsed_seconds": result.elapsed_seconds,
            })
            _write_json(
                status_path,
                {
                    "status": "stopped" if stopped else "complete",
                    "epoch": completed_epochs,
                    "epochs": self.config.trainer.epochs,
                    "run_dir": run_dir,
                    "metrics": result.metrics,
                    "last_eval": last_eval,
                    "best_eval": best_eval,
                },
            )
            if callback is not None:
                callback(ExperimentReport(
                    epoch=selected_epoch, metrics=dict(result.metrics),
                    elapsed_seconds=result.elapsed_seconds, run_dir=run_dir,
                    checkpoint_path=checkpoint_path if checkpoint_path.exists() else None, final=True,
                ))
            self.result = result
            return result
        except Exception as exc:
            _write_json(
                status_path,
                {
                    "status": "failed",
                    "error": repr(exc),
                    "run_dir": run_dir,
                },
            )
            raise

    def _set_seeds(self) -> None:
        seed = self.config.training_seed
        if seed is None:
            seed = getattr(self.config.dataset, "seed", None)
        if seed is None:
            return
        seed_training(int(seed))

    def _selection_score(self, metrics: dict[str, float]) -> dict[str, Any] | None:
        selection = self.config.selection
        if selection is None:
            return _live_eval_score(metrics)
        if selection.metric not in metrics or not math.isfinite(metrics[selection.metric]):
            raise ValueError(f"Selection metric {selection.metric!r} must be present and finite; got {metrics}.")
        return {"name": selection.metric, "value": metrics[selection.metric], "mode": selection.mode}

    def _make_run_dir(self) -> Path:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        default_name = f"{timestamp}_{self.config.dataset.name}_{self.config.model.name}"
        run_name = self.config.run_name or default_name
        return _create_unique_dir(output_dir / _slugify(run_name))

    def _write_artifacts(
        self,
        run_dir: Path,
        model: BaseDynamicsModel,
        history: list[EpochReport],
        evaluation: EvaluationResult,
    ) -> ExperimentResult:
        config_path = run_dir / "config.json"
        history_path = run_dir / "history.csv"
        metrics_path = run_dir / "metrics.json"
        model_path = run_dir / "model.pt"
        report_path = run_dir / "report.md"
        predictions_path = run_dir / "predictions.npz" if self.config.save_predictions else None
        metrics = dict(evaluation.metrics)
        with evaluation_rng(self.config.evaluation_seed, self.config.trainer.device):
            nlb_full = evaluate_model_nlb_submission(
                model=model,
                train_loader=self.data.train_loader(shuffle=False),
                valid_loader=self.data.valid_loader(),
                dataset_config=self.config.dataset,
                device=self.config.trainer.device,
                output_dir=run_dir,
            )
        if nlb_full is not None:
            metrics.update(nlb_full.metrics)

        _write_json(config_path, experiment_config_to_dict(self.config))
        _write_history(history_path, history)
        _write_json(metrics_path, _json_ready(metrics))
        torch.save(model.state_dict(), model_path)
        if predictions_path is not None:
            _write_predictions(predictions_path, evaluation)
        plot_paths = _write_history_plots(run_dir, history) if self.config.save_plots else {}
        _write_report(report_path, self.config, history, metrics, plot_paths)

        return ExperimentResult(
            run_dir=run_dir,
            metrics=metrics,
            history=history,
            config_path=config_path,
            history_path=history_path,
            metrics_path=metrics_path,
            model_path=model_path,
            report_path=report_path,
            predictions_path=predictions_path,
            plot_paths=plot_paths,
        )


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Return a JSON-serializable experiment config snapshot."""

    return {
        "dataset": config.dataset.model_dump(mode="json"),
        "model": config.model.model_dump(mode="json"),
        "preprocessing": config.preprocessing.model_dump(mode="json"),
        "trainer": asdict(config.trainer),
        "batch_size": config.batch_size,
        "selection": None if config.selection is None else config.selection.model_dump(mode="json"),
        "experiment": {
            "output_dir": config.output_dir,
            "run_name": config.run_name,
            "save_predictions": config.save_predictions,
            "save_plots": config.save_plots,
            "training_seed": config.training_seed,
            "evaluation_seed": config.evaluation_seed,
            "save_training_state": config.save_training_state,
        },
    }


def _pack_evaluation(evaluation: EvaluationResult | None) -> dict | None:
    if evaluation is None:
        return None
    return {
        "metrics": evaluation.metrics,
        "predictions": {key: torch.as_tensor(value) for key, value in evaluation.predictions.items()},
        "targets": {key: torch.as_tensor(value) for key, value in evaluation.targets.items()},
    }


def _unpack_evaluation(state: dict | None) -> EvaluationResult | None:
    if state is None:
        return None
    return EvaluationResult(
        metrics=state["metrics"],
        predictions={key: value.cpu().numpy() for key, value in state["predictions"].items()},
        targets={key: value.cpu().numpy() for key, value in state["targets"].items()},
    )


def _unpack_epoch(state: dict) -> EpochReport:
    return EpochReport(
        epoch=state["epoch"], train=StepResult(**state["train"]),
        valid=None if state["valid"] is None else StepResult(**state["valid"]),
        seconds=state["seconds"], metrics=state["metrics"],
    )


def _write_history(path: Path, history: list[EpochReport]) -> None:
    fieldnames = [
        "epoch",
        "seconds",
        "train_loss",
        "valid_loss",
        "objective",
        "metrics",
        "train_metrics",
        "valid_metrics",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for report in history:
            writer.writerow(
                {
                    "epoch": report.epoch + 1,
                    "seconds": report.seconds,
                    "train_loss": report.train.loss,
                    "valid_loss": None if report.valid is None else report.valid.loss,
                    "objective": report.train.objective,
                    "metrics": json.dumps(_json_ready(report.metrics), sort_keys=True),
                    "train_metrics": json.dumps(
                        _json_ready(report.train.metrics),
                        sort_keys=True,
                    ),
                    "valid_metrics": json.dumps(
                        _json_ready({} if report.valid is None else report.valid.metrics),
                        sort_keys=True,
                    ),
                }
            )


def _write_live_metrics(
    path: Path,
    *,
    report: EpochReport,
    epochs: int,
    evaluation_metrics: dict[str, float] | None,
    last_eval: dict[str, Any] | None,
    best_eval: dict[str, Any] | None,
) -> None:
    _write_json(
        path,
        {
            "epoch": report.epoch + 1,
            "epochs": epochs,
            "seconds": report.seconds,
            "train_loss": report.train.loss,
            "valid_loss": None if report.valid is None else report.valid.loss,
            "objective": report.train.objective,
            "metrics": report.metrics,
            "train_metrics": report.train.metrics,
            "valid_metrics": {} if report.valid is None else report.valid.metrics,
            "evaluation_metrics": evaluation_metrics or {},
            "last_eval": last_eval,
            "best_eval": best_eval,
        },
    )


def _format_epoch_progress(
    report: EpochReport,
    evaluation_metrics: dict[str, float] | None,
) -> str:
    valid_loss = float("nan") if report.valid is None else report.valid.loss
    fields = [
        f"epoch={report.epoch + 1}",
        f"train_loss={report.train.loss:.6g}",
        f"valid_loss={valid_loss:.6g}",
    ]
    if evaluation_metrics:
        for key, value in sorted(evaluation_metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            fields.append(f"{key}={display}")
    return " ".join(fields)


def _live_eval_score(metrics: dict[str, float]) -> dict[str, float | str] | None:
    if "co_bps" in metrics and math.isfinite(metrics["co_bps"]):
        return {"name": "co_bps", "value": metrics["co_bps"], "mode": "max"}
    if "poisson_nll" in metrics and math.isfinite(metrics["poisson_nll"]):
        return {"name": "poisson_nll", "value": metrics["poisson_nll"], "mode": "min"}
    return None


def _is_better_live_score(
    score: dict[str, float | str],
    best_score: dict[str, float | str],
) -> bool:
    if score["name"] != best_score["name"] or score["mode"] != best_score["mode"]:
        return True
    value = float(score["value"])
    best_value = float(best_score["value"])
    if score["mode"] == "min":
        return value < best_value
    return value > best_value


def _write_history_plots(run_dir: Path, history: list[EpochReport]) -> dict[str, Path]:
    if not history:
        return {}

    _prepare_matplotlib_cache()
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from nlb2.plotting import plot_context, save_figure, style_axis

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray([report.epoch + 1 for report in history], dtype=np.int64)
    train_loss = np.asarray([report.train.loss for report in history], dtype=np.float64)
    valid_loss = np.asarray(
        [
            float("nan") if report.valid is None else report.valid.loss
            for report in history
        ],
        dtype=np.float64,
    )

    plot_paths: dict[str, Path] = {}
    train_test_path = plots_dir / "train_test_objective_curves.png"
    with plot_context(nrows=1, ncols=1, rel_width=0.86, height_scale=1.35):
        # Render saved artifacts without changing the caller's display backend.
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        _plot_finite_curve(ax, epochs, train_loss, label="train", linestyle="--")
        _plot_finite_curve(ax, epochs, valid_loss, label="test", linestyle="-")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train and Test Objective")
        style_axis(ax)
        ax.legend()
        save_figure(fig, train_test_path)
    plot_paths["train_test_objective"] = train_test_path

    if np.isfinite(valid_loss).any():
        test_path = plots_dir / "test_objective_curves.png"
        with plot_context(nrows=1, ncols=1, rel_width=0.86, height_scale=1.35):
            fig = Figure()
            FigureCanvasAgg(fig)
            ax = fig.subplots()
            _plot_finite_curve(ax, epochs, valid_loss, label="test", linestyle="-")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Test Objective")
            style_axis(ax)
            ax.legend()
            save_figure(fig, test_path)
        plot_paths["test_objective"] = test_path

    return plot_paths


def _plot_finite_curve(ax, epochs: np.ndarray, values: np.ndarray, *, label: str, linestyle: str) -> None:
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return
    ax.plot(epochs[finite], values[finite], label=label, linestyle=linestyle)


def _prepare_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir())
    mpl_dir = cache_root / "nlb2_matplotlib"
    xdg_dir = cache_root / "nlb2_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


def _write_predictions(path: Path, evaluation: EvaluationResult) -> None:
    arrays = {}
    arrays.update({f"pred_{key}": value for key, value in evaluation.predictions.items()})
    arrays.update({f"target_{key}": value for key, value in evaluation.targets.items()})
    np.savez_compressed(path, **arrays)


def _write_report(
    path: Path,
    config: ExperimentConfig,
    history: list[EpochReport],
    metrics: dict[str, float],
    plot_paths: dict[str, Path] | None = None,
) -> None:
    final = history[-1] if history else None
    lines = [
        "# NLB2 Experiment Report",
        "",
        f"- Dataset: `{config.dataset.name}`",
        f"- Model: `{config.model.name}`",
        f"- Epochs: `{config.trainer.epochs}`",
        f"- Batch size: `{config.batch_size}`",
        f"- Device: `{config.trainer.device}`",
        "",
        "## Final Training State",
        "",
    ]
    if final is None:
        lines.append("No epochs were run.")
    else:
        valid_loss = "nan" if final.valid is None else f"{final.valid.loss:.6g}"
        lines.extend(
            [
                f"- Train loss: `{final.train.loss:.6g}`",
                f"- Validation loss: `{valid_loss}`",
                f"- Objective: `{final.train.objective}`",
            ]
        )

    lines.extend(["", "## Evaluation Metrics", ""])
    if metrics:
        for key, value in sorted(metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            lines.append(f"- `{key}`: `{display}`")
    else:
        lines.append("No compatible evaluation metrics were available.")

    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for plot_path in plot_paths.values():
            lines.append(f"- `{plot_path.relative_to(path.parent)}`")

    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_ready(data), indent=2, sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _create_unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=False)
        return path

    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            candidate.mkdir(parents=False)
            return candidate
    raise RuntimeError(f"Could not create a unique run directory for {path}.")
