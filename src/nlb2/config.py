"""Experiment config loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nlb2.data import build_dataset_config
from nlb2.models.base import BaseModelConfig
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig
from nlb2.utils.yaml import load_yaml


class SelectionConfig(BaseModel):
    """Validation metric and checkpoint used to select a trained model."""

    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1)
    mode: Literal["min", "max"]
    checkpoint: Literal["last", "best"] = "last"


@dataclass
class ExperimentConfig:
    dataset: BaseModel
    model: BaseModelConfig
    trainer: TrainerConfig
    preprocessing: PreprocessingConfig
    batch_size: int = 32
    output_dir: str = "runs"
    run_name: str | None = None
    save_predictions: bool = True
    save_plots: bool = True
    training_seed: int | None = None
    evaluation_seed: int | None = None
    selection: SelectionConfig | None = None
    save_training_state: bool = False


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load an experiment from a YAML recipe or a saved JSON configuration."""

    return experiment_config_from_dict(load_yaml(path))


def experiment_config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Validate a resolved experiment, including legacy JSON snapshots."""

    from nlb2 import models as _models  # noqa: F401

    dataset_name = data["dataset"].get("name")

    trainer_data = dict(data.get("trainer", {}))
    batch_size = int(trainer_data.pop("batch_size", data.get("batch_size", 32)))
    if "batch_size" in data and int(data["batch_size"]) != batch_size:
        raise ValueError("Conflicting top-level and trainer.batch_size values.")
    experiment_data = data.get("experiment", {})
    return ExperimentConfig(
        dataset=build_dataset_config(dataset_name, data["dataset"]),
        model=BaseModelConfig.from_dict(data["model"]),
        trainer=TrainerConfig(**trainer_data),
        preprocessing=PreprocessingConfig.model_validate(data.get("preprocessing", {})),
        batch_size=batch_size,
        output_dir=str(experiment_data.get("output_dir", "runs")),
        run_name=experiment_data.get("run_name"),
        save_predictions=bool(experiment_data.get("save_predictions", True)),
        save_plots=bool(experiment_data.get("save_plots", True)),
        training_seed=experiment_data.get("training_seed"),
        evaluation_seed=experiment_data.get("evaluation_seed"),
        selection=SelectionConfig.model_validate(data["selection"]) if data.get("selection") else None,
        save_training_state=bool(experiment_data.get("save_training_state", False)),
    )
