"""Serializable tuning recipes; importing this module does not import Ray."""

from __future__ import annotations

from copy import deepcopy
import inspect
import math
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nlb2.config import ExperimentConfig, SelectionConfig, experiment_config_from_dict
from nlb2.experiment import experiment_config_to_dict
from nlb2.training.strategies import build_strategy
from nlb2.utils.yaml import load_yaml


class SearchParameter(BaseModel):
    """A scalar distribution. randint's upper bound is exclusive."""

    model_config = ConfigDict(extra="forbid")
    distribution: Literal["choice", "uniform", "loguniform", "randint"]
    choices: Optional[list[Any]] = None
    low: Optional[float] = None
    high: Optional[float] = None

    @model_validator(mode="after")
    def validate_domain(self):
        if self.distribution == "choice":
            if not self.choices or self.low is not None or self.high is not None:
                raise ValueError("choice requires nonempty choices and no bounds.")
            for value in self.choices:
                if not isinstance(value, (str, int, float, bool, type(None))):
                    raise ValueError("Choices must be JSON scalars; use parameter_sets for coupled fields.")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Choices must be finite.")
        else:
            if self.choices is not None or self.low is None or self.high is None:
                raise ValueError("Numeric distributions require low and high, and no choices.")
            if not math.isfinite(self.low) or not math.isfinite(self.high) or self.low >= self.high:
                raise ValueError("Distribution bounds must be finite and low < high.")
            if self.distribution == "loguniform" and self.low <= 0:
                raise ValueError("loguniform requires low > 0.")
            if self.distribution == "randint" and (not self.low.is_integer() or not self.high.is_integer()):
                raise ValueError("randint requires integer bounds.")
        return self

    def contains(self, value: Any) -> bool:
        if self.distribution == "choice":
            return value in self.choices
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        if self.distribution == "randint":
            return float(value).is_integer() and self.low <= value < self.high
        return self.low <= value <= self.high

    def example(self) -> Any:
        if self.distribution == "choice":
            return self.choices[0]
        return int(self.low) if self.distribution == "randint" else self.low


class TrialResources(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cpu: int = Field(default=1, ge=1)
    gpu: int = Field(default=0, ge=0, le=1)


def canonical_config(config: ExperimentConfig) -> dict[str, Any]:
    payload = experiment_config_to_dict(config)
    payload["trainer"]["batch_size"] = payload.pop("batch_size")
    # Strategy kwargs are extensible; materialize their defaults before validating paths.
    strategy = build_strategy(config.model.optimization)
    for name, parameter in inspect.signature(type(strategy).__init__).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            payload["model"]["optimization"].setdefault(name, parameter.default)
    return payload


class StudyConfig(BaseModel):
    """One fixed-data, fixed-budget search over ordinary experiment configs.

    Each Ray trial evaluates a candidate on every training seed. Only the mean
    of the completed seed runs is submitted to the search algorithm.
    """

    model_config = ConfigDict(extra="forbid")
    base: dict[str, Any]
    objective: SelectionConfig
    search_space: dict[str, SearchParameter] = Field(default_factory=dict)
    parameter_sets: list[dict[str, Any]] = Field(default_factory=list)
    algorithm: Literal["random", "tpe"] = "random"
    scheduler: Literal["fifo"] = "fifo"
    num_samples: int = Field(default=20, ge=1)
    include_base: bool = True
    points_to_evaluate: list[dict[str, Any]] = Field(default_factory=list)
    training_seeds: list[int] = Field(default_factory=lambda: [0], min_length=1)
    search_seed: int = Field(default=0, ge=0, lt=2**32)
    evaluation_seed: int = Field(default=10000, ge=0, lt=2**32)
    evaluation_interval: Optional[int] = Field(default=None, ge=1)
    resources: TrialResources = Field(default_factory=TrialResources)
    max_concurrent_trials: int = Field(default=1, ge=1)
    time_budget_s: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)
    output_dir: str = "studies"
    name: Optional[str] = None

    @field_validator("base", mode="before")
    @classmethod
    def resolve_base(cls, value):
        config = value if isinstance(value, ExperimentConfig) else experiment_config_from_dict(value)
        return canonical_config(config)

    @field_validator("training_seeds")
    @classmethod
    def validate_seeds(cls, values):
        if len(set(values)) != len(values) or any(seed < 0 or seed >= 2**32 for seed in values):
            raise ValueError("Training seeds must be distinct integers in [0, 2**32).")
        return values

    @model_validator(mode="after")
    def validate_study(self):
        if self.base["dataset"].get("split", "val") != "val":
            raise ValueError("Tuning requires validation data; test splits cannot select hyperparameters.")
        if self.base["dataset"].get("valid_split", "valid") not in {"val", "valid", "validation"}:
            raise ValueError("Tuning requires a validation split, not a test split.")
        if self.base["model"].get("train_source") in {"nwb", "mat"}:
            raise ValueError("Tuning uses Experiment; prepare a DataModule-compatible MINT dataset first.")
        trainer = self.base["trainer"]
        if trainer["epochs"] < 0 or trainer["batch_size"] < 1 or trainer["live_eval_interval"] < 0:
            raise ValueError("Invalid experiment training budget, batch size, or evaluation interval.")
        interval = self.evaluation_interval or trainer["live_eval_interval"]
        if self.objective.checkpoint == "best" and trainer["epochs"] > 0 and interval == 0:
            raise ValueError("Best checkpoint selection requires base.trainer.live_eval_interval > 0.")
        coupled = set(self.parameter_sets[0]) if self.parameter_sets else set()
        if any(set(point) != coupled for point in self.parameter_sets):
            raise ValueError("Every parameter set must specify the same coupled fields.")
        if coupled & self.search_space.keys():
            raise ValueError("parameter_sets and search_space must not overlap.")
        paths = set(self.search_space) | coupled
        if not paths:
            raise ValueError("A study requires a search_space or parameter_sets.")
        for path in paths:
            validate_parameter_path(path)
            get_path(self.base, path)
        for path in paths:
            if any(other.startswith(path + ".") for other in paths):
                raise ValueError("Search paths cannot overlap a parent field.")
        example = {key: domain.example() for key, domain in self.search_space.items()}
        self.materialize({**example, **(self.parameter_sets[0] if self.parameter_sets else {})}, self.training_seeds[0])
        for point in self.initial_points():
            self.validate_candidate(point)
        if len(self.initial_points()) > self.num_samples:
            raise ValueError("num_samples includes the base and all initial points.")
        return self

    def initial_points(self) -> list[dict[str, Any]]:
        paths = set(self.search_space)
        if self.parameter_sets:
            paths.update(self.parameter_sets[0])
        points = deepcopy(self.points_to_evaluate)
        if self.include_base:
            points.insert(0, {path: get_path(self.base, path) for path in sorted(paths)})
        return points

    def validate_candidate(self, values: dict[str, Any]) -> None:
        expected = set(self.search_space) | (set(self.parameter_sets[0]) if self.parameter_sets else set())
        if set(values) != expected:
            raise ValueError(f"Candidate must specify exactly {sorted(expected)}.")
        for path, domain in self.search_space.items():
            if not domain.contains(values[path]):
                raise ValueError(f"Candidate {path}={values[path]!r} is outside its search space.")
        if self.parameter_sets and not any(all(values[key] == value for key, value in point.items()) for point in self.parameter_sets):
            raise ValueError("Candidate does not match a declared parameter set.")
        self.materialize(values, self.training_seeds[0])

    def materialize(self, values: dict[str, Any], seed: int) -> ExperimentConfig:
        payload = deepcopy(self.base)
        for path, value in values.items():
            validate_parameter_path(path)
            set_path(payload, path, value)
        payload["selection"] = self.objective.model_dump()
        if self.evaluation_interval is not None:
            payload["trainer"]["live_eval_interval"] = self.evaluation_interval
        payload["experiment"].update(training_seed=seed, evaluation_seed=self.evaluation_seed)
        # These model-specific generators are independent of torch.manual_seed.
        for field in ("init_seed", "lfads_seed"):
            if field in payload["model"]:
                payload["model"][field] = seed
        config = experiment_config_from_dict(payload)
        if config.batch_size < 1:
            raise ValueError("Batch size must be positive.")
        build_strategy(config.model.optimization)
        return config


def validate_parameter_path(path: str) -> None:
    if not (path.startswith("model.") or path.startswith("preprocessing.") or path == "trainer.batch_size"):
        raise ValueError(f"Cannot tune {path!r}; data, seeds, resources, and training budgets are fixed by the study.")
    if path in {"model.name", "model.optimization", "model.optimization.name", "model.init_seed", "model.lfads_seed"}:
        raise ValueError(f"Cannot tune reserved field {path!r}.")
    if path.endswith("_path"):
        raise ValueError("Input files are fixed study inputs and cannot be search parameters.")


def _parent(payload: dict, path: str):
    parts = path.split(".")
    node = payload
    try:
        for part in parts[:-1]:
            node = node[int(part)] if isinstance(node, list) else node[part]
        key = int(parts[-1]) if isinstance(node, list) else parts[-1]
        node[key]  # Never silently introduce misspelled fields.
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown config path {path!r}.") from exc
    return node, key


def get_path(payload: dict, path: str) -> Any:
    node, key = _parent(payload, path)
    return node[key]


def set_path(payload: dict, path: str, value: Any) -> None:
    node, key = _parent(payload, path)
    node[key] = deepcopy(value)


def load_study_config(path: str | Path) -> StudyConfig:
    """Resolve the base YAML relative to the study file; data paths remain caller-relative."""
    path = Path(path).resolve()
    payload = load_yaml(path)
    if isinstance(payload.get("base"), str):
        base = Path(payload["base"])
        payload["base"] = load_yaml(base if base.is_absolute() else path.parent / base)
    return StudyConfig.model_validate(payload)
