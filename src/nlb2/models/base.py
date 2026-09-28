"""Base model and config registry for latent dynamics methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from nlb2.types import LossOutput, ModelOutput, observations_from_batch
from nlb2.utils.yaml import load_yaml


class OptimizationConfig(BaseModel):
    """Config block that selects an optimization strategy."""

    model_config = ConfigDict(extra="allow")

    name: str = "gradient"

    def kwargs(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("name", None)
        return data


class BaseModelConfig(BaseModel, ABC):
    """Pydantic model config that builds a PyTorch module."""

    model_config = ConfigDict(extra="forbid")

    registry: ClassVar[dict[str, type["BaseModelConfig"]]] = {}

    name: str
    objective: str = "negative_log_likelihood"
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)

    @abstractmethod
    def build(self, n_neurons: int, n_time: int) -> "BaseDynamicsModel":
        """Construct the model for runtime data dimensions."""

    @classmethod
    def register(cls, config_cls: type["BaseModelConfig"]) -> type["BaseModelConfig"]:
        name = config_cls.model_fields["name"].default
        if not isinstance(name, str):
            raise ValueError(f"{config_cls.__name__} must define a string default name.")
        cls.registry[name] = config_cls
        return config_cls

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BaseModelConfig":
        name = data.get("name")
        if name not in cls.registry:
            known = ", ".join(sorted(cls.registry)) or "<none>"
            raise KeyError(f"Unknown model config '{name}'. Registered models: {known}.")
        return cls.registry[name].model_validate(data)


class BaseDynamicsModel(nn.Module, ABC):
    """Base class for models taking `(batch, time, neurons)` tensors."""

    objective: str

    @abstractmethod
    def forward(self, x: Tensor) -> ModelOutput:
        """Run model inference on observations."""

    @abstractmethod
    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        """Compute the model objective for a batch and forward output."""

    def predict_rates(self, x: Tensor) -> Tensor:
        output = self.forward(x)
        if output.rates is not None:
            return output.rates
        if output.reconstruction is not None:
            return output.reconstruction
        raise RuntimeError(f"{type(self).__name__} did not return rates or reconstruction.")

    def evaluation_adapter(self, task: str) -> Any | None:
        """Return a model-specific evaluation adapter for a task, if needed."""

        del task
        return None

    def fit_training_data(self, loader: Any, *, device: torch.device) -> None:
        """Fit data-derived state; library_fit calls this inside its training epoch."""

        del loader, device

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            try:
                return next(self.buffers()).device
            except StopIteration:
                return torch.device("cpu")


class EnsembleDynamicsModel(BaseDynamicsModel):
    """Average predictions and member objectives from compatible dynamics models."""

    def __init__(
        self,
        members: list[BaseDynamicsModel],
        objective: str | None = None,
    ) -> None:
        super().__init__()
        if len(members) < 1:
            raise ValueError("EnsembleDynamicsModel requires at least one member.")
        self.members = nn.ModuleList(members)
        self.objective = objective or f"ensemble_{members[0].objective}"

    def forward(self, x: Tensor) -> ModelOutput:
        member_outputs = [member(x) for member in self.members]
        units = {output.rates_unit for output in member_outputs}
        if len(units) != 1:
            raise ValueError("Ensemble members must return rates in the same units.")
        extras = {"ensemble_size": len(self.members), "member_outputs": member_outputs}
        full_rates = _mean_optional_tensor(
            [output.extras.get("full_rates") for output in member_outputs]
        )
        full_unit = None
        if full_rates is not None:
            full_units = {output.full_rates_unit for output in member_outputs}
            if len(full_units) != 1 or None in full_units:
                raise ValueError("Ensemble full rates must declare matching units.")
            extras["full_rates"] = full_rates
            full_unit = member_outputs[0].full_rates_unit
        return ModelOutput(
            rates=_mean_optional_tensor([output.rates for output in member_outputs]),
            latents=_mean_optional_tensor([output.latents for output in member_outputs]),
            reconstruction=_mean_optional_tensor(
                [output.reconstruction for output in member_outputs]
            ),
            extras=extras,
            rates_unit=member_outputs[0].rates_unit,
            full_rates_unit=full_unit,
        )

    def fit_training_data(self, loader: Any, *, device: torch.device) -> None:
        for member in self.members:
            member.fit_training_data(loader, device=device)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        member_outputs = output.extras.get("member_outputs")
        if member_outputs is None:
            member_outputs = [member(observations_from_batch(batch)) for member in self.members]
        if len(member_outputs) != len(self.members):
            raise ValueError(
                "Ensemble loss received "
                f"{len(member_outputs)} member outputs for {len(self.members)} members."
            )

        losses = [
            member.loss(batch, member_output, epoch=epoch)
            for member, member_output in zip(self.members, member_outputs)
        ]
        total = torch.stack([loss.total for loss in losses]).mean()
        named_terms = {}
        keys = set().union(*(loss.named_terms.keys() for loss in losses))
        for key in keys:
            values = [
                loss.named_terms[key]
                for loss in losses
                if key in loss.named_terms
            ]
            named_terms[key] = _mean_loss_term(values, total)
        named_terms["ensemble_size"] = float(len(self.members))
        return LossOutput(total=total, named_terms=named_terms, objective=self.objective)

    def predict_rates(self, x: Tensor) -> Tensor:
        return torch.stack([member.predict_rates(x) for member in self.members]).mean(dim=0)

    def evaluation_adapter(self, task: str) -> Any | None:
        return self.members[0].evaluation_adapter(task)


def _mean_optional_tensor(values: list[Tensor | None]) -> Tensor | None:
    tensors = [value for value in values if value is not None]
    if len(tensors) != len(values):
        return None
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        return None
    return torch.stack(tensors).mean(dim=0)


def _mean_loss_term(values: list[Tensor | float], reference: Tensor) -> Tensor:
    tensors = [
        value
        if isinstance(value, Tensor)
        else torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
        for value in values
    ]
    return torch.stack(tensors).mean()


def load_model_config(path: str) -> BaseModelConfig:
    """Load a model config from YAML."""

    from nlb2 import models as _models  # noqa: F401

    data = load_yaml(path)
    model_data = data["model"] if "model" in data else data
    return BaseModelConfig.from_dict(model_data)
