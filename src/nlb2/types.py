"""Shared typed outputs for models, losses, and training strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch
from torch import Tensor


@dataclass
class ModelOutput:
    """Standard model output container.

    All fields are optional because methods expose different scientific
    quantities. Downstream metrics can consume whichever fields a model provides.
    """

    rates: Tensor | None = None
    latents: Tensor | None = None
    reconstruction: Tensor | None = None
    distribution: Any | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    rates_unit: Literal["counts", "hz"] = "counts"
    full_rates_unit: Literal["counts", "hz"] | None = None

    def count_rates(self, dt: float | Tensor, *, full: bool = False) -> Tensor | None:
        """Return expected counts per bin, converting Hz exactly once.

        `extras['full_rates']` may contain unsliced neural predictions, but
        must declare its units independently of the main prediction slice.
        """

        rates = self.rates
        unit = self.rates_unit
        if full and self.extras.get("full_rates") is not None:
            rates = self.extras["full_rates"]
            unit = self.full_rates_unit
        if rates is None:
            return None
        if unit == "counts":
            return rates
        if unit == "hz":
            bin_width = torch.as_tensor(dt, dtype=rates.dtype, device=rates.device)
            if not bool((torch.isfinite(bin_width) & (bin_width > 0)).all()):
                raise ValueError("Converting Hz to counts requires a finite positive dt.")
            while bin_width.ndim < rates.ndim:
                bin_width = bin_width.unsqueeze(-1)
            if torch.broadcast_shapes(bin_width.shape, rates.shape) != rates.shape:
                raise ValueError("dt must broadcast to the rate prediction shape.")
            return rates * bin_width
        raise ValueError("Rate predictions must declare units as 'counts' or 'hz'.")


@dataclass
class LossOutput:
    """Standard loss container returned by model-specific objectives."""

    total: Tensor
    named_terms: Mapping[str, Tensor | float] = field(default_factory=dict)
    objective: str = "loss"


@dataclass
class StepResult:
    """Reporting contract returned by optimization strategies."""

    loss: float
    metrics: dict[str, float] = field(default_factory=dict)
    batch_size: int = 0
    objective: str = "loss"

    @classmethod
    def from_loss(cls, loss: LossOutput, batch_size: int) -> "StepResult":
        metrics: dict[str, float] = {}
        for key, value in loss.named_terms.items():
            if isinstance(value, Tensor):
                metrics[key] = float(value.detach().cpu())
            else:
                metrics[key] = float(value)
        return cls(
            loss=float(loss.total.detach().cpu()),
            metrics=metrics,
            batch_size=batch_size,
            objective=loss.objective,
        )


def observations_from_batch(batch: Tensor | Mapping[str, Tensor], key: str = "spikes") -> Tensor:
    """Extract `(batch, time, neurons)` observations from a trainer batch."""

    if isinstance(batch, Tensor):
        return batch
    if key not in batch:
        raise KeyError(f"Batch is missing observation key '{key}'.")
    return batch[key]


def move_batch_to_device(batch: Any, device: torch.device | str) -> Any:
    """Move tensors in a nested batch to a device."""

    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(value, device) for value in batch)
    return batch
