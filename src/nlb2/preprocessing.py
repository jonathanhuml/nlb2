"""Observation preprocessing utilities for benchmark datasets."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
import scipy.signal.windows as signal
from scipy.signal import convolve
import torch
from torch import Tensor
from torch.utils.data import Dataset


class PreprocessingStepConfig(BaseModel):
    """One observation preprocessing step."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["smooth_firing_rate", "anscombe"] = "smooth_firing_rate"
    sampling_precision: float = 20.0
    kern_sd_ms: float = 50.0


class PreprocessingConfig(BaseModel):
    """Preprocessing config applied before model training/evaluation."""

    model_config = ConfigDict(extra="forbid")

    observations: list[PreprocessingStepConfig] = Field(default_factory=list)

    @field_validator("observations", mode="before")
    @classmethod
    def _coerce_observations(cls, value):
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


class PreprocessedDataset(Dataset):
    """Dataset wrapper that replaces `spikes` with preprocessed observations."""

    def __init__(
        self,
        dataset: Dataset,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.raw_spikes = getattr(dataset, "spikes", None)
        if self.raw_spikes is None:
            raise AttributeError("PreprocessedDataset requires a dataset with a `spikes` tensor.")
        self.spikes = apply_preprocessing(self.raw_spikes, self.preprocessing)

        for name in ["rates", "latents", "arrays", "config", "split"]:
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = dict(self.dataset[index])
        item["raw_spikes"] = item["spikes"]
        item["spikes"] = self.spikes[index]
        return item


def apply_preprocessing(x: Tensor, config: PreprocessingConfig | None = None) -> Tensor:
    """Apply configured observation preprocessing to `(trials, time, neurons)`."""

    config = config or PreprocessingConfig()
    out = x
    for step in config.observations:
        if step.name == "smooth_firing_rate":
            out = smooth_firing_rate(
                out,
                sampling_precision=step.sampling_precision,
                kern_sd_ms=step.kern_sd_ms,
            )
        elif step.name == "anscombe":
            out = 2.0 * torch.sqrt(out + 0.375)
        else:
            raise KeyError(f"Unknown preprocessing step '{step.name}'.")
    return out


def smooth_firing_rate(
    spike_trains: Tensor,
    sampling_precision: float = 20.0,
    kern_sd_ms: float = 50.0,
) -> Tensor:
    """Gaussian smoothing matching CASSM's `smooth_firing_rate` utility."""

    if not isinstance(spike_trains, Tensor):
        spike_trains = torch.as_tensor(spike_trains)

    kern_sd = max(int(round(kern_sd_ms / sampling_precision)), 1)
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window = window / np.sum(window)

    spike_np = spike_trains.detach().cpu().numpy()

    def filt(x: np.ndarray) -> np.ndarray:
        return convolve(x, window, mode="same", method="direct")

    smoothed = np.apply_along_axis(filt, 1, spike_np)
    return torch.as_tensor(
        smoothed,
        dtype=spike_trains.dtype,
        device=spike_trains.device,
    )


def preprocessing_from_dict(data: dict[str, Any] | None) -> PreprocessingConfig:
    return PreprocessingConfig.model_validate(data or {})
