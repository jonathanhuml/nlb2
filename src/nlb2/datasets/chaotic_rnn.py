"""Chaotic random-RNN synthetic neural population dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset


class ChaoticRNNDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "chaotic_rnn"
    neurons: int = 50
    hidden_units: int = 50
    num_conditions: int = 8
    num_trials: int = 8
    num_steps: int = 100
    train_fraction: float = 0.8
    seed: int = 0
    g: float = Field(default=1.5, allow_inf_nan=False)
    tau: float = Field(default=0.025, gt=0.0, allow_inf_nan=False)
    dt: float = Field(default=0.010, gt=0.0, allow_inf_nan=False)
    max_firing_rate: float = Field(default=30.0, gt=0.0, allow_inf_nan=False)
    x0_std: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_dataset(self) -> "ChaoticRNNDatasetConfig":
        if self.neurons < 1:
            raise ValueError("neurons must be positive.")
        if self.hidden_units < 1:
            raise ValueError("hidden_units must be positive.")
        if self.neurons > self.hidden_units:
            raise ValueError("neurons must be <= hidden_units.")
        if self.num_conditions < 1:
            raise ValueError("num_conditions must be positive.")
        if self.num_trials < 2:
            raise ValueError("num_trials must be at least 2.")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive.")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        if int(self.train_fraction * self.num_trials) < 1:
            raise ValueError("train_fraction leaves no training trials per condition.")
        if int(self.train_fraction * self.num_trials) >= self.num_trials:
            raise ValueError("train_fraction leaves no validation trials per condition.")
        return self


@dataclass
class ChaoticRNNArrays:
    train_spikes: Tensor
    valid_spikes: Tensor
    train_rates: Tensor
    valid_rates: Tensor
    train_latents: Tensor
    valid_latents: Tensor
    dt: float


def generate_chaotic_rnn_data(config: ChaoticRNNDatasetConfig) -> ChaoticRNNArrays:
    """Generate random-RNN spikes, per-bin rates, and hidden trajectories."""

    rng = np.random.default_rng(config.seed)
    total_trials = config.num_conditions * config.num_trials
    weights = rng.normal(size=(config.hidden_units, config.hidden_units)).astype(np.float32)
    weights /= np.sqrt(float(config.hidden_units))
    recurrent = (config.dt / config.tau) * config.g * weights
    alpha = 1.0 - config.dt / config.tau

    condition_x0 = config.x0_std * rng.normal(
        size=(config.num_conditions, config.hidden_units)
    ).astype(np.float32)
    x = np.repeat(condition_x0, config.num_trials, axis=0).T
    r = np.tanh(x)

    hidden = np.empty(
        (total_trials, config.num_steps, config.hidden_units),
        dtype=np.float32,
    )
    for step in range(config.num_steps):
        x = alpha * x + recurrent @ r
        r = np.tanh(x)
        hidden[:, step, :] = r.T

    sampled = hidden[:, :, : config.neurons].copy()
    normalized = _normalize_trial_channels(sampled)
    mean_counts = normalized * config.max_firing_rate * config.dt
    spikes = rng.poisson(mean_counts).astype(np.float32)

    train_indices, valid_indices = _condition_balanced_split(
        config.num_conditions,
        config.num_trials,
        config.train_fraction,
    )

    def split(array: np.ndarray) -> tuple[Tensor, Tensor]:
        train = torch.from_numpy(array[train_indices].copy()).float()
        valid = torch.from_numpy(array[valid_indices].copy()).float()
        return train, valid

    train_spikes, valid_spikes = split(spikes)
    train_rates, valid_rates = split(mean_counts.astype(np.float32))
    train_latents, valid_latents = split(hidden)
    return ChaoticRNNArrays(
        train_spikes=train_spikes,
        valid_spikes=valid_spikes,
        train_rates=train_rates,
        valid_rates=valid_rates,
        train_latents=train_latents,
        valid_latents=valid_latents,
        dt=config.dt,
    )


class ChaoticRNNDataset(Dataset):
    """PyTorch Dataset for chaotic random-RNN spike-count trajectories."""

    def __init__(
        self,
        config: ChaoticRNNDatasetConfig | None = None,
        split: Literal["train", "valid"] = "train",
        arrays: ChaoticRNNArrays | None = None,
    ) -> None:
        self.config = config or ChaoticRNNDatasetConfig()
        self.split = split
        self.arrays = arrays or generate_chaotic_rnn_data(self.config)

        if split == "train":
            self.spikes = self.arrays.train_spikes
            self.rates = self.arrays.train_rates
            self.latents = self.arrays.train_latents
        elif split == "valid":
            self.spikes = self.arrays.valid_spikes
            self.rates = self.arrays.valid_rates
            self.latents = self.arrays.valid_latents
        else:
            raise ValueError("split must be 'train' or 'valid'.")

    @classmethod
    def make_splits(
        cls,
        config: ChaoticRNNDatasetConfig | None = None,
    ) -> tuple["ChaoticRNNDataset", "ChaoticRNNDataset"]:
        config = config or ChaoticRNNDatasetConfig()
        arrays = generate_chaotic_rnn_data(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "spikes": self.spikes[index],
            "rates": self.rates[index],
            "rates_unit": "counts",
            "latents": self.latents[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }


def _normalize_trial_channels(array: np.ndarray) -> np.ndarray:
    mins = array.min(axis=1, keepdims=True)
    maxs = array.max(axis=1, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-8)
    return ((array - mins) / denom).astype(np.float32)


def _condition_balanced_split(
    num_conditions: int,
    num_trials: int,
    train_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_train = int(train_fraction * num_trials)
    train_indices: list[int] = []
    valid_indices: list[int] = []
    for condition in range(num_conditions):
        offset = condition * num_trials
        train_indices.extend(range(offset, offset + n_train))
        valid_indices.extend(range(offset + n_train, offset + num_trials))
    return np.array(train_indices, dtype=np.int64), np.array(valid_indices, dtype=np.int64)
