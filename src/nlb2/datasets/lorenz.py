"""Lorenz synthetic neural population dataset.

This is a cleaned-up PyTorch Dataset wrapper around the Lorenz synthetic data
pattern used in CASSM: low-dimensional Lorenz dynamics are embedded into a
higher-dimensional neural population by a random readout, then Poisson spikes
are sampled from the resulting rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset


class LorenzDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "lorenz"
    neurons: int = Field(default=32, ge=1)
    num_inits: int = Field(default=4, ge=1)
    num_trials: int = Field(default=8, ge=2)
    num_steps: int = Field(default=100, ge=1)
    burn_steps: int = Field(default=1_000, ge=0)
    train_fraction: float = Field(default=0.8, gt=0.0, lt=1.0)
    seed: int = 0
    latent_dt: float = Field(default=0.015, gt=0.0, allow_inf_nan=False)
    spike_bin_size: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)
    base_rate: float = Field(default=1.0, gt=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_split(self) -> "LorenzDatasetConfig":
        n_train = int(self.train_fraction * self.num_trials)
        if not 1 <= n_train < self.num_trials:
            raise ValueError("train_fraction must leave training and validation trials per condition.")
        return self


@dataclass
class LorenzArrays:
    train_spikes: Tensor
    valid_spikes: Tensor
    train_rates: Tensor
    valid_rates: Tensor
    train_latents: Tensor
    valid_latents: Tensor
    dt: float


def _lorenz_gradient(state: np.ndarray, weights: tuple[float, float, float]) -> np.ndarray:
    y1, y2, y3 = state.T
    sigma, rho, beta = weights
    return np.stack(
        [
            sigma * (y2 - y1),
            y1 * (rho - y3) - y2,
            y1 * y2 - beta * y3,
        ],
        axis=-1,
    )


def _rk4_step(
    state: np.ndarray,
    dt: float,
    weights: tuple[float, float, float],
) -> np.ndarray:
    k1 = dt * _lorenz_gradient(state, weights)
    k2 = dt * _lorenz_gradient(state + 0.5 * k1, weights)
    k3 = dt * _lorenz_gradient(state + 0.5 * k2, weights)
    k4 = dt * _lorenz_gradient(state + k3, weights)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def _simulate_lorenz(
    rng: np.random.Generator,
    num_inits: int,
    num_steps: int,
    burn_steps: int,
    dt: float,
    weights: tuple[float, float, float] = (10.0, 28.0, 8.0 / 3.0),
) -> np.ndarray:
    state = rng.normal(size=(num_inits, 3))
    for _ in range(burn_steps):
        state = _rk4_step(state, dt, weights)

    latents = np.empty((num_steps, num_inits, 3), dtype=np.float32)
    for t in range(num_steps):
        state = _rk4_step(state, dt, weights)
        latents[t] = state

    latents -= latents.mean(axis=(0, 1), keepdims=True)
    latents /= np.abs(latents).max() + 1e-8
    return latents


def generate_lorenz_data(config: LorenzDatasetConfig) -> LorenzArrays:
    """Generate Lorenz spikes, rates, and latent trajectories."""

    rng = np.random.default_rng(config.seed)
    latents = _simulate_lorenz(
        rng=rng,
        num_inits=config.num_inits,
        num_steps=config.num_steps,
        burn_steps=config.burn_steps,
        dt=config.latent_dt,
    )

    projection = (rng.random((3, config.neurons)) + 1.0) * np.sign(
        rng.normal(size=(3, config.neurons))
    )
    rates = np.exp(latents @ projection + np.log(config.base_rate)).astype(np.float32)

    # Convert from (time, init, neurons) to repeated trial samples.
    latents = np.broadcast_to(
        latents[None, ...],
        (config.num_trials, config.num_steps, config.num_inits, 3),
    ).transpose(0, 2, 1, 3)
    rates = np.broadcast_to(
        rates[None, ...],
        (config.num_trials, config.num_steps, config.num_inits, config.neurons),
    ).transpose(0, 2, 1, 3)
    spikes = rng.poisson(rates * config.spike_bin_size).astype(np.float32)

    n_train_trials = int(config.train_fraction * config.num_trials)

    def split(array: np.ndarray) -> tuple[Tensor, Tensor]:
        train = array[:n_train_trials].reshape(-1, config.num_steps, array.shape[-1])
        valid = array[n_train_trials:].reshape(-1, config.num_steps, array.shape[-1])
        return torch.from_numpy(train.copy()).float(), torch.from_numpy(valid.copy()).float()

    train_spikes, valid_spikes = split(spikes)
    train_rates, valid_rates = split(rates)
    train_latents, valid_latents = split(latents.astype(np.float32))

    return LorenzArrays(
        train_spikes=train_spikes,
        valid_spikes=valid_spikes,
        train_rates=train_rates,
        valid_rates=valid_rates,
        train_latents=train_latents,
        valid_latents=valid_latents,
        dt=config.spike_bin_size,
    )


class LorenzDataset(Dataset):
    """PyTorch Dataset returning observations plus optional truth for metrics."""

    def __init__(
        self,
        config: LorenzDatasetConfig | None = None,
        split: Literal["train", "valid"] = "train",
        arrays: LorenzArrays | None = None,
    ) -> None:
        self.config = config or LorenzDatasetConfig()
        self.split = split
        self.arrays = arrays or generate_lorenz_data(self.config)

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
        config: LorenzDatasetConfig | None = None,
    ) -> tuple["LorenzDataset", "LorenzDataset"]:
        config = config or LorenzDatasetConfig()
        arrays = generate_lorenz_data(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "spikes": self.spikes[index],
            "rates": self.rates[index],
            "rates_unit": "hz",
            "latents": self.latents[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }
