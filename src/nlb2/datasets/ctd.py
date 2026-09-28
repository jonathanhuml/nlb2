"""Computation-through-Dynamics Toolkit H5 datasets.

CTD generated datasets store observed spike counts, simulator firing activity,
and task-trained latent trajectories in split-prefixed H5 datasets such as
``train_recon_data``, ``train_activity``, and ``train_latents``.  This module
adapts those files to the NLB2 synthetic dataset contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Union

import h5py
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset


class CTDDatasetConfig(BaseModel):
    """Config for loading CTD generated H5 datasets."""

    model_config = ConfigDict(extra="forbid")

    name: str = "ctd"
    task: str = "nbff"
    data_path: Path
    observation_key_template: str = "{split}_recon_data"
    rates_key_template: str = "{split}_activity"
    rates_unit: Literal["counts", "hz"] = Field(
        default="counts",
        description=(
            "Units of stored activity. CTD simulator activity is a Poisson mean "
            "per bin; set 'hz' explicitly for imported physical firing rates."
        ),
    )
    latents_key_template: str = "{split}_latents"
    train_split: str = "train"
    valid_split: str = "valid"
    dt: float = 1.0
    max_train_trials: Optional[int] = None
    max_valid_trials: Optional[int] = None
    trim_to_observed_neurons: bool = True
    metadata: Dict[str, Union[int, float, str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self) -> "CTDDatasetConfig":
        alias_to_task = {
            "ctd_chaotic_delayed_matching": "chaotic_delayed_matching",
            "ctd_multitask": "multitask",
            "ctd_nbff": "nbff",
            "ctd_phase_coded_memory": "phase_coded_memory",
            "ctd_random_target": "random_target",
        }
        if self.name in alias_to_task:
            self.task = alias_to_task[self.name]
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        for value, field_name in [
            (self.max_train_trials, "max_train_trials"),
            (self.max_valid_trials, "max_valid_trials"),
        ]:
            if value is not None and value < 1:
                raise ValueError(f"{field_name} must be positive when set.")
        return self


@dataclass
class CTDArrays:
    train_spikes: Tensor
    valid_spikes: Tensor
    train_rates: Tensor
    valid_rates: Tensor
    train_latents: Tensor
    valid_latents: Tensor
    dt: float


def load_ctd_h5(config: CTDDatasetConfig) -> CTDArrays:
    """Load CTD split arrays from an H5 file."""

    path = config.data_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"CTD dataset file not found: {path}")

    with h5py.File(path, "r") as handle:
        train_spikes = _read_split_array(
            handle,
            config.observation_key_template,
            config.train_split,
            config.max_train_trials,
        )
        valid_spikes = _read_split_array(
            handle,
            config.observation_key_template,
            config.valid_split,
            config.max_valid_trials,
        )
        train_rates = _read_split_array(
            handle,
            config.rates_key_template,
            config.train_split,
            config.max_train_trials,
        )
        valid_rates = _read_split_array(
            handle,
            config.rates_key_template,
            config.valid_split,
            config.max_valid_trials,
        )
        train_latents = _read_split_array(
            handle,
            config.latents_key_template,
            config.train_split,
            config.max_train_trials,
        )
        valid_latents = _read_split_array(
            handle,
            config.latents_key_template,
            config.valid_split,
            config.max_valid_trials,
        )

    if config.trim_to_observed_neurons:
        train_rates = _match_observed_neurons(train_rates, train_spikes)
        valid_rates = _match_observed_neurons(valid_rates, valid_spikes)

    _validate_shapes("train", train_spikes, train_rates, train_latents)
    _validate_shapes("valid", valid_spikes, valid_rates, valid_latents)

    return CTDArrays(
        train_spikes=train_spikes,
        valid_spikes=valid_spikes,
        train_rates=train_rates,
        valid_rates=valid_rates,
        train_latents=train_latents,
        valid_latents=valid_latents,
        dt=float(config.dt),
    )


class CTDDataset(Dataset):
    """PyTorch Dataset wrapper around CTD generated H5 arrays."""

    def __init__(
        self,
        config: CTDDatasetConfig,
        split: Literal["train", "valid"] = "train",
        arrays: CTDArrays | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.arrays = arrays or load_ctd_h5(config)

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
        config: CTDDatasetConfig,
    ) -> tuple["CTDDataset", "CTDDataset"]:
        arrays = load_ctd_h5(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        return {
            "spikes": self.spikes[index],
            "rates": self.rates[index],
            "rates_unit": self.config.rates_unit,
            "latents": self.latents[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }


def _read_split_array(
    handle: h5py.File,
    template: str,
    split: str,
    max_trials: int | None,
) -> Tensor:
    key = template.format(split=split)
    if key not in handle:
        available = ", ".join(sorted(handle.keys()))
        raise KeyError(f"CTD H5 is missing '{key}'. Available keys: {available}")
    dataset = handle[key]
    array = dataset[:max_trials] if max_trials is not None else dataset[()]
    tensor = torch.from_numpy(array).float()
    if tensor.ndim != 3:
        raise ValueError(f"CTD key '{key}' must have shape (trials, time, dim).")
    if max_trials is not None:
        tensor = tensor[:max_trials]
    return tensor.contiguous()


def _match_observed_neurons(rates: Tensor, spikes: Tensor) -> Tensor:
    if rates.shape[-1] == spikes.shape[-1]:
        return rates
    if rates.shape[-1] > spikes.shape[-1]:
        return rates[..., : spikes.shape[-1]].contiguous()
    raise ValueError(
        "CTD rates have fewer neurons than observations: "
        f"{rates.shape[-1]} < {spikes.shape[-1]}"
    )


def _validate_shapes(split: str, spikes: Tensor, rates: Tensor, latents: Tensor) -> None:
    if spikes.shape != rates.shape:
        raise ValueError(
            f"CTD {split} spikes/rates shape mismatch: "
            f"{tuple(spikes.shape)} != {tuple(rates.shape)}"
        )
    if spikes.shape[:2] != latents.shape[:2]:
        raise ValueError(
            f"CTD {split} spikes/latents trial-time mismatch: "
            f"{tuple(spikes.shape[:2])} != {tuple(latents.shape[:2])}"
        )
