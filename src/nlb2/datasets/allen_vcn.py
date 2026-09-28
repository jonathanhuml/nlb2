"""Prepared Allen Visual Coding Neuropixels low-trial datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_ALLEN_VCN_H5_PATH = Path("data") / "real" / "allen_vcn" / "allen_vcn_low_trial_20ms.h5"

ALLEN_VCN_GROUPS: tuple[str, ...] = (
    "flash_fc_top",
    "flash_bo_top",
    "flash_fc_second",
    "bo_drifting_one_tf",
    "fc_movie_one",
    "bo_movie_one",
    "bo_movie_three",
)

ALLEN_VCN_DATASETS: tuple[str, ...] = tuple(f"allen_vcn_{group}" for group in ALLEN_VCN_GROUPS)
ALLEN_VCN_GROUP_ALIASES: dict[str, str] = {
    "allen_vcn": "flash_fc_top",
    **{f"allen_vcn_{group}": group for group in ALLEN_VCN_GROUPS},
}


class AllenVCNDatasetConfig(BaseModel):
    """Config for prepared Allen Visual Coding Neuropixels H5 groups."""

    model_config = ConfigDict(extra="forbid")

    name: str = "allen_vcn"
    data_path: Path = DEFAULT_ALLEN_VCN_H5_PATH
    group: Optional[str] = None
    max_train_trials: Optional[int] = Field(default=None, ge=1)
    max_valid_trials: Optional[int] = Field(default=None, ge=1)
    input_key: str = "spikes"
    target_key: str = "heldout_spikes"
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _resolve_group(self) -> "AllenVCNDatasetConfig":
        if self.group is None:
            if self.name not in ALLEN_VCN_GROUP_ALIASES:
                known = ", ".join(sorted(ALLEN_VCN_GROUP_ALIASES))
                raise ValueError(
                    f"Allen VCN dataset name '{self.name}' does not imply a group. "
                    f"Use one of {known}, or set group explicitly."
                )
            self.group = ALLEN_VCN_GROUP_ALIASES[self.name]
        if self.group not in ALLEN_VCN_GROUPS:
            known_groups = ", ".join(ALLEN_VCN_GROUPS)
            raise ValueError(f"Unknown Allen VCN group '{self.group}'. Known groups: {known_groups}.")
        return self


@dataclass
class AllenVCNArrays:
    train_heldin_spikes: Tensor
    train_heldout_spikes: Tensor
    valid_heldin_spikes: Tensor
    valid_heldout_spikes: Tensor
    train_full_spikes: Tensor | None
    valid_full_spikes: Tensor | None
    train_condition_ids: Tensor | None
    valid_condition_ids: Tensor | None
    dt: float
    metadata: dict[str, object]


def load_allen_vcn_h5(config: AllenVCNDatasetConfig) -> AllenVCNArrays:
    """Load one prepared Allen VCN H5 group."""

    path = config.data_path.expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Allen VCN H5 not found: {path}. "
            "Set data_path to an existing prepared Allen VCN H5 file."
        )
    if config.group is None:
        raise ValueError("Allen VCN config group was not resolved.")

    with h5py.File(path, "r") as handle:
        if config.group not in handle:
            available = ", ".join(sorted(handle.keys()))
            raise KeyError(
                f"Allen VCN H5 {path} is missing group '{config.group}'. "
                f"Available groups: {available}"
            )
        group = handle[config.group]
        train_heldin = _read_array(group, "train_spikes_heldin", config.max_train_trials)
        train_heldout = _read_array(group, "train_spikes_heldout", config.max_train_trials)
        valid_heldin = _read_array(group, "eval_spikes_heldin", config.max_valid_trials)
        valid_heldout = _read_array(group, "eval_spikes_heldout", config.max_valid_trials)
        train_full = _read_optional_array(group, "train_spikes_full", config.max_train_trials)
        valid_full = _read_optional_array(group, "eval_spikes_full", config.max_valid_trials)
        train_condition_ids = _read_optional_int_array(group, "train_condition_ids", config.max_train_trials)
        valid_condition_ids = _read_optional_int_array(group, "eval_condition_ids", config.max_valid_trials)
        metadata = {key: _decode_attr(value) for key, value in group.attrs.items()}

    _validate_pair("train", train_heldin, train_heldout)
    _validate_pair("valid", valid_heldin, valid_heldout)
    dt = float(metadata.get("bin_size_s", metadata.get("dt", 0.02)))

    return AllenVCNArrays(
        train_heldin_spikes=train_heldin,
        train_heldout_spikes=train_heldout,
        valid_heldin_spikes=valid_heldin,
        valid_heldout_spikes=valid_heldout,
        train_full_spikes=train_full,
        valid_full_spikes=valid_full,
        train_condition_ids=train_condition_ids,
        valid_condition_ids=valid_condition_ids,
        dt=dt,
        metadata=metadata,
    )


class AllenVCNDataset(Dataset):
    """PyTorch Dataset for prepared Allen VCN held-in/held-out co-smoothing."""

    def __init__(
        self,
        config: AllenVCNDatasetConfig,
        split: str = "train",
        arrays: AllenVCNArrays | None = None,
    ) -> None:
        self.config = config
        self.split = split
        self.arrays = arrays or load_allen_vcn_h5(config)

        if split == "train":
            self.spikes = self.arrays.train_heldin_spikes
            self.raw_spikes = self.arrays.train_heldout_spikes
            self.full_spikes = self.arrays.train_full_spikes
            self.condition_ids = self.arrays.train_condition_ids
        elif split == "valid":
            self.spikes = self.arrays.valid_heldin_spikes
            self.raw_spikes = self.arrays.valid_heldout_spikes
            self.full_spikes = self.arrays.valid_full_spikes
            self.condition_ids = self.arrays.valid_condition_ids
        else:
            raise ValueError("split must be 'train' or 'valid'.")

    @classmethod
    def make_splits(
        cls,
        config: AllenVCNDatasetConfig,
    ) -> tuple["AllenVCNDataset", "AllenVCNDataset"]:
        arrays = load_allen_vcn_h5(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {
            "spikes": self.spikes[index],
            "heldin_spikes": self.spikes[index],
            "raw_spikes": self.raw_spikes[index],
            "heldout_spikes": self.raw_spikes[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }
        if self.full_spikes is not None:
            item["full_spikes"] = self.full_spikes[index]
        if self.condition_ids is not None:
            item["condition_id"] = self.condition_ids[index]
        return item


def _read_array(group: h5py.Group, key: str, max_trials: int | None) -> Tensor:
    if key not in group:
        available = ", ".join(sorted(group.keys()))
        raise KeyError(f"Allen VCN group is missing '{key}'. Available keys: {available}")
    array = group[key][:max_trials] if max_trials is not None else group[key][()]
    tensor = torch.from_numpy(np.asarray(array).copy()).float()
    if tensor.ndim != 3:
        raise ValueError(f"Allen VCN key '{key}' must have shape (trials, time, neurons).")
    return tensor.contiguous()


def _read_optional_array(group: h5py.Group, key: str, max_trials: int | None) -> Tensor | None:
    if key not in group:
        return None
    return _read_array(group, key, max_trials)


def _read_optional_int_array(group: h5py.Group, key: str, max_trials: int | None) -> Tensor | None:
    if key not in group:
        return None
    array = group[key][:max_trials] if max_trials is not None else group[key][()]
    return torch.from_numpy(np.asarray(array).copy()).long().contiguous()


def _validate_pair(split: str, heldin: Tensor, heldout: Tensor) -> None:
    if heldin.shape[:2] != heldout.shape[:2]:
        raise ValueError(
            f"Allen VCN {split} held-in/held-out trial-time mismatch: "
            f"{tuple(heldin.shape[:2])} != {tuple(heldout.shape[:2])}"
        )


def _decode_attr(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
