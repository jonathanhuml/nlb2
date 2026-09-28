"""Public data-loading API for NLB2 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset

from nlb2.datasets import (
    ALLEN_VCN_DATASETS,
    AllenVCNDataset,
    AllenVCNDatasetConfig,
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    CTDDataset,
    CTDDatasetConfig,
    LorenzDataset,
    LorenzDatasetConfig,
    NLBDataset,
    NLBDatasetConfig,
)
from nlb2.preprocessing import PreprocessedDataset, PreprocessingConfig


DATASET_CONFIGS: dict[str, type[BaseModel]] = {
    "allen_vcn": AllenVCNDatasetConfig,
    **{name: AllenVCNDatasetConfig for name in ALLEN_VCN_DATASETS},
    "area2_bump": NLBDatasetConfig,
    "chaotic_rnn": ChaoticRNNDatasetConfig,
    "ctd": CTDDatasetConfig,
    "ctd_chaotic_delayed_matching": CTDDatasetConfig,
    "ctd_multitask": CTDDatasetConfig,
    "ctd_nbff": CTDDatasetConfig,
    "ctd_phase_coded_memory": CTDDatasetConfig,
    "ctd_random_target": CTDDatasetConfig,
    "dmfc_rsg": NLBDatasetConfig,
    "lorenz": LorenzDatasetConfig,
    "mc_maze": NLBDatasetConfig,
    "mc_maze_large": NLBDatasetConfig,
    "mc_maze_medium": NLBDatasetConfig,
    "mc_maze_small": NLBDatasetConfig,
    "mc_rtt": NLBDatasetConfig,
}


def available_datasets() -> tuple[str, ...]:
    """Return registered dataset names."""

    return tuple(sorted(DATASET_CONFIGS))


def build_dataset_config(name: str, data: dict[str, Any] | None = None) -> BaseModel:
    """Build a dataset config from a registered name and optional overrides."""

    if name not in DATASET_CONFIGS:
        known = ", ".join(available_datasets()) or "<none>"
        raise KeyError(f"Unknown dataset '{name}'. Registered datasets: {known}.")

    payload = {"name": name}
    payload.update(data or {})
    return DATASET_CONFIGS[name].model_validate(payload)


def make_dataset_splits(config: BaseModel) -> tuple[Dataset, Dataset]:
    """Instantiate train/validation PyTorch datasets from a dataset config."""

    if isinstance(config, ChaoticRNNDatasetConfig):
        return ChaoticRNNDataset.make_splits(config)
    if isinstance(config, AllenVCNDatasetConfig):
        return AllenVCNDataset.make_splits(config)
    if isinstance(config, CTDDatasetConfig):
        return CTDDataset.make_splits(config)
    if isinstance(config, LorenzDatasetConfig):
        return LorenzDataset.make_splits(config)
    if isinstance(config, NLBDatasetConfig):
        return NLBDataset.make_splits(config)
    raise TypeError(f"Unsupported dataset config type {type(config).__name__}.")


@dataclass
class DataModule:
    """Small PyTorch-style data module used by the public experiment API."""

    config: BaseModel
    batch_size: int = 32
    preprocessing: PreprocessingConfig | None = None
    num_workers: int = 0
    pin_memory: bool = False

    def __post_init__(self) -> None:
        self.preprocessing = self.preprocessing or PreprocessingConfig()
        self.train_dataset: Dataset | None = None
        self.valid_dataset: Dataset | None = None

    def setup(self) -> None:
        """Create train and validation datasets."""

        train_dataset, valid_dataset = make_dataset_splits(self.config)
        if self.preprocessing and self.preprocessing.observations:
            train_dataset = PreprocessedDataset(train_dataset, self.preprocessing)
            valid_dataset = PreprocessedDataset(valid_dataset, self.preprocessing)

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def train_loader(self, shuffle: bool = True) -> DataLoader:
        """Return the training dataloader."""

        return DataLoader(
            self._require_dataset(self.train_dataset, "train"),
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def valid_loader(self) -> DataLoader:
        """Return the validation dataloader."""

        return DataLoader(
            self._require_dataset(self.valid_dataset, "valid"),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @property
    def n_neurons(self) -> int:
        dataset = self._require_dataset(self.train_dataset, "train")
        return int(getattr(dataset, "spikes").shape[-1])

    @property
    def n_time(self) -> int:
        dataset = self._require_dataset(self.train_dataset, "train")
        return int(getattr(dataset, "spikes").shape[1])

    @staticmethod
    def _require_dataset(dataset: Dataset | None, split: str) -> Dataset:
        if dataset is None:
            raise RuntimeError(f"DataModule.setup() must be called before using {split} data.")
        return dataset
