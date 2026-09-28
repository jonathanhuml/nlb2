"""Neural Latents Benchmark dataset wrappers and preparation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Literal, Optional
from urllib.error import URLError
from urllib.request import urlretrieve

import h5py
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor
from torch.utils.data import Dataset

from nlb2.datasets._nlb_resample import resample_nwb_dataset


NLBCoreDataset = Literal[
    "mc_maze",
    "mc_maze_large",
    "mc_maze_medium",
    "mc_maze_small",
    "mc_rtt",
    "area2_bump",
    "dmfc_rsg",
]
NLBSplit = Literal["val", "test"]
NLBBinSize = Literal[5, 20]
NLBInputMode = Literal[
    "heldin",
    "heldin_full_reconstruction",
    "full_observed",
    "autolfads",
    "lfads_torch",
]

NLB_DATASETS: tuple[str, ...] = (
    "area2_bump",
    "mc_maze",
    "mc_maze_large",
    "mc_maze_medium",
    "mc_maze_small",
    "mc_rtt",
    "dmfc_rsg",
)
NLB_BIN_SIZES_MS: tuple[int, ...] = (5, 20)
NLB_TARGET_H5_URL = (
    "https://media.githubusercontent.com/media/neurallatents/nlb_tools/main/data/eval_data_test.h5"
)

DATASET_TO_DANDISET: dict[str, str] = {
    "area2_bump": "000127",
    "mc_maze": "000128",
    "mc_maze_large": "000138",
    "mc_maze_medium": "000139",
    "mc_maze_small": "000140",
    "mc_rtt": "000129",
    "dmfc_rsg": "000130",
}

DATASET_TO_TEST_NWB: dict[str, Path] = {
    "area2_bump": Path("sub-Han/sub-Han_desc-test_ecephys.nwb"),
    "mc_maze": Path("sub-Jenkins/sub-Jenkins_ses-full_desc-test_ecephys.nwb"),
    "mc_maze_large": Path("sub-Jenkins/sub-Jenkins_ses-large_desc-test_ecephys.nwb"),
    "mc_maze_medium": Path("sub-Jenkins/sub-Jenkins_ses-medium_desc-test_ecephys.nwb"),
    "mc_maze_small": Path("sub-Jenkins/sub-Jenkins_ses-small_desc-test_ecephys.nwb"),
    "mc_rtt": Path("sub-Indy/sub-Indy_desc-test_ecephys.nwb"),
    "dmfc_rsg": Path("sub-Haydn/sub-Haydn_desc-test_ecephys.nwb"),
}

DATASET_TO_TRAIN_NWB: dict[str, Path] = {
    "area2_bump": Path("sub-Han/sub-Han_desc-train_behavior+ecephys.nwb"),
    "mc_maze": Path("sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"),
    "mc_maze_large": Path(
        "sub-Jenkins/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb"
    ),
    "mc_maze_medium": Path(
        "sub-Jenkins/sub-Jenkins_ses-medium_desc-train_behavior+ecephys.nwb"
    ),
    "mc_maze_small": Path(
        "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
    ),
    "mc_rtt": Path("sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"),
    "dmfc_rsg": Path("sub-Haydn/sub-Haydn_desc-train_ecephys.nwb"),
}


class NLBDatasetConfig(BaseModel):
    """Config for an NLB co-smoothing evaluation H5."""

    model_config = ConfigDict(extra="forbid")

    name: NLBCoreDataset = "mc_maze"
    data_path: Optional[str] = None
    split: NLBSplit = "test"
    bin_size_ms: NLBBinSize = 5
    bin_size: Optional[float] = Field(
        default=None,
        description="Legacy seconds-valued bin size. Prefer bin_size_ms.",
    )
    group: Optional[str] = None
    max_trials: Optional[int] = None
    input_key: str = "eval_spikes_heldin"
    target_key: str = "eval_spikes_heldout"
    input_mode: NLBInputMode = "heldin"
    include_forward: bool = False
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _sync_legacy_bin_size(self) -> "NLBDatasetConfig":
        if self.bin_size is None:
            self.bin_size = float(self.bin_size_ms) / 1000.0
        else:
            bin_ms = int(round(float(self.bin_size) * 1000.0))
            if bin_ms not in NLB_BIN_SIZES_MS:
                raise ValueError("bin_size must correspond to 5 ms or 20 ms.")
            self.bin_size_ms = bin_ms  # type: ignore[assignment]
        if self.input_mode == "lfads_torch":
            self.input_mode = "heldin_full_reconstruction"  # type: ignore[assignment]
        elif self.input_mode == "autolfads":
            self.input_mode = "full_observed"  # type: ignore[assignment]
        return self

    @property
    def resolved_data_path(self) -> Path:
        if self.data_path is not None:
            return Path(self.data_path)
        return default_nlb_h5_path(self.name, self.split, self.bin_size_ms)

    @property
    def resolved_group(self) -> str:
        if self.group is not None:
            return self.group
        return nlb_group_name(self.name, self.bin_size_ms)


@dataclass
class NLBArrays:
    """Loaded held-in input and held-out target arrays."""

    train_heldin_spikes: Tensor | None
    train_heldout_spikes: Tensor | None
    eval_heldin_spikes: Tensor
    eval_heldout_spikes: Tensor
    train_heldin_forward_spikes: Tensor | None
    train_heldout_forward_spikes: Tensor | None
    eval_heldin_forward_spikes: Tensor | None
    eval_heldout_forward_spikes: Tensor | None
    dt: float


def nlb_group_name(dataset: str, bin_size_ms: int) -> str:
    """Return the standard nlb_tools H5 group for a dataset/bin-size pair."""

    if bin_size_ms == 5:
        return dataset
    return f"{dataset}_{bin_size_ms}"


def default_nlb_h5_path(dataset: str, split: str, bin_size_ms: int) -> Path:
    """Return the repository-local NLB2 NLB H5 path."""

    return Path("data") / "real" / "nlb" / f"{dataset}_{split}_{bin_size_ms}ms.h5"


def load_nlb_h5(config: NLBDatasetConfig, *, require_training: bool = True) -> NLBArrays:
    """Load held-in and held-out spikes from a NLB2/NLB H5 file."""

    path = config.resolved_data_path
    if not path.exists():
        raise FileNotFoundError(
            f"NLB H5 not found: {path}. Run `nlb2 prepare-nlb --datasets "
            f"{config.name} --splits {config.split} --bin-sizes-ms {config.bin_size_ms}` first."
        )

    with h5py.File(path, "r") as handle:
        group = _select_h5_group(handle, config.resolved_group)
        training_keys = ("train_spikes_heldin", "train_spikes_heldout")
        missing = [key for key in training_keys if key not in group]
        if require_training and missing:
            raise ValueError(
                f"{path}: missing training tensors {', '.join(missing)}. "
                "Training requires separate training data; evaluation tensors cannot "
                "be substituted. Prepare an NLB train/evaluation file first."
            )
        eval_heldin = np.asarray(group[config.input_key])
        eval_heldout = np.asarray(group[config.target_key])
        train_heldin = _optional_array(group, "train_spikes_heldin")
        train_heldout = _optional_array(group, "train_spikes_heldout")
        train_heldin_forward = _optional_array(group, "train_spikes_heldin_forward")
        train_heldout_forward = _optional_array(group, "train_spikes_heldout_forward")
        eval_heldin_forward = _optional_array(group, "eval_spikes_heldin_forward")
        eval_heldout_forward = _optional_array(group, "eval_spikes_heldout_forward")

    if eval_heldin.shape[:2] != eval_heldout.shape[:2]:
        raise ValueError(
            f"{path}: eval held-in shape {eval_heldin.shape} is incompatible with "
            f"held-out {eval_heldout.shape}."
        )
    if (
        train_heldin is not None
        and train_heldout is not None
        and train_heldin.shape[:2] != train_heldout.shape[:2]
    ):
        raise ValueError(
            f"{path}: train held-in shape {train_heldin.shape} is incompatible with "
            f"held-out {train_heldout.shape}."
        )

    if config.max_trials is not None:
        eval_heldin = eval_heldin[: config.max_trials]
        eval_heldout = eval_heldout[: config.max_trials]
        train_heldin = _slice_optional(train_heldin, config.max_trials)
        train_heldout = _slice_optional(train_heldout, config.max_trials)
        train_heldin_forward = _slice_optional(train_heldin_forward, config.max_trials)
        train_heldout_forward = _slice_optional(train_heldout_forward, config.max_trials)
        eval_heldin_forward = _slice_optional(eval_heldin_forward, config.max_trials)
        eval_heldout_forward = _slice_optional(eval_heldout_forward, config.max_trials)

    return NLBArrays(
        train_heldin_spikes=_optional_tensor(train_heldin),
        train_heldout_spikes=_optional_tensor(train_heldout),
        eval_heldin_spikes=torch.from_numpy(eval_heldin.copy()).float(),
        eval_heldout_spikes=torch.from_numpy(eval_heldout.copy()).float(),
        train_heldin_forward_spikes=_optional_tensor(train_heldin_forward),
        train_heldout_forward_spikes=_optional_tensor(train_heldout_forward),
        eval_heldin_forward_spikes=_optional_tensor(eval_heldin_forward),
        eval_heldout_forward_spikes=_optional_tensor(eval_heldout_forward),
        dt=float(config.bin_size_ms) / 1000.0,
    )


def _optional_array(group: h5py.Group | h5py.File, key: str) -> np.ndarray | None:
    if key not in group:
        return None
    return np.asarray(group[key])


def _slice_optional(array: np.ndarray | None, max_trials: int) -> np.ndarray | None:
    if array is None:
        return None
    return array[:max_trials]


def _optional_tensor(array: np.ndarray | None) -> Tensor | None:
    if array is None:
        return None
    return torch.from_numpy(array.copy()).float()


def _concat_channels(left: Tensor, right: Tensor | None) -> Tensor:
    if right is None:
        return left
    if left.shape[:-1] != right.shape[:-1]:
        raise ValueError(
            f"Cannot concatenate NLB tensors with shapes {tuple(left.shape)} "
            f"and {tuple(right.shape)}."
        )
    return torch.cat([left, right.to(device=left.device, dtype=left.dtype)], dim=-1)


def _forward_steps(heldin_forward: Tensor | None, heldout_forward: Tensor | None) -> int:
    if heldin_forward is not None:
        return int(heldin_forward.shape[1])
    if heldout_forward is not None:
        return int(heldout_forward.shape[1])
    return 0


def _full_observed_sequence(
    heldin: Tensor,
    heldout: Tensor,
    heldin_forward: Tensor | None,
    heldout_forward: Tensor | None,
    *,
    include_forward: bool,
) -> Tensor:
    full = _concat_channels(heldin, heldout)
    if not include_forward or heldin_forward is None:
        return full
    forward = _concat_channels(heldin_forward, heldout_forward)
    return torch.cat([full, forward.to(device=full.device, dtype=full.dtype)], dim=1)


class NLBDataset(Dataset):
    """PyTorch Dataset for NLB held-in to held-out co-smoothing."""

    def __init__(
        self,
        config: Optional[NLBDatasetConfig] = None,
        split: Literal["train", "valid"] = "train",
        arrays: Optional[NLBArrays] = None,
    ) -> None:
        self.config = config or NLBDatasetConfig()
        self.split = split
        self.arrays = arrays or load_nlb_h5(self.config, require_training=split == "train")
        if split == "train":
            if self.arrays.train_heldin_spikes is None or self.arrays.train_heldout_spikes is None:
                raise ValueError("Training requires train_spikes_heldin and train_spikes_heldout.")
            self.heldin_spikes = self.arrays.train_heldin_spikes
            self.raw_spikes = self.arrays.train_heldout_spikes
            self.heldin_forward_spikes = self.arrays.train_heldin_forward_spikes
            self.heldout_forward_spikes = self.arrays.train_heldout_forward_spikes
        elif split == "valid":
            self.heldin_spikes = self.arrays.eval_heldin_spikes
            self.raw_spikes = self.arrays.eval_heldout_spikes
            self.heldin_forward_spikes = self.arrays.eval_heldin_forward_spikes
            self.heldout_forward_spikes = self.arrays.eval_heldout_forward_spikes
        else:
            raise ValueError("split must be 'train' or 'valid'.")
        self.reconstruction_spikes = _full_observed_sequence(
            self.heldin_spikes,
            self.raw_spikes,
            self.heldin_forward_spikes,
            self.heldout_forward_spikes,
            include_forward=bool(self.config.include_forward),
        )
        self.spikes = self._build_input_spikes()

    @classmethod
    def make_splits(
        cls,
        config: Optional[NLBDatasetConfig] = None,
    ) -> tuple["NLBDataset", "NLBDataset"]:
        config = config or NLBDatasetConfig()
        arrays = load_nlb_h5(config)
        return cls(config, "train", arrays), cls(config, "valid", arrays)

    def __len__(self) -> int:
        return int(self.spikes.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = {
            "spikes": self.spikes[index],
            "heldin_spikes": self.heldin_spikes[index],
            "raw_spikes": self.raw_spikes[index],
            "heldout_spikes": self.raw_spikes[index],
            "dt": torch.tensor(self.arrays.dt, dtype=torch.float32),
        }
        if self.config.input_mode in {
            "full_observed",
            "heldin_full_reconstruction",
        }:
            item["reconstruction_spikes"] = self.reconstruction_spikes[index]
        if self.heldin_forward_spikes is not None:
            item["heldin_forward_spikes"] = self.heldin_forward_spikes[index]
        if self.heldout_forward_spikes is not None:
            item["heldout_forward_spikes"] = self.heldout_forward_spikes[index]
        return item

    def _build_input_spikes(self) -> Tensor:
        if self.config.input_mode in {"heldin", "heldin_full_reconstruction"}:
            return self.heldin_spikes
        if self.config.input_mode != "full_observed":
            raise ValueError(f"Unsupported NLB input_mode '{self.config.input_mode}'.")

        full = _concat_channels(self.heldin_spikes, self.raw_spikes)
        if self.split == "valid":
            full = _concat_channels(self.heldin_spikes, torch.zeros_like(self.raw_spikes))

        if not self.config.include_forward:
            return full

        forward_steps = _forward_steps(self.heldin_forward_spikes, self.heldout_forward_spikes)
        if forward_steps == 0:
            return full
        if self.split == "train" and self.heldin_forward_spikes is not None:
            forward = _concat_channels(
                self.heldin_forward_spikes,
                self.heldout_forward_spikes,
            )
        else:
            forward = torch.zeros(
                full.shape[0],
                forward_steps,
                full.shape[-1],
                dtype=full.dtype,
                device=full.device,
            )
        return torch.cat([full, forward.to(dtype=full.dtype, device=full.device)], dim=1)


@dataclass(frozen=True)
class PreparedNLBFile:
    """Summary of one prepared NLB file."""

    dataset: str
    split: str
    bin_size_ms: int
    path: Path
    heldin_shape: tuple[int, ...]
    heldout_shape: tuple[int, ...]


def prepare_nlb_data(
    datasets: Iterable[str] = NLB_DATASETS,
    splits: Iterable[str] = ("test",),
    bin_sizes_ms: Iterable[int] = NLB_BIN_SIZES_MS,
    output_dir: Path | str = Path("data/real/nlb"),
    target_h5: Path | str | None = None,
    nwb_root: Path | str = Path("data/real/nlb/dandi"),
    search_roots: Iterable[Path | str] | None = None,
    download: bool = False,
    overwrite: bool = False,
    include_psth: bool = False,
    train_trial_split: str = "train",
    eval_trial_split: str | None = None,
) -> list[PreparedNLBFile]:
    """Prepare NLB2-ready NLB H5 files from DANDI NWB and NLB target tensors."""

    output_dir = Path(output_dir)
    nwb_root = Path(nwb_root)
    split_list = list(splits)
    target_path = (
        _resolve_target_h5(
            target_h5,
            download=download,
            output_dir=output_dir,
        )
        if "test" in split_list
        else None
    )
    roots = _default_nwb_search_roots(nwb_root)
    if search_roots is not None:
        roots.extend(Path(p) for p in search_roots)
    roots = list(dict.fromkeys(roots))
    prepared: list[PreparedNLBFile] = []

    for dataset in datasets:
        _validate_dataset(dataset)
        for split in split_list:
            if split not in {"val", "test"}:
                raise ValueError(f"Unsupported NLB split '{split}'. Expected 'val' or 'test'.")
            for bin_size_ms in bin_sizes_ms:
                if int(bin_size_ms) not in NLB_BIN_SIZES_MS:
                    raise ValueError("Only 5 ms and 20 ms NLB tensors are supported.")
                output = output_dir / f"{dataset}_{split}_{int(bin_size_ms)}ms.h5"
                if output.exists() and not overwrite:
                    prepared.append(_validate_prepared_h5(output, dataset, split, int(bin_size_ms)))
                    continue
                output.parent.mkdir(parents=True, exist_ok=True)
                if split == "test":
                    if target_path is None:
                        raise RuntimeError("Internal error: test split requires an NLB target H5.")
                    result = _prepare_test_h5(
                        dataset=dataset,
                        bin_size_ms=int(bin_size_ms),
                        output=output,
                        target_h5=target_path,
                        nwb_root=nwb_root,
                        search_roots=roots,
                        download=download,
                    )
                else:
                    result = _prepare_validation_h5(
                        dataset=dataset,
                        bin_size_ms=int(bin_size_ms),
                        output=output,
                        nwb_root=nwb_root,
                        search_roots=roots,
                        download=download,
                        include_psth=include_psth,
                        train_trial_split=train_trial_split,
                        eval_trial_split=eval_trial_split or "val",
                    )
                prepared.append(result)
    return prepared


def _select_h5_group(handle: h5py.File, group_name: str) -> h5py.Group | h5py.File:
    if group_name in handle:
        return handle[group_name]
    if "eval_spikes_heldin" in handle and "eval_spikes_heldout" in handle:
        return handle
    keys = ", ".join(handle.keys())
    raise KeyError(f"Could not find group '{group_name}' or root eval tensors. H5 keys: {keys}")


def _validate_dataset(dataset: str) -> None:
    if dataset not in NLB_DATASETS:
        known = ", ".join(NLB_DATASETS)
        raise ValueError(f"Unknown NLB dataset '{dataset}'. Expected one of: {known}.")


def _resolve_target_h5(
    path: Path | str | None,
    *,
    download: bool = False,
    output_dir: Path | str = Path("data/real/nlb"),
) -> Path:
    if path is not None:
        target = Path(path)
        if target.exists():
            return target
        if download:
            return _download_target_h5(target)
        raise FileNotFoundError(f"NLB target H5 not found: {target}")

    output_target = Path(output_dir) / "eval_data_test.h5"
    if download:
        if output_target.exists():
            return output_target
        return _download_target_h5(output_target)

    candidates = [
        output_target,
        Path("data/real/eval_data_test.h5"),
        Path("data/eval_data_test.h5"),
        Path("../nlb_tools/data/eval_data_test.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if download:
        return _download_target_h5(Path(output_dir) / "eval_data_test.h5")
    raise FileNotFoundError(
        "Could not find NLB public test target H5. Pass --target-h5 pointing to "
        "nlb_tools/data/eval_data_test.h5, or rerun with --download."
    )


def _download_target_h5(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urlretrieve(NLB_TARGET_H5_URL, path)
    except (OSError, URLError) as exc:
        raise RuntimeError(
            "Could not download NLB public test target H5 from "
            f"{NLB_TARGET_H5_URL}. Pass --target-h5 if you already have it locally."
        ) from exc
    if not h5py.is_hdf5(path):
        path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded NLB public test target is not a valid HDF5 file. "
            "Pass --target-h5 if you already have eval_data_test.h5 locally."
        )
    return path


def _default_nwb_search_roots(nwb_root: Path) -> list[Path]:
    return [
        nwb_root,
        Path("data/real/dandi"),
    ]


def _prepare_test_h5(
    dataset: str,
    bin_size_ms: int,
    output: Path,
    target_h5: Path,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> PreparedNLBFile:
    heldin = _build_eval_heldin(
        dataset=dataset,
        split="test",
        bin_size_ms=bin_size_ms,
        nwb_root=nwb_root,
        search_roots=search_roots,
        download=download,
    )
    train_dict = _build_train_tensors(
        dataset=dataset,
        trial_split=["train", "val"],
        bin_size_ms=bin_size_ms,
        nwb_root=nwb_root,
        search_roots=search_roots,
        download=download,
    )
    group_name = nlb_group_name(dataset, bin_size_ms)
    target_dict = _read_group_dict(target_h5, group_name)
    target_dict.update(train_dict)
    target_dict["eval_spikes_heldin"] = heldin.astype(np.float32, copy=False)
    _write_flat_h5(output, target_dict)
    return _validate_prepared_h5(output, dataset, "test", bin_size_ms)


def _prepare_validation_h5(
    dataset: str,
    bin_size_ms: int,
    output: Path,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
    include_psth: bool,
    train_trial_split: str,
    eval_trial_split: str,
) -> PreparedNLBFile:
    nwb_path = _resolve_nwb_path(dataset, "train", nwb_root, search_roots, download)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        input_h5 = tmp_dir / "eval_input.h5"
        train_h5 = tmp_dir / "train_input.h5"
        target_h5 = tmp_dir / "target.h5"
        try:
            from nlb_tools.make_tensors import (
                make_train_input_tensors,
                make_eval_input_tensors,
                make_eval_target_tensors,
            )
            from nlb_tools.nwb_interface import NWBDataset
        except ImportError as exc:
            raise RuntimeError("nlb_tools is required to build NLB tensors from NWB.") from exc

        dataset_obj = NWBDataset(nwb_path)
        resample_nwb_dataset(dataset_obj, bin_size_ms)
        make_train_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=train_trial_split,
            include_behavior=True,
            include_forward_pred=True,
            save_file=True,
            save_path=str(train_h5),
        )
        make_eval_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=eval_trial_split,
            save_file=True,
            save_path=str(input_h5),
        )
        make_eval_target_tensors(
            dataset_obj,
            dataset_name=dataset,
            train_trial_split=train_trial_split,
            eval_trial_split=eval_trial_split,
            include_psth=include_psth,
            save_file=True,
            save_path=str(target_h5),
        )

        data = _read_group_or_root_dict(train_h5, nlb_group_name(dataset, bin_size_ms))
        data.update(_read_group_or_root_dict(input_h5, nlb_group_name(dataset, bin_size_ms)))
        data.update(_read_group_dict(target_h5, nlb_group_name(dataset, bin_size_ms)))
        _write_flat_h5(output, data)
    return _validate_prepared_h5(output, dataset, "val", bin_size_ms)


def _build_train_tensors(
    dataset: str,
    trial_split: str | list[str],
    bin_size_ms: int,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> dict[str, np.ndarray]:
    nwb_path = _resolve_nwb_path(dataset, "train", nwb_root, search_roots, download)
    with tempfile.TemporaryDirectory() as tmp:
        train_h5 = Path(tmp) / "train_input.h5"
        try:
            from nlb_tools.make_tensors import make_train_input_tensors
            from nlb_tools.nwb_interface import NWBDataset
        except ImportError as exc:
            raise RuntimeError("nlb_tools is required to build NLB tensors from NWB.") from exc

        dataset_obj = NWBDataset(nwb_path)
        resample_nwb_dataset(dataset_obj, bin_size_ms)
        make_train_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=trial_split,
            include_behavior=True,
            include_forward_pred=True,
            save_file=True,
            save_path=str(train_h5),
        )
        return _read_group_or_root_dict(train_h5, nlb_group_name(dataset, bin_size_ms))


def _build_eval_heldin(
    dataset: str,
    split: str,
    bin_size_ms: int,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> np.ndarray:
    nwb_path = _resolve_nwb_path(dataset, split, nwb_root, search_roots, download)
    with tempfile.TemporaryDirectory() as tmp:
        input_h5 = Path(tmp) / "eval_input.h5"
        try:
            from nlb_tools.make_tensors import make_eval_input_tensors
            from nlb_tools.nwb_interface import NWBDataset
        except ImportError as exc:
            raise RuntimeError("nlb_tools is required to build NLB tensors from NWB.") from exc

        dataset_obj = NWBDataset(nwb_path)
        resample_nwb_dataset(dataset_obj, bin_size_ms)
        make_eval_input_tensors(
            dataset_obj,
            dataset_name=dataset,
            trial_split=split,
            save_file=True,
            save_path=str(input_h5),
        )
        data = _read_group_or_root_dict(input_h5, nlb_group_name(dataset, bin_size_ms))
    return np.asarray(data["eval_spikes_heldin"])


def _resolve_nwb_path(
    dataset: str,
    split: str,
    nwb_root: Path,
    search_roots: list[Path],
    download: bool,
) -> Path:
    rel = DATASET_TO_TEST_NWB[dataset] if split == "test" else DATASET_TO_TRAIN_NWB[dataset]
    dandiset = DATASET_TO_DANDISET[dataset]
    candidates = []
    for root in search_roots:
        candidates.append(root / dandiset / rel)
        candidates.append(root / rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if download:
        _download_dandiset(dataset, nwb_root)
        downloaded = nwb_root / dandiset / rel
        if downloaded.exists():
            return downloaded
        alternate = nwb_root / rel
        if alternate.exists():
            return alternate

    formatted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find {dataset} {split} NWB. Checked:\n  {formatted}\n"
        "Rerun with --download or pass --nwb-root/--search-root."
    )


def _download_dandiset(dataset: str, nwb_root: Path) -> None:
    dandiset = DATASET_TO_DANDISET[dataset]
    nwb_root.mkdir(parents=True, exist_ok=True)
    if shutil.which("dandi") is None:
        raise RuntimeError(
            "dandi CLI not found. Install NLB2 with its data dependencies or pass local NWB files."
        )
    subprocess.run(
        [
            "dandi",
            "download",
            f"https://dandiarchive.org/dandiset/{dandiset}",
            "-o",
            str(nwb_root),
        ],
        check=True,
    )


def _read_group_dict(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if group_name not in handle:
            keys = ", ".join(handle.keys())
            raise KeyError(f"{path} does not contain group '{group_name}'. H5 keys: {keys}")
        return {
            key: np.asarray(value)
            for key, value in handle[group_name].items()
            if isinstance(value, h5py.Dataset)
        }


def _read_group_or_root_dict(path: Path, group_name: str) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        group = handle[group_name] if group_name in handle else handle
        return {
            key: np.asarray(value)
            for key, value in group.items()
            if isinstance(value, h5py.Dataset)
        }


def _write_flat_h5(path: Path, data: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        for key, value in data.items():
            handle.create_dataset(key, data=value, compression="gzip")


def _validate_prepared_h5(
    path: Path,
    dataset: str,
    split: str,
    bin_size_ms: int,
) -> PreparedNLBFile:
    with h5py.File(path, "r") as handle:
        group = _select_h5_group(handle, nlb_group_name(dataset, bin_size_ms))
        heldin_shape = tuple(group["eval_spikes_heldin"].shape)
        heldout_shape = tuple(group["eval_spikes_heldout"].shape)
    if heldin_shape[:2] != heldout_shape[:2]:
        raise ValueError(f"{path}: held-in {heldin_shape} and held-out {heldout_shape} mismatch.")
    return PreparedNLBFile(
        dataset=dataset,
        split=split,
        bin_size_ms=bin_size_ms,
        path=path,
        heldin_shape=heldin_shape,
        heldout_shape=heldout_shape,
    )
