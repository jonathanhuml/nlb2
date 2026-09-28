import h5py
import numpy as np
import pytest
from torch.utils.data import DataLoader

from nlb2.data import available_datasets, build_dataset_config, make_dataset_splits
from nlb2.datasets import AllenVCNDatasetConfig


def test_allen_vcn_registry_and_loader(tmp_path):
    path = tmp_path / "allen_vcn.h5"
    spikes = np.arange(6 * 4 * 5, dtype=np.uint16).reshape(6, 4, 5)

    with h5py.File(path, "w") as handle:
        group = handle.create_group("flash_fc_top")
        group.attrs["bin_size_s"] = 0.02
        group.create_dataset("train_spikes_heldin", data=spikes[:4, :, :3])
        group.create_dataset("train_spikes_heldout", data=spikes[:4, :, 3:])
        group.create_dataset("eval_spikes_heldin", data=spikes[4:, :, :3])
        group.create_dataset("eval_spikes_heldout", data=spikes[4:, :, 3:])
        group.create_dataset("train_spikes_full", data=spikes[:4])
        group.create_dataset("eval_spikes_full", data=spikes[4:])
        group.create_dataset("train_condition_ids", data=np.array([0, 1, 0, 1]))
        group.create_dataset("eval_condition_ids", data=np.array([0, 1]))

    assert "allen_vcn_flash_fc_top" in available_datasets()
    config = build_dataset_config(
        "allen_vcn_flash_fc_top",
        {"data_path": path},
    )
    assert isinstance(config, AllenVCNDatasetConfig)
    assert config.group == "flash_fc_top"

    train, valid = make_dataset_splits(config)
    assert train.spikes.shape == (4, 4, 3)
    assert train.raw_spikes.shape == (4, 4, 2)
    assert valid.spikes.shape == (2, 4, 3)
    item = train[0]
    assert item["heldout_spikes"].shape == (4, 2)
    assert item["full_spikes"].shape == (4, 5)
    assert item["dt"].item() == pytest.approx(0.02)

    batch = next(iter(DataLoader(train, batch_size=2)))
    assert batch["spikes"].shape == (2, 4, 3)
    assert batch["heldout_spikes"].shape == (2, 4, 2)
