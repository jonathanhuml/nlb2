"""NWB timestamp order must not change spike bins or behavioral filtering."""

import numpy as np
import pandas as pd
import pytest
from nlb_tools.nwb_interface import NWBDataset

from nlb2.datasets._nlb_resample import resample_nwb_dataset


def dataset_with_data(data):
    dataset = NWBDataset.__new__(NWBDataset)
    dataset.data = data
    dataset.bin_width = 1
    return dataset


@pytest.fixture
def clock_data():
    columns = pd.MultiIndex.from_tuples(
        [("spikes", 0), ("heldout_spikes", 1), ("hand_vel", 0)],
        names=("signal_type", "channel"),
    )
    values = np.zeros((12, 3), dtype=np.float32)
    values[[1, 3, 6, 11], 0] = 1
    values[:, 1] = np.arange(12)
    values[5:10, 1] = np.nan
    values[:, 2] = np.sin(np.arange(12))
    index = pd.to_timedelta(np.arange(12), unit="ms").rename("clock_time")
    return pd.DataFrame(values, index=index, columns=columns)


def test_unordered_clock_preserves_upstream_binning_filtering_and_missing_values(clock_data):
    expected = dataset_with_data(clock_data.copy())
    expected.resample(5)
    shuffled = clock_data.iloc[[0, 1, 2, 5, 6, 8, 9, 10, 11, 3, 4, 7]].copy()
    actual = dataset_with_data(shuffled)
    resample_nwb_dataset(actual, 5)
    pd.testing.assert_frame_equal(actual.data, expected.data)
    np.testing.assert_array_equal(actual.data["spikes"].to_numpy().ravel(), [2, 1, 1])
    np.testing.assert_allclose(
        actual.data["heldout_spikes"].to_numpy().ravel(), [10, np.nan, 21], equal_nan=True
    )
    assert actual.bin_width == 5
    assert actual.data.index.freq == pd.tseries.frequencies.to_offset("5ms")


def test_already_ordered_clock_matches_upstream_resampling(clock_data):
    expected = dataset_with_data(clock_data.copy())
    expected.resample(5)
    actual = dataset_with_data(clock_data.copy())
    resample_nwb_dataset(actual, 5)
    pd.testing.assert_frame_equal(actual.data, expected.data)


def test_missing_clock_rows_are_rejected_before_resampling(clock_data):
    data = clock_data.drop(clock_data.index[3])
    dataset = dataset_with_data(data)
    with pytest.raises(ValueError, match="regular clock"):
        resample_nwb_dataset(dataset, 5)
    assert dataset.data is data
    assert dataset.bin_width == 1


def test_duplicate_clock_rows_are_rejected(clock_data):
    data = pd.concat([clock_data, clock_data.iloc[:1]])
    with pytest.raises(ValueError, match="unique clock"):
        resample_nwb_dataset(dataset_with_data(data), 5)


def test_off_grid_timestamps_are_not_silently_rounded(clock_data):
    clock = clock_data.index.to_numpy(copy=True)
    clock[3] += np.timedelta64(1, "ns")
    clock_data.index = pd.TimedeltaIndex(clock)
    with pytest.raises(ValueError, match="regular clock"):
        resample_nwb_dataset(dataset_with_data(clock_data), 5)
