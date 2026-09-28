"""Validate NWB clock ordering before upstream NLB positional resampling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def resample_nwb_dataset(dataset: Any, bin_size_ms: int) -> None:
    """Sort clock rows, reject irregular sampling, then use NLB resampling.

    NWBDataset.load joins behavioral and spike series with an outer concat.
    Depending on pandas, missing behavioral timestamps can be appended after
    the existing rows instead of sorted. Upstream resample groups consecutive
    rows, so chronological ordering must be restored before binning.
    """
    index = dataset.data.index
    if not isinstance(index, pd.TimedeltaIndex) or index.empty or index.hasnans:
        raise ValueError("NLB resampling requires a nonempty timedelta clock without NaT.")
    if not index.is_unique:
        raise ValueError("NLB resampling requires unique clock timestamps.")
    ordered = dataset.data if index.is_monotonic_increasing else dataset.data.sort_index()
    native_step = np.timedelta64(pd.Timedelta(milliseconds=float(dataset.bin_width)).value, "ns")
    clock = ordered.index.to_numpy(dtype="timedelta64[ns]")
    if native_step <= np.timedelta64(0, "ns") or not np.all(np.diff(clock) == native_step):
        raise ValueError(
            "NLB resampling requires a regular clock matching the native bin width; "
            "missing or off-grid timestamps cannot be rebinned by row position."
        )
    dataset.data = ordered
    dataset.resample(bin_size_ms)
