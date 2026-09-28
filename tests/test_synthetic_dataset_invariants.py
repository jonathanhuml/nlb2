"""Synthetic data splits, observation units, and short-trial preprocessing."""

import numpy as np
import pytest
import torch
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from nlb2.datasets import ChaoticRNNDataset, ChaoticRNNDatasetConfig, LorenzDataset, LorenzDatasetConfig
from nlb2.preprocessing import smooth_firing_rate


@pytest.mark.parametrize("overrides", [
    {"neurons": 0}, {"num_inits": 0}, {"num_trials": 1}, {"num_steps": 0},
    {"burn_steps": -1}, {"train_fraction": 0}, {"train_fraction": 1},
    {"num_trials": 2, "train_fraction": 0.1}, {"latent_dt": 0},
    {"latent_dt": float("nan")}, {"spike_bin_size": -1},
    {"spike_bin_size": float("inf")}, {"base_rate": 0},
])
def test_lorenz_rejects_invalid_generation_and_empty_splits(overrides):
    with pytest.raises(ValueError):
        LorenzDatasetConfig(**overrides)


@pytest.mark.parametrize("overrides", [
    {"tau": float("nan")}, {"dt": float("inf")},
    {"max_firing_rate": float("nan")}, {"g": float("inf")},
    {"x0_std": -1}, {"x0_std": float("nan")},
])
def test_chaotic_rnn_rejects_nonfinite_dynamics_parameters(overrides):
    with pytest.raises(ValueError):
        ChaoticRNNDatasetConfig(**overrides)


@pytest.mark.parametrize("kind", ["lorenz", "chaotic_rnn"])
def test_synthetic_splits_share_condition_truth_but_not_spike_samples(kind):
    if kind == "lorenz":
        config = LorenzDatasetConfig(neurons=4, num_inits=3, num_trials=5, num_steps=10,
                                     burn_steps=10, train_fraction=0.6, spike_bin_size=0.25)
        train, valid = LorenzDataset.make_splits(config)
        train_truth = train.latents.reshape(3, 3, 10, -1)
        valid_truth = valid.latents.reshape(2, 3, 10, -1)
        torch.testing.assert_close(train_truth[0], valid_truth[0], rtol=0, atol=0)
        torch.testing.assert_close(train_truth[0], train_truth[1], rtol=0, atol=0)
        assert train[0]["rates_unit"] == "hz"
        assert train[0]["dt"].item() == config.spike_bin_size
    else:
        config = ChaoticRNNDatasetConfig(neurons=4, hidden_units=4, num_conditions=3,
                                        num_trials=5, num_steps=10, train_fraction=0.6,
                                        dt=0.01)
        train, valid = ChaoticRNNDataset.make_splits(config)
        train_truth = train.latents.reshape(3, 3, 10, -1)
        valid_truth = valid.latents.reshape(3, 2, 10, -1)
        torch.testing.assert_close(train_truth[:, 0], valid_truth[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(train_truth[:, 0], train_truth[:, 1], rtol=0, atol=0)
        assert train[0]["rates_unit"] == "counts"
        assert train[0]["dt"].item() == pytest.approx(config.dt)
        assert train.rates.max() <= config.max_firing_rate * config.dt
    assert len(train) == 9
    assert len(valid) == 6
    assert train.arrays is valid.arrays
    assert train.spikes.data_ptr() != valid.spikes.data_ptr()
    assert not torch.equal(train.spikes[:len(valid)], valid.spikes)


@pytest.mark.parametrize("kind", ["lorenz", "chaotic_rnn"])
def test_poisson_counts_match_declared_rate_units(kind):
    if kind == "lorenz":
        dataset = LorenzDataset(LorenzDatasetConfig(
            neurons=3, num_inits=2, num_trials=1000, num_steps=6, burn_steps=4,
            spike_bin_size=0.25, base_rate=2.0, seed=15,
        ))
        expected_counts = dataset.rates * dataset.arrays.dt
    else:
        dataset = ChaoticRNNDataset(ChaoticRNNDatasetConfig(
            neurons=3, hidden_units=3, num_conditions=2, num_trials=1000,
            num_steps=6, dt=0.01, seed=15,
        ))
        expected_counts = dataset.rates
    expected_total = expected_counts.sum().item()
    assert abs(dataset.spikes.sum().item() - expected_total) < 6 * np.sqrt(expected_total)


@pytest.mark.parametrize("time", [1, 3, 15, 40])
def test_smoothing_preserves_short_trial_length_and_alignment(time):
    x = torch.arange(2 * time * 3, dtype=torch.float64).reshape(2, time, 3)
    actual = smooth_firing_rate(x, sampling_precision=20, kern_sd_ms=50)
    window = gaussian(12, 2, sym=True)
    window /= window.sum()
    expected = convolve(x[0, :, 0].numpy(), window, mode="same", method="direct")
    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
    np.testing.assert_allclose(actual[0, :, 0].numpy(), expected)
    if time >= len(window):
        np.testing.assert_allclose(actual[0, :, 0].numpy(), np.convolve(x[0, :, 0], window, "same"))
