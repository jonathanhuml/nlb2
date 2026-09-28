"""Input, target, and dataset-shape regressions for STNDT and LFADS."""

from types import SimpleNamespace

import pytest
import torch

from nlb2.datasets.chaotic_rnn import ChaoticRNNDataset, ChaoticRNNDatasetConfig
from nlb2.datasets.lorenz import LorenzDataset, LorenzDatasetConfig
from nlb2.models.lfads import LFADSConfig
from nlb2.models.stndt import STNDTConfig
from nlb2.preprocessing import PreprocessedDataset, PreprocessingConfig, PreprocessingStepConfig


def _config(name, **overrides):
    if name == "stndt":
        values = dict(num_heads=1, num_layers=1, hidden_size=8, do_contrast=False,
                      dropout=0.0, dropout_rates=0.0, dropout_embedding=0.0)
        values.update(overrides)
        return STNDTConfig(**values)
    values = dict(generator_dim=4, factor_dim=2, g0_encoder_dim=4,
                  controller_encoder_dim=4, controller_dim=4, keep_prob=1.0,
                  initialize_log_rate_bias=False)
    values.update(overrides)
    return LFADSConfig(**values)


@pytest.mark.parametrize("name", ["stndt", "lfads"])
@pytest.mark.parametrize("dataset_name", ["lorenz", "chaotic_rnn"])
def test_preprocessed_synthetic_build_does_not_add_fake_heldout_channels(name, dataset_name):
    if dataset_name == "lorenz":
        config = LorenzDatasetConfig(neurons=3, num_inits=2, num_trials=3, num_steps=4, burn_steps=10)
        train, _ = LorenzDataset.make_splits(config)
    else:
        config = ChaoticRNNDatasetConfig(neurons=3, hidden_units=4, num_conditions=2,
                                        num_trials=3, num_steps=4)
        train, _ = ChaoticRNNDataset.make_splits(config)
    wrapped = PreprocessedDataset(train, PreprocessingConfig(
        observations=[PreprocessingStepConfig(name="anscombe")],
    ))
    model = _config(name).build_from_data(SimpleNamespace(
        config=config, train_dataset=wrapped, n_neurons=3, n_time=4,
    ))
    model.eval()
    assert model(wrapped.spikes[:2]).rates.shape == (2, 4, 3)


def test_stndt_nlb_build_uses_actual_heldout_channels_through_wrapper():
    heldin = torch.zeros(2, 4, 3)
    dataset = SimpleNamespace(
        config=SimpleNamespace(input_mode="heldin"), spikes=heldin, heldin_spikes=heldin,
        raw_spikes=torch.zeros(2, 4, 2), heldin_forward_spikes=torch.zeros(2, 2, 3),
        heldout_forward_spikes=torch.zeros(2, 2, 2),
    )
    wrapper = SimpleNamespace(dataset=dataset, spikes=heldin, raw_spikes=heldin)
    model = _config("stndt").build_from_data(SimpleNamespace(
        train_dataset=wrapper, n_neurons=3, n_time=4,
    ))
    assert model.output_neurons == 5
    assert model.fwd_steps == 2


@pytest.mark.parametrize("embedding", ["linear", "identity", "spike"])
def test_stndt_double_precision_including_random_mask_replacement(embedding):
    model = _config("stndt", linear_embedder=embedding == "linear",
                    embed_dim=0 if embedding == "identity" else 1,
                    mask_ratio=1.0, mask_token_ratio=0.0, mask_random_ratio=1.0).build(3, 4).double()
    x = torch.ones(2, 4, 3, dtype=torch.int64)
    output = model(x)
    assert output.rates.dtype == torch.float64
    loss = model.loss(x, output).total
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("name", ["stndt", "lfads"])
def test_poisson_loss_uses_raw_targets_without_changing_encoder_input(name):
    model = _config(name).build(3, 4).eval()
    raw = torch.zeros(2, 4, 3)
    encoder_input = 2.0 * torch.sqrt(raw + 0.375)
    output = model(encoder_input)
    actual = model.loss({"spikes": encoder_input, "raw_spikes": raw}, output).total
    expected = model.loss(raw, output).total
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["stndt", "lfads"])
def test_missing_reconstruction_targets_are_masked_without_nan_gradients(name):
    model = _config(name).build(3, 4).eval()
    x = torch.ones(2, 4, 3)
    target = x.clone()
    target[0, 1, 2] = torch.nan
    output = model(x)
    loss = model.loss({"spikes": x, "reconstruction_spikes": target}, output).total
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="finite reconstruction target"):
        model.loss({"spikes": x, "reconstruction_spikes": torch.full_like(x, torch.nan)}, output)


def test_stndt_full_observed_loss_uses_reconstruction_instead_of_masked_inputs():
    model = _config("stndt").build(5, 4).eval()
    x = torch.ones(2, 4, 5)
    x[..., 3:] = 0.0
    target = torch.ones_like(x)
    target[..., 3:] = 4.0
    output = model(x)
    batch = {"spikes": x, "heldin_spikes": x[..., :3],
             "heldout_spikes": target[..., 3:], "reconstruction_spikes": target}
    actual = model.loss(batch, output).total
    expected = (output.rates - target * output.extras["log_rates"]).mean()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["stndt", "lfads"])
def test_nonfinite_encoder_input_fails_before_corrupting_model(name):
    model = _config(name).build(3, 4)
    with pytest.raises(ValueError, match="finite spike-count observations"):
        model(torch.full((2, 4, 3), torch.nan))
