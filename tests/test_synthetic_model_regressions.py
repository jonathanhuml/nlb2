import math
from types import SimpleNamespace

import pytest
import torch
from torch.distributions import Normal, kl_divergence

from nlb2.datasets import (
    ChaoticRNNDataset, ChaoticRNNDatasetConfig, LorenzDataset, LorenzDatasetConfig,
)
from nlb2.models import LangevinFlowConfig, NDTConfig
from nlb2.models.langevin_flow import LangevinFlow
from nlb2.datasets.nlb import NLBArrays, NLBDataset, NLBDatasetConfig
from nlb2.preprocessing import PreprocessedDataset, PreprocessingConfig


def small_config(method, **kwargs):
    if method == "ndt":
        return NDTConfig(hidden_size=8, num_layers=1, num_heads=2,
                         dropout=0, dropout_rates=0, dropout_embedding=0, **kwargs)
    return LangevinFlowConfig(hidden_size=4, transformer_feedforward=8,
                              dropout=0, coordinated_dropout_rate=1, **kwargs)


@pytest.mark.parametrize("method", ["ndt", "langevin_flow"])
@pytest.mark.parametrize("dataset_name", ["lorenz", "chaotic_rnn"])
def test_synthetic_raw_counts_are_not_heldout_neurons(method, dataset_name):
    if dataset_name == "lorenz":
        config = LorenzDatasetConfig(neurons=4, num_inits=2, num_trials=4,
                                     num_steps=8, burn_steps=10)
        train, _ = LorenzDataset.make_splits(config)
    else:
        config = ChaoticRNNDatasetConfig(neurons=4, hidden_units=8, num_conditions=2,
                                        num_trials=4, num_steps=8)
        train, _ = ChaoticRNNDataset.make_splits(config)
    wrapped = PreprocessedDataset(train)
    data = SimpleNamespace(config=config, train_dataset=wrapped, n_neurons=4, n_time=8)
    model = small_config(method).build_from_data(data)
    assert model.output_neurons == 4
    assert model(wrapped.spikes[:2]).rates.shape == (2, 8, 4)


@pytest.mark.parametrize("method", ["ndt", "langevin_flow"])
def test_full_observed_targets_use_heldout_counts(method):
    model = small_config(method).build(4, 3)
    heldin, heldout = torch.ones(2, 3, 2), torch.full((2, 3, 2), 7.)
    batch = {"spikes": torch.cat([heldin, torch.zeros_like(heldout)], -1),
             "heldin_spikes": heldin, "heldout_spikes": heldout}
    expected = torch.cat([heldin, heldout], -1)
    torch.testing.assert_close(model._reconstruction_target(batch, expected), expected)


@pytest.mark.parametrize("method", ["ndt", "langevin_flow"])
def test_discrete_inputs_follow_model_precision(method):
    model = small_config(method).build(4, 3).double().eval()
    output = model(torch.ones(2, 3, 4, dtype=torch.int64))
    assert output.rates.dtype == torch.float64
    assert torch.isfinite(output.rates).all()


def test_ndt_ignores_nonfinite_targets_and_has_finite_gradients():
    model = small_config("ndt").build(4, 3).eval()
    target = torch.ones(2, 3, 4)
    target[0, -1] = torch.nan
    output = model(target)
    loss = model.loss({"spikes": target}, output).total
    finite = torch.isfinite(target)
    log_rates = output.extras["log_rates"]
    expected = (log_rates.exp()[finite] - target[finite] * log_rates[finite]).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    with pytest.raises(ValueError, match="finite target"):
        model.loss({"spikes": torch.full_like(target, torch.nan)}, output)


def test_langevin_transition_kl_uses_distribution_mean_and_log_variance():
    model = small_config("langevin_flow", gamma=0.2, velocity_prior_var=0.7).build(4, 3)
    z, v = torch.randn(2, 4), torch.randn(2, 4)
    std = math.sqrt(2 * model.gamma)
    _, mean, deterministic_kl = model._langevin_step(z, v, std, sample=False)
    _, _, sampled_kl = model._langevin_step(z, v, std, sample=True)
    expected = kl_divergence(Normal(mean, std), Normal(torch.zeros_like(mean), math.sqrt(0.7)))
    expected = expected.sum() / z.shape[0]
    torch.testing.assert_close(deterministic_kl, expected)
    torch.testing.assert_close(sampled_kl, expected)


def test_langevin_rejects_zero_potential_groups():
    with pytest.raises(ValueError, match="potential_groups must be positive"):
        LangevinFlowConfig(potential_groups=0)


@pytest.mark.parametrize("method", ["ndt", "langevin_flow"])
def test_synthetic_poisson_target_uses_untransformed_counts(method):
    model = small_config(method).build(4, 3)
    raw = torch.zeros(2, 3, 4)
    batch = {"spikes": torch.full_like(raw, 1.2247), "raw_spikes": raw}
    torch.testing.assert_close(model._reconstruction_target(batch, raw), raw)


def test_ndt_full_observed_heldout_neurons_follow_sampled_mask():
    model = small_config("ndt").build(4, 3).train()
    x = torch.ones(2, 3, 4)
    output = model(x)
    sampled_mask = torch.zeros_like(x, dtype=torch.bool)
    sampled_mask[:, 0] = True
    output.extras["observed_loss_mask"] = sampled_mask
    batch = {"spikes": x, "heldin_spikes": x[..., :2], "heldout_spikes": x[..., 2:]}
    mask = model._loss_mask_for_target(batch, output, x)
    torch.testing.assert_close(mask, sampled_mask)


def test_ndt_unobserved_heldout_neurons_are_always_loss_targets():
    model = small_config("ndt", output_neurons=4).build(2, 3).train()
    x = torch.ones(2, 3, 2)
    output = model(x)
    output.extras["observed_loss_mask"] = torch.zeros_like(x, dtype=torch.bool)
    batch = {"spikes": x, "heldin_spikes": x, "heldout_spikes": x + 1}
    target = model._reconstruction_target(batch, output.extras["log_rates"])
    mask = model._loss_mask_for_target(batch, output, target)
    assert mask[..., 2:].all()
    assert not mask[..., :2].any()


def test_ndt_missing_masked_targets_do_not_fall_back_to_visible_or_padded_entries():
    model = small_config("ndt", output_neurons=6, fwd_steps=2).build(4, 3).train()
    x = torch.ones(2, 3, 4)
    x[:, -1] = torch.nan
    output = model(x)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, -1] = True
    output.extras["observed_loss_mask"] = mask
    loss = model.loss({"spikes": x}, output)
    assert loss.total.requires_grad
    assert loss.total.item() == 0
    assert loss.named_terms["mask_fraction"].item() == 0
    loss.total.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad) == 0
    with pytest.raises(ValueError, match="finite target"):
        model.loss({"spikes": torch.full_like(x, torch.nan)}, output)


@pytest.mark.parametrize("method", ["ndt", "langevin_flow"])
@pytest.mark.parametrize("input_mode", ["heldin", "full_observed"])
def test_preprocessed_nlb_metadata_preserves_input_and_full_readout_dimensions(method, input_mode):
    heldin, heldout = torch.ones(2, 3, 4), torch.full((2, 3, 2), 2.)
    arrays = NLBArrays(heldin, heldout, heldin, heldout, None, None, None, None, 0.005)
    dataset = NLBDataset(NLBDatasetConfig(name="mc_maze", input_mode=input_mode), arrays=arrays)
    wrapped = PreprocessedDataset(dataset, PreprocessingConfig(observations=[{"name": "anscombe"}]))
    data = SimpleNamespace(train_dataset=wrapped, n_neurons=wrapped.spikes.shape[-1], n_time=3)
    model = small_config(method).build_from_data(data)
    assert model.n_neurons == wrapped.spikes.shape[-1]
    assert model.output_neurons == 6
    assert model(wrapped.spikes).rates.shape == (2, 3, 6)


@pytest.mark.parametrize("fwd_steps", [0, 2])
@pytest.mark.parametrize("alignment", ["current", "upstream_lagged"])
def test_langevin_gru_observed_and_forward_alignment(fwd_steps, alignment):
    torch.manual_seed(43)
    config = small_config("langevin_flow", sample_eval=False, fwd_steps=fwd_steps,
                          encoder_input_alignment=alignment)
    model = config.build(4, 3).eval()
    assert model.encoder_input_alignment == alignment
    assert LangevinFlowConfig.model_validate_json(config.model_dump_json()).encoder_input_alignment == alignment
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 4
    with torch.no_grad():
        output = model(x)
        hidden = model.encoder(x[:, 0])
        hidden_steps = [hidden]
        for time in range(1, x.shape[1] + fwd_steps):
            input_index = time if alignment == "current" else time - 1
            if time >= x.shape[1]:
                input_index = -1
            hidden = model.encoder(x[:, input_index], hidden)
            hidden_steps.append(hidden)
        expected = torch.stack(hidden_steps, dim=1)
        torch.testing.assert_close(output.latents[..., 2 * model.hidden_size:], expected)
        changed = x.clone()
        changed[:, -1] += 10
        changed_output = model(changed)
    if fwd_steps or alignment == "current":
        assert not torch.equal(changed_output.latents[:, -1], output.latents[:, -1])
        assert not torch.equal(changed_output.rates, output.rates)
    else:
        torch.testing.assert_close(changed_output.latents, output.latents)
        torch.testing.assert_close(changed_output.rates, output.rates)


def test_langevin_rejects_unknown_encoder_alignment():
    with pytest.raises(ValueError, match="encoder_input_alignment"):
        LangevinFlowConfig(encoder_input_alignment="unknown")
    with pytest.raises(ValueError, match="encoder_input_alignment"):
        LangevinFlow(4, 3, 4, encoder_input_alignment="unknown")
