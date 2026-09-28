"""Raw-count and dataset-exposure contracts for the analytic posterior solver."""

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from nlb2.datasets.chaotic_rnn import ChaoticRNNDataset, ChaoticRNNDatasetConfig
from nlb2.datasets.lorenz import LorenzDataset, LorenzDatasetConfig
from nlb2.models.ilqr_vae import ILQRVAEConfig
from nlb2.preprocessing import PreprocessedDataset, PreprocessingConfig, PreprocessingStepConfig


@pytest.mark.parametrize("dataset_name", ["lorenz", "chaotic_rnn"])
def test_preprocessed_synthetic_data_uses_raw_counts_and_dataset_dt(dataset_name):
    if dataset_name == "lorenz":
        config = LorenzDatasetConfig(
            neurons=3, num_inits=2, num_trials=3, num_steps=4, burn_steps=10,
            spike_bin_size=0.2,
        )
        train, valid = LorenzDataset.make_splits(config)
    else:
        config = ChaoticRNNDatasetConfig(
            neurons=3, hidden_units=4, num_conditions=2, num_trials=3, num_steps=4,
            dt=0.01,
        )
        train, valid = ChaoticRNNDataset.make_splits(config)
    preprocessing = PreprocessingConfig(observations=[PreprocessingStepConfig(name="anscombe")])
    wrapped_train = PreprocessedDataset(train, preprocessing)
    wrapped_valid = PreprocessedDataset(valid, preprocessing)
    model_config = ILQRVAEConfig(
        latent_dim=4, input_dim=2, max_iter=1,
        readout_bias_initialization="empirical_rates",
    )
    model = model_config.build_from_data(SimpleNamespace(
        config=config, train_dataset=wrapped_train, n_neurons=3, n_time=4,
    ))
    assert model.dt == model.core.dt == train.arrays.dt
    assert model.core.n_neurons == 3
    reference = model_config.model_copy(update={"dt": train.arrays.dt}).build(3, 4)
    reference.initialize_readout_bias_from_counts(train.spikes)
    torch.testing.assert_close(model.core.bias, reference.core.bias)

    batch = next(iter(DataLoader(wrapped_train, batch_size=2)))
    prepared = model.loss_forward_observations(batch, batch["spikes"])
    torch.testing.assert_close(prepared, batch["raw_spikes"])
    assert not torch.equal(prepared, batch["spikes"])
    output = model(prepared)
    torch.manual_seed(3)
    transformed_loss = model.loss(batch, output).total
    torch.manual_seed(3)
    raw_loss = model.loss(batch["raw_spikes"], output).total
    torch.testing.assert_close(transformed_loss, raw_loss)
    transformed_loss.backward()
    assert torch.isfinite(transformed_loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    model.eval()
    adapter = model.evaluation_adapter("synthetic")
    assert adapter.use_raw_spikes
    result = adapter.evaluate(model, DataLoader(wrapped_valid, batch_size=2), torch.device("cpu"))
    with torch.no_grad():
        expected = model(valid.spikes).rates / model.dt
    torch.testing.assert_close(torch.from_numpy(result.predictions["rates"]), expected)


@pytest.mark.parametrize("overrides", [
    {"dt": 0}, {"dt": -0.1}, {"dt": float("inf")}, {"dt": float("nan")},
    {"input_dim": 0}, {"latent_dim": 0}, {"latent_dim": 3, "input_dim": 2},
    {"n_posterior_samples": 0}, {"max_iter": -1}, {"ilqr_fallback_max_iter": -1},
    {"held_in_neurons": 0}, {"output_neuron_start": -1}, {"output_neurons": 0},
    {"include_elbo_constants": False}, {"solver": "lbfgs"},
    {"ilqr_failure_fallback": "lbfgs"},
])
def test_invalid_training_configs_fail_early(overrides):
    with pytest.raises(ValueError):
        ILQRVAEConfig(**overrides)


def test_inference_only_objective_can_omit_constants_and_use_lbfgs():
    ILQRVAEConfig(
        objective="posterior_control", include_elbo_constants=False,
        solver="lbfgs", trainable_parameters=False,
    )
