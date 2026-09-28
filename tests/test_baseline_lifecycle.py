from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from nlb2.config import ExperimentConfig
from nlb2.datasets import LorenzDatasetConfig
from nlb2.experiment import Experiment
from nlb2.metrics import evaluate_model
from nlb2.models.baselines import PSTHConfig, SmoothingConfig, _rates_from_nlb_condition_psth
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig


def test_psth_requires_training_and_rejects_empty_training():
    model = PSTHConfig(kern_sd_ms=0).build(n_neurons=2, n_time=4)
    with pytest.raises(RuntimeError, match="fitted on training data"):
        model(torch.ones(3, 4, 2))
    with pytest.raises(ValueError, match="training loader"):
        model.fit_training_data(None, device="cpu")
    with pytest.raises(ValueError, match="nonempty training"):
        model.fit_training_data([], device="cpu")


def test_psth_fit_uses_raw_training_counts_and_is_idempotent():
    model = PSTHConfig(kern_sd_ms=0).build(n_neurons=2, n_time=4)
    batches = [
        {"spikes": torch.full((2, 4, 2), 50.0), "raw_spikes": torch.ones(2, 4, 2)},
        {"spikes": torch.full((1, 4, 2), 50.0), "raw_spikes": torch.full((1, 4, 2), 2.0)},
    ]
    model.fit_training_data(batches, device="cpu")
    model.fit_training_data([torch.full((3, 4, 2), 100.0)], device="cpu")
    expected = torch.full((1, 4, 2), 4 / 3)
    torch.testing.assert_close(model(torch.zeros(1, 4, 2)).rates, expected)
    torch.testing.assert_close(model(torch.ones(1, 4, 2, dtype=torch.long)).rates, expected)
    with pytest.raises(ValueError, match="time/neuron dimensions"):
        model(torch.zeros(1, 3, 2))


def test_psth_fitted_checkpoint_round_trip_and_dimension_validation(tmp_path):
    config = PSTHConfig(kern_sd_ms=0)
    model = config.build(n_neurons=2, n_time=4)
    model.fit_training_data([torch.rand(3, 4, 2)], device="cpu")
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, weights_only=True)
    restored = config.build(n_neurons=2, n_time=4)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored(torch.zeros(2, 4, 2)).rates, model(torch.ones(2, 4, 2)).rates)
    incompatible = config.build(n_neurons=3, n_time=4)
    with pytest.raises(RuntimeError, match="checkpoint dimensions"):
        incompatible.load_state_dict(state)


def test_psth_training_epoch_saves_fitted_model(tmp_path):
    experiment = Experiment(ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=2, num_inits=2, num_trials=3,
                                    num_steps=8, burn_steps=4, train_fraction=0.67),
        model=PSTHConfig(kern_sd_ms=0), trainer=TrainerConfig(epochs=1),
        batch_size=2, output_dir=str(tmp_path), preprocessing=PreprocessingConfig(),
    ))
    result = experiment.run()
    assert result.completed_epochs == len(result.history) == 1
    assert np.isfinite(result.history[0].train.loss)
    assert np.isfinite(result.history[0].valid.loss)
    assert np.isfinite(result.metrics["rate_mse"])
    saved = torch.load(result.model_path, weights_only=True)
    assert saved["psth_rates"].shape == (8, 2)
    torch.testing.assert_close(saved["psth_rates"], experiment.data.train_dataset.spikes.mean(dim=0))


def test_smoothing_synthetic_adapter_uses_raw_counts_and_correct_units():
    model = SmoothingConfig(kern_sd_ms=0).build(n_neurons=2, n_time=4)
    raw = torch.ones(2, 4, 2)
    batch = dict(spikes=raw * 100, raw_spikes=raw, rates=raw * 4,
                 rates_unit=["hz", "hz"], dt=torch.full((2,), 0.25))
    result = evaluate_model(model, [batch])
    np.testing.assert_allclose(result.predictions["count_rates"], raw)
    assert result.metrics["rate_mse"] == 0
    assert model.loss(batch, model(raw)).total == 1


def test_nlb_psth_never_uses_evaluation_raw_spikes_as_training(tmp_path):
    path = tmp_path / "conditions.h5"
    with h5py.File(path, "w") as handle:
        handle["train_cond_idx"] = [[0]]
        handle["eval_cond_idx"] = [[0]]
    dataset = SimpleNamespace(config=SimpleNamespace(resolved_data_path=path),
                              raw_spikes=torch.full((1, 4, 2), 999.0))
    result = _rates_from_nlb_condition_psth(dataset, torch.zeros(1, 4, 2), torch.ones(1), 1e-9)
    assert result is None
