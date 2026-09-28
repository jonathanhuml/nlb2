from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from nlb2.config import ExperimentConfig
from nlb2.datasets import LorenzDataset, LorenzDatasetConfig, NLBDatasetConfig
from nlb2.experiment import Experiment
from nlb2.metrics import evaluate_model
from nlb2.models.mint import MINTConfig, InterpOptions, fit_poisson_interp
from nlb2.preprocessing import PreprocessedDataset, PreprocessingConfig
from torch.utils.data import DataLoader, Subset
from nlb2.training import TrainerConfig
from nlb2.training.strategies import build_strategy
from nlb2.types import ModelOutput


CUDA_DEVICE = pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
)


def _nlb_experiment(tmp_path, name="mc_maze", source="h5", device="cpu"):
    path = tmp_path / "prepared.h5"
    rng = np.random.default_rng(4)
    with h5py.File(path, "w") as handle:
        for split, n_trials in (("train", 4), ("eval", 2)):
            handle.create_dataset(f"{split}_spikes_heldin", data=rng.poisson(0.3, (n_trials, 12, 3)).astype("float32"))
            handle.create_dataset(f"{split}_spikes_heldout", data=rng.poisson(0.4, (n_trials, 12, 2)).astype("float32"))
        handle.create_dataset("train_cond_idx", data=np.array([[0, 1], [2, 3]]))
    cfg = ExperimentConfig(
        dataset=NLBDatasetConfig(name=name, data_path=str(path), bin_size_ms=5),
        model=MINTConfig(dataset=name, train_source=source, sigma=1, delta=2,
                         window_length=4, interp=0, causal=False, lfads_epochs=1,
                         lfads_generator_dim=4, lfads_factor_dim=3, lfads_encoder_dim=4,
                         lfads_controller_dim=4, lfads_batch_size=2),
        trainer=TrainerConfig(epochs=1, device=device), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path / "runs"), batch_size=2,
    )
    experiment = Experiment(cfg)
    model = experiment.build_model()
    return experiment, model


def _fit(experiment, model):
    experiment.trainer.fit(model, build_strategy(model.config.optimization),
                           experiment.data.train_loader(), experiment.data.valid_loader())


@pytest.mark.parametrize("dataset", ["area2_bump", "mc_maze", "dmfc_rsg", "mc_rtt"])
def test_prepared_nlb_fits_dynamic_neuron_counts_and_masks_heldout(tmp_path, dataset):
    experiment, model = _nlb_experiment(tmp_path, dataset)
    _fit(experiment, model)
    assert model.n_heldin == 3 and model.n_heldout == 2
    assert model.Ts == pytest.approx(0.005)
    assert model.library_training["trials"] == 4
    assert len(model.Omega_plus) == (4 if dataset == "mc_rtt" else 2)
    batch = next(iter(experiment.data.valid_loader()))
    first = model(batch["heldin_spikes"])
    contaminated = torch.cat([batch["heldin_spikes"], torch.full_like(batch["heldout_spikes"], 10000)], -1)
    second = model(contaminated)
    torch.testing.assert_close(first.rates, second.rates)
    assert first.rates.shape == (2, 12, 2)
    assert first.extras["full_rates"].shape == (2, 12, 5)
    assert torch.isfinite(first.rates).all()
    assert first.full_rates_unit == first.rates_unit == "counts"
    library = [item.clone() for item in model.Omega_plus]
    _fit(experiment, model)
    for before, after in zip(library, model.Omega_plus):
        torch.testing.assert_close(before, after)


@pytest.mark.parametrize("device", ["cpu", CUDA_DEVICE])
def test_fitted_checkpoint_restores_predictions_without_training_data(tmp_path, device):
    experiment, model = _nlb_experiment(tmp_path)
    _fit(experiment, model)
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"]
    expected = model(query).extras["full_rates"]
    model.to(device)
    fitted_tensors = model.Omega_plus + model.Phi_plus + [
        model.V, model.first_idx0, model.last_idx0, model.first_tau_prime_idx0,
        model.shifted_idx1, model.shifted_idx2, model.lambda_range, model.rates, model.L,
    ]
    assert all(item.device.type == device for item in fitted_tensors)
    torch.testing.assert_close(model(query.to(device)).extras["full_rates"].cpu(), expected)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = MINTConfig().build(n_neurons=3, n_time=12).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    torch.testing.assert_close(restored(query.to(device)).extras["full_rates"].cpu(), expected)
    restored.to("cpu")
    torch.testing.assert_close(restored(query).extras["full_rates"], expected)
    assert restored.library_training == model.library_training


@pytest.mark.parametrize("device", ["cpu", CUDA_DEVICE])
def test_raw_mc_rtt_trains_lfads_and_restores_template_checkpoint(tmp_path, device):
    experiment, model = _nlb_experiment(tmp_path, "mc_rtt", "lfads", device=device)
    _fit(experiment, model)
    assert model.library_training["lfads_epochs"] == 1
    assert np.isfinite(model.library_training["lfads_final_loss"])
    assert model.device.type == device
    assert all(trial.device.type == device for trial in model.Omega_plus)
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"].to(device)
    restored = model.config.build(3, 12).to(device)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(query).rates, model(query).rates)


@pytest.mark.parametrize("epoch_budget", [1, 200])
def test_standard_synthetic_experiment_fits_and_saves_mint(tmp_path: Path, epoch_budget):
    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=3, num_inits=2, num_trials=3,
                                    num_steps=8, burn_steps=4, train_fraction=0.67,
                                    spike_bin_size=0.25, seed=3),
        model=MINTConfig(sigma=1, window_length=2, delta=1, interp=0),
        trainer=TrainerConfig(epochs=epoch_budget), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path), batch_size=2,
    )
    experiment = Experiment(config)
    result = experiment.run()
    assert result.model_path.exists()
    assert len(experiment.model.Omega_plus) == 2
    assert experiment.model.config.dataset == "lorenz"
    assert experiment.model.Ts == 0.25
    assert np.isfinite(result.metrics["co_bps"])
    assert result.completed_epochs == len(result.history) == 1
    report = result.history[0]
    assert report.train.batch_size == len(experiment.data.train_dataset)
    assert report.valid.batch_size == len(experiment.data.valid_dataset)
    assert np.isfinite(report.train.loss) and np.isfinite(report.valid.loss)
    assert report.valid.loss == pytest.approx(result.metrics["poisson_nll"])
    assert report.seconds > 0


def test_library_fit_rejects_zero_training_epochs(tmp_path):
    experiment, model = _nlb_experiment(tmp_path)
    experiment.trainer.config.epochs = 0
    with pytest.raises(ValueError, match="library_fit requires epochs >= 1"):
        _fit(experiment, model)
    assert model.V is None
    assert not model.library_training


def test_mint_training_is_inside_recorded_epoch_and_resume_does_not_refit(tmp_path, monkeypatch):
    experiment, _ = _nlb_experiment(tmp_path)
    experiment.config.save_training_state = True
    experiment.config.dataset.split = "val"
    from nlb2.models.mint import MINT

    original_fit = MINT.fit_training_data
    training_calls = []

    def fit(model, loader, *, device):
        assert model.V is None
        original_fit(model, loader, device=device)
        training_calls.append(model.library_training.copy())

    monkeypatch.setattr(MINT, "fit_training_data", fit)
    result = experiment.run()
    assert len(training_calls) == 1
    assert result.completed_epochs == 1
    saved = torch.load(result.model_path, weights_only=True)
    assert saved["_extra_state"]["library_training"] == training_calls[0]
    resumed = Experiment(experiment.config).run(resume_from=result.run_dir / "training_state.pt")
    assert len(training_calls) == 1
    assert resumed.completed_epochs == 1
    with np.load(result.predictions_path) as first, np.load(resumed.predictions_path) as second:
        np.testing.assert_array_equal(first["pred_rates"], second["pred_rates"])


def test_mint_loss_measures_raw_count_likelihood():
    model = MINTConfig().build(n_neurons=2, n_time=3)
    raw = torch.ones(1, 3, 2)
    transformed = raw * 100
    rates = raw * 2
    batch = {"spikes": transformed, "raw_spikes": raw}
    loss = model.loss(batch, ModelOutput(rates=rates))
    assert loss.total == pytest.approx(2 - np.log(2))
    torch.testing.assert_close(model.loss_forward_observations(batch, transformed), raw)
    assert loss.named_terms["poisson_nll"] == loss.total


def _iterative_lorenz_config(tmp_path, epochs=3):
    return ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=3, num_inits=2, num_trials=4,
                                    num_steps=8, burn_steps=4, seed=0),
        model=MINTConfig(dataset="lorenz", train_source="lfads", lfads_epochs=99,
                         lfads_batch_size=2, lfads_generator_dim=4, lfads_factor_dim=3,
                         lfads_encoder_dim=4, lfads_controller_dim=4,
                         delta=1, window_length=2, interp=0),
        trainer=TrainerConfig(epochs=epochs, live_eval_interval=1),
        preprocessing=PreprocessingConfig(), batch_size=2,
        output_dir=str(tmp_path), save_training_state=True, evaluation_seed=0,
    )


def test_iterative_mint_budget_updates_weights_and_library_each_epoch(tmp_path):
    config = _iterative_lorenz_config(tmp_path)
    experiment = Experiment(config)
    snapshots = []
    libraries = []

    def snapshot(report):
        if not report.final:
            saved = torch.load(report.checkpoint_path, weights_only=True)
            snapshots.append(saved["strategy"]["library_trainer"])
            libraries.append([x.clone() for x in experiment.model.Omega_plus])

    result = experiment.run(callback=snapshot)
    assert result.completed_epochs == len(result.history) == len(snapshots) == 3
    assert experiment.model.library_training["lfads_epochs"] == 3
    assert experiment.model.library_training["trials"] == 6
    for index, state in enumerate(snapshots, start=1):
        assert state["epochs_completed"] == index
        assert {int(item["step"]) for item in state["optimizer"]["state"].values()} == {3 * index}
    for previous, current in zip(snapshots, snapshots[1:]):
        assert any(not torch.equal(previous["estimator"][key], current["estimator"][key])
                   for key in current["estimator"])
    for previous, current in zip(libraries, libraries[1:]):
        assert any(not torch.equal(a, b) for a, b in zip(previous, current))
    assert all(np.isfinite(report.train.metrics["trajectory_loss"]) for report in result.history)
    assert all(np.isfinite(report.valid.loss) for report in result.history)


def test_iterative_mint_resume_preserves_optimizer_and_predictions(tmp_path):
    config = _iterative_lorenz_config(tmp_path)
    uninterrupted = Experiment(config).run()
    interrupted = Experiment(config).run(callback=lambda report: report.final or report.epoch != 1)
    assert interrupted.completed_epochs == 1
    resumed = Experiment(config).run(resume_from=interrupted.run_dir / "training_state.pt")
    assert resumed.completed_epochs == len(resumed.history) == 3
    assert [report.train.loss for report in resumed.history] == [report.train.loss for report in uninterrupted.history]
    with np.load(uninterrupted.predictions_path) as first, np.load(resumed.predictions_path) as second:
        np.testing.assert_array_equal(first["pred_rates"], second["pred_rates"])


def test_direct_mint_fit_retains_legacy_rate_training_budget(tmp_path):
    config = _iterative_lorenz_config(tmp_path)
    config.model.lfads_epochs = 2
    experiment = Experiment(config)
    model = experiment.build_model()
    model.fit_training_data(experiment.data.train_loader(), device="cpu")
    assert model.library_training["lfads_epochs"] == 2
    assert model.library_training["source"] == "lfads"
    assert np.isfinite(model.library_training["lfads_final_loss"])


def test_delta_counts_conversion_preserves_constant_training_intensity(tmp_path):
    experiment, model = _nlb_experiment(tmp_path)
    model.config.sigma = 0
    dataset = experiment.data.train_dataset
    dataset.heldin_spikes.fill_(0.25)
    dataset.raw_spikes.fill_(0.5)
    _fit(experiment, model)
    query = torch.full((1, 12, 3), 0.25)
    torch.testing.assert_close(model(query).rates, torch.full((1, 12, 2), 0.5, dtype=torch.float64))


def test_nlb_experiment_exports_same_counts_as_direct_predictions(tmp_path):
    experiment, _ = _nlb_experiment(tmp_path)
    with pytest.warns(RuntimeWarning, match="co-bps only"):
        result = experiment.run()
    query = next(iter(experiment.data.valid_loader()))["heldin_spikes"]
    expected = experiment.model(query).rates.detach().numpy()
    with h5py.File(result.run_dir / "nlb_submission.h5", "r") as handle:
        actual = handle["mc_maze"]["eval_rates_heldout"][()]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
    assert result.metrics["nlb_co_bps"] == pytest.approx(result.metrics["co_bps"], abs=1e-6)


def test_cli_routes_synthetic_mint_through_standard_training(tmp_path, monkeypatch):
    from argparse import Namespace
    from nlb2.cli import run_command

    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(neurons=2, num_inits=1, num_trials=3,
                                    num_steps=6, burn_steps=3, train_fraction=0.67),
        model=MINTConfig(dataset="lorenz", sigma=1, window_length=2, delta=1, interp=0),
        trainer=TrainerConfig(epochs=1), preprocessing=PreprocessingConfig(),
        output_dir=str(tmp_path), run_name="cli-mint", batch_size=2,
    )
    monkeypatch.setattr("nlb2.cli.build_experiment_config", lambda args: config)
    assert run_command(Namespace(config=None, resume_from=None)) == 0
    assert (tmp_path / "cli-mint" / "model.pt").exists()


def test_interpolation_does_not_stop_on_negative_newton_step():
    x1 = torch.tensor([0.672034, 1.251349, 5.312476, 4.094878], dtype=torch.float64)
    x2 = torch.tensor([5.102956, 3.884396, 2.468927, 3.123640], dtype=torch.float64)
    spikes = torch.tensor([7, 2, 1, 7], dtype=torch.float64)
    alpha = fit_poisson_interp(spikes, x1, x2, InterpOptions(), 0)
    assert alpha == pytest.approx(1.0)


@pytest.mark.parametrize("interp", [1, 2])
def test_interpolation_with_one_available_library_state(interp):
    model = MINTConfig(dataset="lorenz", sigma=0, delta=1, window_length=4,
                       interp=interp).build(2, 4)
    spikes = torch.ones(2, 4)
    model.fit_library([spikes], [spikes], np.array([0]))
    assert torch.isfinite(model(spikes.T.unsqueeze(0)).rates).all()
    with pytest.raises(ValueError, match="query duration"):
        model(spikes.T[:2].unsqueeze(0))


def test_unsmoothed_legacy_library_preserves_delta_count_units():
    model = MINTConfig(dataset="lorenz", sigma=0, delta=2, window_length=4,
                       interp=0).build(2, 8)
    spikes = torch.full((2, 8), 0.25)
    model.fit_library([spikes], [spikes], np.array([0]))
    expected = torch.full((1, 8, 2), 0.25, dtype=torch.float64)
    torch.testing.assert_close(model(spikes.T.unsqueeze(0)).rates, expected)


def test_synthetic_fit_and_evaluation_ignore_transformed_observations():
    config = LorenzDatasetConfig(neurons=2, num_inits=2, num_trials=3, num_steps=8,
                                 burn_steps=4, train_fraction=0.67, spike_bin_size=0.25)
    training, validation = LorenzDataset.make_splits(config)
    transformed = PreprocessingConfig(observations=[{"name": "anscombe"}])
    raw_model = MINTConfig(dataset="lorenz", sigma=1, window_length=2, interp=0).build(2, 8)
    transformed_model = raw_model.config.build(2, 8)
    raw_model.fit_training_data(DataLoader(training, batch_size=2), device="cpu")
    transformed_model.fit_training_data(DataLoader(PreprocessedDataset(training, transformed), batch_size=2), device="cpu")
    for raw, processed in zip(raw_model.Omega_plus, transformed_model.Omega_plus):
        torch.testing.assert_close(raw, processed)
    expected = evaluate_model(raw_model, DataLoader(validation, batch_size=2))
    actual = evaluate_model(transformed_model, DataLoader(PreprocessedDataset(validation, transformed), batch_size=2))
    np.testing.assert_allclose(actual.predictions["count_rates"], expected.predictions["count_rates"])
    assert actual.metrics["rate_mse"] == expected.metrics["rate_mse"]


@pytest.mark.parametrize("nested", [False, True])
def test_subset_preserves_training_condition_membership(nested):
    config = LorenzDatasetConfig(neurons=2, num_inits=2, num_trials=3, num_steps=8,
                                 burn_steps=4, train_fraction=0.67)
    training = LorenzDataset(config)
    subset = Subset(Subset(training, [3, 1, 0]), [0, 1]) if nested else Subset(training, [1, 3])
    model = MINTConfig(dataset="lorenz", sigma=1, window_length=2, interp=0).build(2, 8)
    model.fit_training_data(DataLoader(subset, batch_size=2), device="cpu")
    assert model._training_conditions.tolist() == [1, 1]
    assert len(model.Omega_plus) == 1
