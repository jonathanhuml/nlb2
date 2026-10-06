"""bGPFA training, posterior identity, and checkpoint regression tests."""

import io
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from nlb2.config import ExperimentConfig, load_experiment_config
from nlb2.datasets import LorenzDatasetConfig
from nlb2.experiment import Experiment
from nlb2.metrics import evaluate_model
from nlb2.models import BGPFAConfig
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig
from nlb2.training.strategies import build_strategy


def small_config(**kwargs):
    return BGPFAConfig(
        latent_dim=2,
        binsize=5.0,
        ell0=2.0,
        n_mc_train=1,
        n_mc_eval=1,
        nlb_latent_infer_steps=3,
        nlb_latent_infer_n_mc=1,
        nlb_latent_infer_lr=0.02,
        nlb_decoder="ridge",
        optimization={
            "name": "mgplvm_full_batch_gradient",
            "n_mc": 1,
            "lr": 0.02,
        },
        **kwargs,
    )


@pytest.mark.parametrize("likelihood", ["gaussian", "poisson"])
@pytest.mark.parametrize("n_time", [7, 8])
def test_training_and_inference_without_mgplvm(monkeypatch, likelihood, n_time):
    # An installed copy of mgplvm must not mask a remaining runtime dependency.
    monkeypatch.setitem(sys.modules, "mgplvm", None)
    torch.manual_seed(5)
    x = torch.poisson(torch.ones(3, n_time, 3))
    config = small_config(likelihood=likelihood)
    model = config.build(n_neurons=3, n_time=n_time)
    strategy = build_strategy(config.optimization)
    strategy.setup(model)
    assert np.isfinite(strategy.step(model, x, epoch=0).loss)
    model.eval()
    output = model(x[:2] + 1)
    assert output.rates.shape == (2, n_time, 3)
    assert output.latents.shape == (2, n_time, 2)
    assert torch.isfinite(output.rates).all()
    assert torch.isfinite(output.latents).all()


@pytest.mark.parametrize("n_neurons,latent_dim", [(1, 1), (1, 3), (2, 2), (2, 3), (3, 5)])
@pytest.mark.parametrize("initialization", ["gp_prior", "fa"])
def test_low_neuron_training_preserves_all_latent_dimensions(n_neurons, latent_dim, initialization):
    torch.manual_seed(24)
    x = torch.poisson(torch.ones(4, 8, n_neurons))
    config = BGPFAConfig(
        latent_dim=latent_dim,
        ell0=2.0,
        latent_init=initialization,
        observation_init="fa" if initialization == "fa" else "mgplvm",
        n_mc_train=1,
        n_mc_eval=1,
        nlb_latent_infer_steps=2,
        nlb_latent_infer_n_mc=1,
        optimization={"name": "mgplvm_full_batch_gradient", "n_mc": 1, "lr": 0.02},
    )
    model = config.build(n_neurons=n_neurons, n_time=8)
    strategy = build_strategy(config.optimization)
    strategy.setup(model)
    for epoch in range(2):
        assert np.isfinite(strategy.step(model, x, epoch).loss)
    assert model._train_mod.obs.dim_scale.shape == (latent_dim, 1)
    assert (model._train_mod.obs.dim_scale > 0).all()
    assert all(torch.isfinite(value).all() for value in model.state_dict().values())
    model.eval()
    expected = model(x)
    assert expected.latents.shape == (4, 8, latent_dim)
    assert torch.isfinite(expected.rates).all()
    heldout = model(x[:2] + 1)
    assert heldout.latents.shape == (2, 8, latent_dim)
    assert torch.isfinite(heldout.rates).all()
    restored = config.build(n_neurons=n_neurons, n_time=8)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    torch.testing.assert_close(restored(x).rates, expected.rates, rtol=0, atol=0)


@pytest.mark.parametrize("n_time", [0, 1])
def test_bgpfa_rejects_time_axes_without_a_gp_interval(n_time):
    with pytest.raises(ValueError, match="at least two time bins"):
        BGPFAConfig().build(n_neurons=3, n_time=n_time)


@pytest.fixture(params=["gaussian", "poisson"])
def fitted(request):
    torch.manual_seed(42)
    x = torch.poisson(torch.ones(4, 8, 3))
    config = small_config(likelihood=request.param)
    model = config.build(n_neurons=3, n_time=8)
    strategy = build_strategy(config.optimization)
    strategy.setup(model)
    before = model.mgplvm_training_model(x).obs._q_mu.detach().clone()
    for epoch in range(3):
        result = strategy.step(model, x, epoch)
        assert np.isfinite(result.loss)
    assert not torch.equal(model._train_mod.obs._q_mu, before)
    model.eval()
    return config, model, x


def test_eval_batches_with_equal_size_use_their_own_posterior(fitted):
    _, model, x = fitted
    first = x[:2] + 1
    second = x[2:] + 2
    with torch.no_grad():
        mod_first = model.infer_latents(first, max_steps=3, n_mc=1, lrate=0.02)
        mod_second = model.infer_latents(second, max_steps=3, n_mc=1, lrate=0.02)
        second_output = model(second)
        first_output = model(first)
    assert second_output.extras["mgplvm_model"] is mod_second
    assert first_output.extras["mgplvm_model"] is not mod_second
    assert not torch.equal(first_output.latents, second_output.latents)
    assert not torch.equal(mod_first.lat_dist.lat_mu, mod_second.lat_dist.lat_mu)
    assert torch.isfinite(first_output.rates).all()
    assert torch.isfinite(second_output.rates).all()


def test_eval_equal_train_count_does_not_select_training_posterior(fitted):
    _, model, x = fitted
    with torch.no_grad():
        train_output = model(x.clone())
        eval_output = model(x + 1)
    assert train_output.extras["mgplvm_model"] is model._train_mod
    assert eval_output.extras["mgplvm_model"] is not model._train_mod
    assert not torch.equal(train_output.latents, eval_output.latents)


def test_permuted_training_rows_are_inferred_in_input_order(fitted):
    _, model, x = fitted
    reordered = x[torch.tensor([2, 0, 3, 1])]
    with torch.no_grad():
        output = model(reordered)
    assert output.extras["mgplvm_model"] is not model._train_mod
    assert torch.equal(model._eval_cache[0], reordered.double())
    assert not torch.equal(output.latents, model._train_mod.lat_dist.lat_mu)
    model.train()
    with pytest.raises(ValueError, match="same trial order"):
        model.mgplvm_training_model(reordered)


def test_inference_freezes_observation_model_and_learned_gp_prior(fitted):
    _, model, x = fitted
    learned_ell = model._train_mod.lat_dist._ell.detach().clone()
    train_state = {key: value.clone() for key, value in model.state_dict().items()}
    inferred = model.infer_latents(x[:2] + 1, max_steps=3, n_mc=1, lrate=0.02)
    torch.testing.assert_close(inferred.lat_dist._ell, learned_ell, rtol=0, atol=0)
    for key, expected in model._train_mod.obs.state_dict().items():
        torch.testing.assert_close(inferred.obs.state_dict()[key], expected, rtol=0, atol=0)
    for key, expected in train_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


def test_inference_inside_torch_inference_mode(fitted):
    _, model, x = fitted
    with torch.inference_mode():
        output = model(x + 1)
    assert torch.isfinite(output.latents).all()
    assert torch.isfinite(output.rates).all()


def test_nlb_readout_infers_training_batches_and_decodes_eval_in_one_pass(fitted, monkeypatch):
    _, model, x = fitted
    inferred = []
    original = model.infer_latents

    def record(batch, **kwargs):
        mod = original(batch, **kwargs)
        inferred.append((batch.clone(), mod.lat_dist.lat_mu.detach().cpu().clone()))
        return mod

    monkeypatch.setattr(model, "infer_latents", record)
    train_loader = [
        {"spikes": item, "heldout_spikes": item[..., :1] + 1}
        for item in x.split(2)
    ]
    eval_loader = [
        {"spikes": item, "heldout_spikes": item[..., :1] + 1}
        for item in (x + 2).split(2)
    ]
    result = evaluate_model(model, eval_loader, train_loader=train_loader)
    assert len(inferred) == 4
    for item, batch in zip(inferred, train_loader + eval_loader):
        torch.testing.assert_close(item[0], batch["spikes"].double())
    expected_latents = torch.cat([item[1] for item in inferred[2:]])
    np.testing.assert_allclose(result.predictions["latents"], expected_latents.numpy())
    assert result.predictions["rates"].shape == (4, 8, 1)
    assert np.isfinite(result.metrics["poisson_nll"])


def test_fresh_checkpoint_load_matches_fitted_predictions(fitted):
    config, model, x = fitted
    model.infer_latents(x[:2] + 1, max_steps=2, n_mc=1)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = config.build(n_neurons=3, n_time=8)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    assert restored._eval_cache is None
    with torch.no_grad():
        expected = model(x)
        actual = restored(x)
    torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)
    torch.testing.assert_close(actual.rates, expected.rates, rtol=0, atol=0)
    torch.manual_seed(7)
    first = model(x + 1)
    torch.manual_seed(7)
    second = restored(x + 1)
    torch.testing.assert_close(second.rates, first.rates, rtol=0, atol=0)


def test_legacy_checkpoint_load_without_training_identity_infers_eval(fitted):
    config, model, x = fitted
    state = model.state_dict()
    del state["_train_observations"]
    restored = config.build(n_neurons=3, n_time=8)
    restored.load_state_dict(state)
    restored.eval()
    assert restored._train_observations.numel() == 0
    assert restored(x).extras["mgplvm_model"] is not restored._train_mod


def test_loading_existing_model_preserves_optimizer_parameters(fitted):
    _, model, _ = fitted
    parameter = model._train_mod.obs._q_mu
    optimizer = torch.optim.Adam([parameter])
    state = {key: value.clone() for key, value in model.state_dict().items()}
    model.load_state_dict(state)
    assert optimizer.param_groups[0]["params"][0] is model._train_mod.obs._q_mu


def test_training_and_dtype_moves_clear_inference_state(fitted):
    _, model, x = fitted
    model(x + 1)
    assert model._eval_cache is not None
    model.train()
    assert model._eval_cache is None
    model.eval()
    model(x + 1)
    model.float()
    assert model._eval_cache is None
    assert model._train_mod.lat_dist.ts.dtype == torch.float32
    output = model(x + 1)
    assert output.latents.dtype == torch.float32
    assert output.rates.dtype == torch.float32
    assert torch.isfinite(output.rates).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_fit_inference_and_checkpoint():
    torch.manual_seed(6)
    config = small_config(dtype="float32")
    model = config.build(n_neurons=3, n_time=8).cuda()
    x = torch.poisson(torch.ones(4, 8, 3, device="cuda"))
    strategy = build_strategy(config.optimization)
    strategy.setup(model)
    result = strategy.step(model, x, epoch=0)
    assert np.isfinite(result.loss)
    model.eval()
    assert model(x + 1).rates.is_cuda
    restored = config.build(n_neurons=3, n_time=8)
    restored.load_state_dict(model.state_dict())
    restored.cuda().eval()
    torch.testing.assert_close(restored(x).rates, model(x).rates)
    restored.cpu()
    assert restored(x.cpu() + 1).rates.device.type == "cpu"


def test_standard_synthetic_experiment_infers_new_trials(tmp_path, monkeypatch):
    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(
            neurons=3, num_inits=2, num_trials=3, num_steps=8, burn_steps=5, seed=0
        ),
        model=small_config(),
        trainer=TrainerConfig(epochs=1),
        preprocessing=PreprocessingConfig(),
        batch_size=2,
        output_dir=str(tmp_path),
        run_name="bgpfa-smoke",
    )
    experiment = Experiment(config)
    calls = []
    model_type = type(config.model.build(3, 8))
    original = model_type.infer_latents

    def record(self, batch, **kwargs):
        calls.append(batch.detach().clone())
        return original(self, batch, **kwargs)

    monkeypatch.setattr(model_type, "infer_latents", record)
    result = experiment.run()
    assert calls
    assert np.isfinite(result.metrics["rate_mse"])
    assert result.model_path.exists()
    restored = config.model.build(3, 8)
    restored.load_state_dict(torch.load(result.model_path, weights_only=True))


@pytest.mark.parametrize("dataset", ["mc_maze", "area2_bump", "mc_rtt", "dmfc_rsg"])
def test_core_nlb_5ms_recipe_is_portable_and_trains_from_scratch(dataset):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs" / "experiment" / "real" / dataset / "bgpfa"
        / f"bgpfa_{dataset}_nlb_5ms.yaml"
    )
    config = load_experiment_config(str(path))
    assert config.dataset.name == dataset
    assert config.dataset.data_path is None
    assert config.dataset.split == "val"
    assert config.dataset.bin_size_ms == 5
    assert config.model.binsize == 5.0
    assert config.model.optimization.name == "mgplvm_full_batch_gradient"
    assert config.trainer.epochs > 0
    assert config.model.build(n_neurons=3, n_time=8)._train_mod is None
