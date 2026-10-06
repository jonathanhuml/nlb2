import math
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nlb2.datasets import (
    ChaoticRNNDataset,
    ChaoticRNNDatasetConfig,
    CTDDataset,
    CTDDatasetConfig,
    LorenzDataset,
    LorenzDatasetConfig,
    NLBDataset,
    NLBDatasetConfig,
)
from nlb2.metrics import compute_available_metrics
from nlb2.metrics import evaluate_model
from nlb2.models import (
    BGPFAConfig,
    CASSMConfig,
    EnsembleDynamicsModel,
    GPFAConfig,
    KalmanConfig,
    LangevinFlowConfig,
    LFADSConfig,
    NDTConfig,
    PSTHConfig,
    SmoothingConfig,
    STNDTConfig,
)
from nlb2.types import ModelOutput, StepResult
from nlb2.training.strategies import build_strategy


def _assert_all_trainable_parameters_receive_gradients(model, loss, allowed_missing=()):
    allowed_missing = set(allowed_missing)
    model.zero_grad(set_to_none=True)
    loss.backward()
    missing = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None and name not in allowed_missing
    ]
    nonfinite = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad
        and param.grad is not None
        and not bool(param.grad.isfinite().all())
    ]
    assert missing == []
    assert nonfinite == []


def test_model_contracts_smoke():
    config = LorenzDatasetConfig(
        neurons=6,
        num_inits=2,
        num_trials=4,
        num_steps=16,
        burn_steps=20,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    x = batch["spikes"]

    cassm = CASSMConfig(projection_dim=3).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    cassm_out = cassm(x)
    cassm_loss = cassm.loss(batch, cassm_out)
    assert cassm.predict_rates(x).shape == x.shape
    assert cassm_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(cassm, cassm_loss.total)

    kalman = KalmanConfig().build(n_neurons=x.shape[-1], n_time=x.shape[1])
    kalman_out = kalman(x)
    kalman_loss = kalman.loss(batch, kalman_out)
    assert kalman.predict_rates(x).shape == x.shape
    assert kalman_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(kalman, kalman_loss.total)

    gpfa_config = GPFAConfig(latent_dim=2)
    gpfa = gpfa_config.build(n_neurons=x.shape[-1], n_time=x.shape[1])
    gpfa_out = gpfa(x)
    gpfa_loss = gpfa.loss(batch, gpfa_out)
    assert gpfa_out.latents.shape[:2] == x.shape[:2]
    assert gpfa_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(gpfa, gpfa_loss.total)

    gradient = build_strategy(gpfa_config.optimization)
    gradient.setup(gpfa)
    result = gradient.step(gpfa, batch, epoch=0)
    assert result.batch_size == x.shape[0]

    lfads = LFADSConfig(
        generator_dim=8,
        inferred_input_dim=1,
        factor_dim=4,
        g0_encoder_dim=8,
        controller_encoder_dim=8,
        controller_dim=8,
        keep_prob=1.0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    lfads_out = lfads(x)
    lfads_loss = lfads.loss(batch, lfads_out)
    assert lfads_out.rates.shape == x.shape
    assert lfads_out.latents.shape[:2] == x.shape[:2]
    assert lfads.predict_rates(x).shape == x.shape
    assert lfads_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(lfads, lfads_loss.total)

    langevin_flow = LangevinFlowConfig(
        hidden_size=8,
        transformer_feedforward=16,
        coordinated_dropout_rate=1.0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    langevin_out = langevin_flow(x)
    langevin_loss = langevin_flow.loss(batch, langevin_out)
    assert langevin_out.rates.shape == x.shape
    assert langevin_out.latents.shape[:2] == x.shape[:2]
    assert langevin_flow.predict_rates(x).shape == x.shape
    assert langevin_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(langevin_flow, langevin_loss.total)
    langevin_flow.eval()
    langevin_valid_out = langevin_flow(x)
    langevin_valid_loss = langevin_flow.loss(batch, langevin_valid_out, epoch=500)
    assert langevin_valid_loss.named_terms["kl_weight"] == 0.0
    assert torch.allclose(
        langevin_valid_loss.total,
        langevin_valid_loss.named_terms["reconstruction_nll"],
    )

    optimizer = torch.optim.Adam(langevin_flow.parameters(), lr=1e-3, weight_decay=0.2)
    langevin_flow.on_before_optimizer_step(optimizer, epoch=0)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.0)
    langevin_flow.on_before_optimizer_step(optimizer, epoch=250)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)

    ndt = NDTConfig(
        hidden_size=16,
        num_layers=1,
        embed_dim=2,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    ndt_out = ndt(x)
    ndt_loss = ndt.loss(batch, ndt_out)
    assert ndt_out.rates.shape == x.shape
    assert ndt_out.latents.shape[:2] == x.shape[:2]
    assert ndt.predict_rates(x).shape == x.shape
    assert ndt_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(ndt, ndt_loss.total)

    stndt = STNDTConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        do_contrast=True,
        contrast_lambda=0.1,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    stndt_out = stndt(x)
    stndt_loss = stndt.loss(batch, stndt_out)
    assert stndt_out.rates.shape == x.shape
    assert stndt_out.latents.shape[:2] == x.shape[:2]
    assert stndt.predict_rates(x).shape == x.shape
    assert stndt_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(
        stndt,
        stndt_loss.total,
        allowed_missing={
            "encoder.layers.0.spatial_self_attn.out_proj.weight",
            "encoder.layers.0.spatial_self_attn.out_proj.bias",
        },
    )

    stndt_ensemble = STNDTConfig(
        ensemble=True,
        ensemble_size=2,
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        do_contrast=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    stndt_ensemble_out = stndt_ensemble(x)
    stndt_ensemble_loss = stndt_ensemble.loss(batch, stndt_ensemble_out)
    assert isinstance(stndt_ensemble, EnsembleDynamicsModel)
    assert len(stndt_ensemble.members) == 2
    assert stndt_ensemble_out.rates.shape == x.shape
    assert stndt_ensemble_out.extras["ensemble_size"] == 2
    assert stndt_ensemble.predict_rates(x).shape == x.shape
    assert stndt_ensemble_loss.total.ndim == 0
    _assert_all_trainable_parameters_receive_gradients(
        stndt_ensemble,
        stndt_ensemble_loss.total,
        allowed_missing={
            "members.0.encoder.layers.0.spatial_self_attn.out_proj.weight",
            "members.0.encoder.layers.0.spatial_self_attn.out_proj.bias",
            "members.1.encoder.layers.0.spatial_self_attn.out_proj.weight",
            "members.1.encoder.layers.0.spatial_self_attn.out_proj.bias",
        },
    )


def test_langevin_flow_prediction_samples_average_log_rates():
    x = torch.zeros(1, 2, 3)
    model = LangevinFlowConfig(
        hidden_size=8,
        transformer_feedforward=16,
        coordinated_dropout_rate=1.0,
        prediction_samples=2,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    log_values = iter([math.log(2.0), math.log(8.0)])

    def fake_forward(batch, sample):
        del sample
        log_rates = torch.full_like(batch, next(log_values))
        return ModelOutput(rates=log_rates.exp(), extras={"log_rates": log_rates})

    model._forward = fake_forward
    rates = model.predict_rates(x)
    assert torch.allclose(rates, torch.full_like(x, 4.0))


def test_langevin_flow_defaults_and_current_bin_recurrent_inputs():
    config = LangevinFlowConfig()
    assert config.potential_kernel_size == 3
    assert config.transformer_heads == 2
    assert config.encoder_input_alignment == "current"

    model = config.build(n_neurons=2, n_time=3)
    assert model.potential.kernel_size == 3
    assert model.decoder.self_attn.num_heads == 2

    class RecordingGRUCell(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size
            self.calls = []

        def forward(self, input, hidden=None):
            del hidden
            self.calls.append(input.detach().clone())
            return input.new_zeros(input.shape[0], self.hidden_size)

    recorder = RecordingGRUCell(model.hidden_size)
    model.encoder = recorder
    model.eval()
    x = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)

    model._forward(x, sample=False)

    assert len(recorder.calls) == 3
    assert torch.equal(recorder.calls[0], x[:, 0])
    assert torch.equal(recorder.calls[1], x[:, 1])
    assert torch.equal(recorder.calls[2], x[:, 2])


def test_langevin_flow_uses_valid_forward_steps_when_train_forward_missing():
    class DatasetConfig:
        include_forward = True

    class TrainDataset:
        config = DatasetConfig()
        heldin_spikes = torch.zeros(2, 3, 4)
        raw_spikes = torch.zeros(2, 3, 2)
        heldin_forward_spikes = None
        heldout_forward_spikes = None

    class ValidDataset:
        heldin_forward_spikes = torch.zeros(2, 5, 4)
        heldout_forward_spikes = torch.zeros(2, 5, 2)

    class Data:
        n_neurons = 4
        n_time = 3
        train_dataset = TrainDataset()
        valid_dataset = ValidDataset()

    model = LangevinFlowConfig(
        hidden_size=8,
        transformer_feedforward=16,
    ).build_from_data(Data())

    assert model.fwd_steps == 5
    assert model.output_neurons == 6


def test_gradient_strategy_reduce_on_plateau_scheduler():
    x = torch.zeros(1, 2, 3)
    config = LangevinFlowConfig(
        hidden_size=8,
        transformer_feedforward=16,
        coordinated_dropout_rate=1.0,
        optimization={
            "name": "gradient",
            "optimizer": "Adam",
            "lr": 1e-3,
            "lr_scheduler": "ReduceLROnPlateau",
            "scheduler_factor": 0.5,
            "scheduler_patience": 0,
            "scheduler_min_lr": 1e-5,
        },
    )
    model = config.build(n_neurons=x.shape[-1], n_time=x.shape[1])
    strategy = build_strategy(config.optimization)
    strategy.setup(model)

    assert strategy.optimizer is not None
    assert strategy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    strategy.on_validation_end(model, 0, StepResult(loss=1.0, batch_size=1))
    strategy.on_validation_end(model, 1, StepResult(loss=1.1, batch_size=1))
    assert strategy.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_bgpfa_config_uses_differentiable_full_batch_strategy():
    config = BGPFAConfig(latent_dim=2, n_mc_train=1, n_mc_eval=1)
    assert config.optimization.name == "mgplvm_full_batch_gradient"
    strategy = build_strategy(config.optimization)
    assert strategy.name == "mgplvm_full_batch_gradient"
    assert strategy.steps_per_epoch == 1

    multi_step_strategy = build_strategy(
        BGPFAConfig(
            optimization={
                "name": "mgplvm_full_batch_gradient",
                "steps_per_epoch": 4,
            }
        ).optimization
    )
    assert multi_step_strategy.steps_per_epoch == 4

    with pytest.raises(ValueError, match="does not support optimization.name='em'"):
        BGPFAConfig(optimization={"name": "em"})


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="BGPFA smoke test requires scikit-learn",
)
def test_bgpfa_internal_core_smoke():
    config = LorenzDatasetConfig(
        neurons=4,
        num_inits=1,
        num_trials=2,
        num_steps=6,
        burn_steps=5,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    batch = next(iter(DataLoader(train_ds, batch_size=len(train_ds))))
    x = batch["spikes"]

    bgpfa = BGPFAConfig(
        latent_dim=1,
        n_mc_train=1,
        n_mc_eval=1,
        kl_burnin_epochs=0,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    strategy = build_strategy(BGPFAConfig().optimization)
    strategy.setup(bgpfa)
    result = strategy.step(bgpfa, batch, epoch=0)

    assert result.batch_size == x.shape[0]
    assert result.objective == "negative_elbo"
    assert bgpfa.predict_rates(x).shape == x.shape


def test_chaotic_rnn_dataset_contract():
    config = ChaoticRNNDatasetConfig(
        neurons=5,
        hidden_units=8,
        num_conditions=3,
        num_trials=4,
        num_steps=12,
        seed=0,
    )
    train_ds, valid_ds = ChaoticRNNDataset.make_splits(config)

    assert train_ds.spikes.shape == (9, 12, 5)
    assert valid_ds.spikes.shape == (3, 12, 5)
    assert train_ds.rates.shape == train_ds.spikes.shape
    assert train_ds.latents.shape == (9, 12, 8)

    sample = train_ds[0]
    assert set(sample) == {"spikes", "rates", "rates_unit", "latents", "dt"}
    assert sample["rates_unit"] == "counts"


def test_ctd_dataset_loads_generated_h5_contract(tmp_path: Path):
    path = tmp_path / "ctd.h5"
    train_spikes = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
    valid_spikes = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    train_activity = train_spikes + 0.5
    valid_activity = valid_spikes + 0.5
    train_latents = np.ones((3, 4, 2), dtype=np.float32)
    valid_latents = np.ones((2, 4, 2), dtype=np.float32) * 2
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_recon_data", data=train_spikes)
        handle.create_dataset("valid_recon_data", data=valid_spikes)
        handle.create_dataset("train_activity", data=train_activity)
        handle.create_dataset("valid_activity", data=valid_activity)
        handle.create_dataset("train_latents", data=train_latents)
        handle.create_dataset("valid_latents", data=valid_latents)

    config = CTDDatasetConfig(
        name="ctd_nbff",
        task="nbff",
        data_path=path,
        dt=0.01,
    )
    train_ds, valid_ds = CTDDataset.make_splits(config)

    assert train_ds.spikes.shape == train_spikes.shape
    assert valid_ds.rates.shape == valid_activity.shape
    assert train_ds.latents.shape == train_latents.shape
    assert train_ds[0]["dt"].item() == pytest.approx(0.01)
    assert set(train_ds[0]) == {"spikes", "rates", "rates_unit", "latents", "dt"}
    assert train_ds[0]["rates_unit"] == "counts"


def test_nlb_dataset_loads_grouped_20ms_h5(tmp_path: Path):
    path = tmp_path / "nlb.h5"
    heldin = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    heldout = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    with h5py.File(path, "w") as handle:
        group = handle.create_group("mc_rtt_20")
        group.create_dataset("train_spikes_heldin", data=heldin + 1)
        group.create_dataset("train_spikes_heldout", data=heldout + 1)
        group.create_dataset("eval_spikes_heldin", data=heldin)
        group.create_dataset("eval_spikes_heldout", data=heldout)

    config = NLBDatasetConfig(name="mc_rtt", data_path=str(path), bin_size_ms=20)
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (2, 3, 4)
    assert train_ds.raw_spikes.shape == (2, 3, 2)
    assert valid_ds[0]["dt"].item() == pytest.approx(0.02)
    assert set(train_ds[0]) == {
        "spikes",
        "heldin_spikes",
        "raw_spikes",
        "heldout_spikes",
        "dt",
    }


def test_nlb_dataset_uses_train_tensors_when_available(tmp_path: Path):
    path = tmp_path / "nlb_train_eval.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=np.ones((3, 4, 2), dtype=np.float32))
        handle.create_dataset("train_spikes_heldout", data=np.ones((3, 4, 1), dtype=np.float32) * 2)
        handle.create_dataset("eval_spikes_heldin", data=np.ones((2, 4, 2), dtype=np.float32) * 3)
        handle.create_dataset("eval_spikes_heldout", data=np.ones((2, 4, 1), dtype=np.float32) * 4)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (3, 4, 2)
    assert train_ds.raw_spikes.shape == (3, 4, 1)
    assert valid_ds.spikes.shape == (2, 4, 2)
    assert valid_ds.raw_spikes.shape == (2, 4, 1)
    assert train_ds[0]["spikes"].mean().item() == pytest.approx(1.0)
    assert valid_ds[0]["spikes"].mean().item() == pytest.approx(3.0)


def test_nlb_dataset_full_observed_mode_uses_full_train_and_zero_filled_eval(tmp_path: Path):
    path = tmp_path / "nlb_full_observed.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=np.ones((3, 4, 2), dtype=np.float32))
        handle.create_dataset("train_spikes_heldout", data=np.ones((3, 4, 1), dtype=np.float32) * 2)
        handle.create_dataset(
            "train_spikes_heldin_forward",
            data=np.ones((3, 2, 2), dtype=np.float32) * 3,
        )
        handle.create_dataset(
            "train_spikes_heldout_forward",
            data=np.ones((3, 2, 1), dtype=np.float32) * 4,
        )
        handle.create_dataset("eval_spikes_heldin", data=np.ones((2, 4, 2), dtype=np.float32) * 5)
        handle.create_dataset("eval_spikes_heldout", data=np.ones((2, 4, 1), dtype=np.float32) * 6)
        handle.create_dataset(
            "eval_spikes_heldin_forward",
            data=np.ones((2, 2, 2), dtype=np.float32) * 7,
        )
        handle.create_dataset(
            "eval_spikes_heldout_forward",
            data=np.ones((2, 2, 1), dtype=np.float32) * 8,
        )

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="full_observed",
        include_forward=True,
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (3, 6, 3)
    assert valid_ds.spikes.shape == (2, 6, 3)
    assert train_ds.spikes[:, :4, 2].mean().item() == pytest.approx(2.0)
    assert train_ds.spikes[:, 4:, 2].mean().item() == pytest.approx(4.0)
    assert valid_ds.spikes[:, :4, 2].sum().item() == pytest.approx(0.0)
    assert valid_ds.spikes[:, 4:].sum().item() == pytest.approx(0.0)
    assert valid_ds.reconstruction_spikes[:, :4, 2].mean().item() == pytest.approx(6.0)
    assert valid_ds.reconstruction_spikes[:, 4:, 2].mean().item() == pytest.approx(8.0)
    assert "reconstruction_spikes" in train_ds[0]


def test_nlb_dataset_heldin_full_reconstruction_mode_uses_heldin_input_full_target(
    tmp_path: Path,
):
    path = tmp_path / "nlb_heldin_full_reconstruction.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=np.ones((3, 4, 2), dtype=np.float32))
        handle.create_dataset("train_spikes_heldout", data=np.ones((3, 4, 1), dtype=np.float32) * 2)
        handle.create_dataset(
            "train_spikes_heldin_forward",
            data=np.ones((3, 2, 2), dtype=np.float32) * 3,
        )
        handle.create_dataset(
            "train_spikes_heldout_forward",
            data=np.ones((3, 2, 1), dtype=np.float32) * 4,
        )
        handle.create_dataset("eval_spikes_heldin", data=np.ones((2, 4, 2), dtype=np.float32) * 5)
        handle.create_dataset("eval_spikes_heldout", data=np.ones((2, 4, 1), dtype=np.float32) * 6)
        handle.create_dataset(
            "eval_spikes_heldin_forward",
            data=np.ones((2, 2, 2), dtype=np.float32) * 7,
        )
        handle.create_dataset(
            "eval_spikes_heldout_forward",
            data=np.ones((2, 2, 1), dtype=np.float32) * 8,
        )

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="heldin_full_reconstruction",
        include_forward=True,
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)

    assert train_ds.spikes.shape == (3, 4, 2)
    assert valid_ds.spikes.shape == (2, 4, 2)
    assert train_ds.reconstruction_spikes.shape == (3, 6, 3)
    assert valid_ds.reconstruction_spikes.shape == (2, 6, 3)
    assert valid_ds.spikes.mean().item() == pytest.approx(5.0)
    assert valid_ds.reconstruction_spikes[:, :4, 2].mean().item() == pytest.approx(6.0)
    assert valid_ds.reconstruction_spikes[:, 4:, 2].mean().item() == pytest.approx(8.0)
    assert "reconstruction_spikes" in valid_ds[0]

    legacy = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="lfads_torch",
        include_forward=True,
    )
    assert legacy.input_mode == "heldin_full_reconstruction"


def test_gpfa_nlb_adapter_fits_heldout_decoder(tmp_path: Path):
    path = tmp_path / "gpfa_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = (train_heldin[..., :1] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :1] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = GPFAConfig(
        latent_dim=1,
        init_method="normal",
        init_seed=0,
        learn_kernel_params=False,
    ).build(n_neurons=train_ds.spikes.shape[-1], n_time=train_ds.spikes.shape[1])

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )
    heldin_rates = model.predict_rates(valid_ds.spikes)

    assert heldin_rates.shape == eval_heldin.shape
    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_smoothing_nlb_adapter_fits_heldout_decoder(tmp_path: Path):
    path = tmp_path / "smoothing_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 8, 3)).astype(np.float32)
    train_heldout = (train_heldin[..., :1] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 3)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :1] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = SmoothingConfig(
        kern_sd_ms=10.0,
        bin_size_ms=5.0,
        nlb_poisson_max_iter=20,
    ).build(n_neurons=train_ds.spikes.shape[-1], n_time=train_ds.spikes.shape[1])

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_langevin_flow_nlb_adapter_scores_direct_heldout_slice(tmp_path: Path):
    path = tmp_path / "langevin_flow_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = rng.poisson(0.3, size=(5, 6, 2)).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = rng.poisson(0.3, size=(2, 6, 2)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = LangevinFlowConfig(
        hidden_size=8,
        transformer_feedforward=16,
        coordinated_dropout_rate=1.0,
    )._build(
        n_neurons=train_ds.spikes.shape[-1],
        n_time=train_ds.spikes.shape[1],
        output_neurons=train_ds.spikes.shape[-1] + train_ds.raw_spikes.shape[-1],
    )
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert output.rates.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_ndt_nlb_adapter_scores_direct_heldout_slice(tmp_path: Path):
    path = tmp_path / "ndt_nlb.h5"
    rng = np.random.default_rng(4)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = (train_heldin[..., :2] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :2] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = NDTConfig(
        hidden_size=18,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        embed_dim=0,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    target = model._reconstruction_target(batch, output.extras["log_rates"])
    loss_mask = model._loss_mask_for_target(batch, output, target)
    loss = model.loss(batch, output)
    assert batch["spikes"].shape[-1] == train_heldin.shape[-1]
    assert output.rates.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert target.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert loss_mask[:, :, train_heldin.shape[-1] :].all()
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_ndt_nlb_direct_output_allows_padded_head_dimension(tmp_path: Path):
    path = tmp_path / "ndt_nlb_padded.h5"
    rng = np.random.default_rng(44)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = rng.poisson(0.4, size=(5, 6, 2)).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = rng.poisson(0.4, size=(2, 6, 2)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = NDTConfig(
        output_neurons=train_heldin.shape[-1] + train_heldout.shape[-1] + 1,
        hidden_size=18,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        embed_dim=0,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    target = model._reconstruction_target(batch, output.extras["log_rates"])
    loss_mask = model._loss_mask_for_target(batch, output, target)
    padded_target, padded_mask = model._pad_target_and_mask_to_rates(
        target,
        loss_mask,
        output.extras["log_rates"],
    )

    assert output.rates.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1] + 1
    assert padded_target.shape == output.rates.shape
    assert not padded_mask[..., -1].any()
    assert model.loss(batch, output).total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert np.isfinite(result.metrics["co_bps"])


def test_lfads_nlb_adapter_scores_direct_heldout_slice(tmp_path: Path):
    path = tmp_path / "lfads_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 6, 3)).astype(np.float32)
    train_heldout = rng.poisson(0.3, size=(5, 6, 2)).astype(np.float32)
    train_heldin_forward = rng.poisson(0.4, size=(5, 2, 3)).astype(np.float32)
    train_heldout_forward = rng.poisson(0.2, size=(5, 2, 2)).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 6, 3)).astype(np.float32)
    eval_heldout = rng.poisson(0.3, size=(2, 6, 2)).astype(np.float32)
    eval_heldin_forward = rng.poisson(0.4, size=(2, 2, 3)).astype(np.float32)
    eval_heldout_forward = rng.poisson(0.2, size=(2, 2, 2)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("train_spikes_heldin_forward", data=train_heldin_forward)
        handle.create_dataset("train_spikes_heldout_forward", data=train_heldout_forward)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)
        handle.create_dataset("eval_spikes_heldin_forward", data=eval_heldin_forward)
        handle.create_dataset("eval_spikes_heldout_forward", data=eval_heldout_forward)

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="heldin_full_reconstruction",
        include_forward=True,
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = LFADSConfig(
        generator_dim=12,
        inferred_input_dim=2,
        factor_dim=5,
        g0_encoder_dim=6,
        controller_encoder_dim=6,
        controller_dim=6,
        keep_prob=1.0,
        readout_neurons=train_heldin.shape[-1] + train_heldout.shape[-1],
        output_neuron_start=train_heldin.shape[-1],
        output_neurons=train_heldout.shape[-1],
        reconstruction_time_steps=train_heldin.shape[1] + train_heldin_forward.shape[1],
        controller_lag=1,
        dt=0.005,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert output.rates.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert output.rates.shape[1] == train_heldin.shape[1] + train_heldin_forward.shape[1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_stndt_nlb_adapter_scores_direct_heldout_slice(tmp_path: Path):
    path = tmp_path / "stndt_nlb.h5"
    rng = np.random.default_rng(0)
    train_heldin = rng.poisson(0.5, size=(5, 8, 4)).astype(np.float32)
    train_heldout = (train_heldin[..., :2] + 0.1).astype(np.float32)
    train_heldin_forward = rng.poisson(0.4, size=(5, 2, 4)).astype(np.float32)
    train_heldout_forward = rng.poisson(0.2, size=(5, 2, 2)).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 4)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :2] + 0.1).astype(np.float32)
    eval_heldin_forward = rng.poisson(0.4, size=(2, 2, 4)).astype(np.float32)
    eval_heldout_forward = rng.poisson(0.2, size=(2, 2, 2)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("train_spikes_heldin_forward", data=train_heldin_forward)
        handle.create_dataset("train_spikes_heldout_forward", data=train_heldout_forward)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)
        handle.create_dataset("eval_spikes_heldin_forward", data=eval_heldin_forward)
        handle.create_dataset("eval_spikes_heldout_forward", data=eval_heldout_forward)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = STNDTConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        do_contrast=False,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert output.rates.shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_stndt_nlb_adapter_scores_full_observed_heldout_slice(tmp_path: Path):
    path = tmp_path / "stndt_nlb_full_observed.h5"
    rng = np.random.default_rng(1)
    train_heldin = rng.poisson(0.5, size=(5, 8, 4)).astype(np.float32)
    train_heldout = (train_heldin[..., :2] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 4)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :2] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="full_observed",
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = STNDTConfig(
        hidden_size=18,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        do_contrast=False,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert batch["spikes"].shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert output.rates.shape[1] == train_heldin.shape[1]
    assert output.rates.shape[-1] == batch["spikes"].shape[-1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_ndt_nlb_adapter_scores_full_observed_heldout_slice(tmp_path: Path):
    path = tmp_path / "ndt_nlb_full_observed.h5"
    rng = np.random.default_rng(2)
    train_heldin = rng.poisson(0.5, size=(5, 8, 4)).astype(np.float32)
    train_heldout = (train_heldin[..., :2] + 0.1).astype(np.float32)
    train_heldin_forward = rng.poisson(0.4, size=(5, 2, 4)).astype(np.float32)
    train_heldout_forward = rng.poisson(0.2, size=(5, 2, 2)).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 4)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :2] + 0.1).astype(np.float32)
    eval_heldin_forward = rng.poisson(0.4, size=(2, 2, 4)).astype(np.float32)
    eval_heldout_forward = rng.poisson(0.2, size=(2, 2, 2)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("train_spikes_heldin_forward", data=train_heldin_forward)
        handle.create_dataset("train_spikes_heldout_forward", data=train_heldout_forward)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)
        handle.create_dataset("eval_spikes_heldin_forward", data=eval_heldin_forward)
        handle.create_dataset("eval_spikes_heldout_forward", data=eval_heldout_forward)

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="full_observed",
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = NDTConfig(
        hidden_size=18,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        dropout_rates=0.0,
        dropout_embedding=0.0,
        embed_dim=0,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert batch["spikes"].shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert output.rates.shape[1] == train_heldin.shape[1]
    assert output.rates.shape[-1] == batch["spikes"].shape[-1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_langevin_flow_nlb_adapter_scores_full_observed_heldout_slice(tmp_path: Path):
    path = tmp_path / "langevin_flow_nlb_full_observed.h5"
    rng = np.random.default_rng(3)
    train_heldin = rng.poisson(0.5, size=(5, 8, 4)).astype(np.float32)
    train_heldout = (train_heldin[..., :2] + 0.1).astype(np.float32)
    eval_heldin = rng.poisson(0.5, size=(2, 8, 4)).astype(np.float32)
    eval_heldout = (eval_heldin[..., :2] + 0.1).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)

    config = NLBDatasetConfig(
        name="mc_maze",
        data_path=str(path),
        bin_size_ms=5,
        input_mode="full_observed",
    )
    train_ds, valid_ds = NLBDataset.make_splits(config)
    data = type(
        "Data",
        (),
        {
            "n_neurons": train_ds.spikes.shape[-1],
            "n_time": train_ds.spikes.shape[1],
            "train_dataset": train_ds,
        },
    )()
    model = LangevinFlowConfig(
        hidden_size=8,
        potential_groups=4,
        transformer_heads=2,
        transformer_feedforward=16,
        dropout=0.0,
        prediction_samples=1,
    ).build_from_data(data)
    batch = next(iter(DataLoader(train_ds, batch_size=2)))
    output = model(batch["spikes"])
    loss = model.loss(batch, output)
    assert batch["spikes"].shape[-1] == train_heldin.shape[-1] + train_heldout.shape[-1]
    assert output.rates.shape[-1] == batch["spikes"].shape[-1]
    assert loss.total.ndim == 0

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=5),
    )

    assert result.predictions["rates"].shape == eval_heldout.shape
    assert result.targets["spikes"].shape == eval_heldout.shape
    assert "co_bps" in result.metrics
    assert np.isfinite(result.metrics["co_bps"])


def test_psth_nlb_adapter_uses_training_condition_psths(tmp_path: Path):
    path = tmp_path / "psth_nlb.h5"
    train_heldin = np.ones((2, 4, 2), dtype=np.float32)
    train_heldout = np.array(
        [
            [[2.0], [2.0], [2.0], [2.0]],
            [[3.0], [3.0], [3.0], [3.0]],
        ],
        dtype=np.float32,
    )
    eval_heldin = np.ones((3, 4, 2), dtype=np.float32)
    eval_heldout = np.array(
        [
            [[2.0], [2.0], [2.0], [2.0]],
            [[3.0], [3.0], [3.0], [3.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ],
        dtype=np.float32,
    )
    psth = np.zeros((2, 4, 3), dtype=np.float32)
    psth[0, :, 2] = 99.0
    psth[1, :, 2] = 99.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_spikes_heldin", data=train_heldin)
        handle.create_dataset("train_spikes_heldout", data=train_heldout)
        handle.create_dataset("eval_spikes_heldin", data=eval_heldin)
        handle.create_dataset("eval_spikes_heldout", data=eval_heldout)
        handle.create_dataset("psth", data=psth)
        cond_ds = handle.create_dataset("eval_cond_idx", (2,), dtype=h5py.vlen_dtype(np.dtype("int64")))
        cond_ds[0] = np.array([0, 2], dtype=np.int64)
        cond_ds[1] = np.array([1], dtype=np.int64)
        train_cond_ds = handle.create_dataset(
            "train_cond_idx",
            (2,),
            dtype=h5py.vlen_dtype(np.dtype("int64")),
        )
        train_cond_ds[0] = np.array([0], dtype=np.int64)
        train_cond_ds[1] = np.array([1], dtype=np.int64)

    config = NLBDatasetConfig(name="mc_maze", data_path=str(path), bin_size_ms=5)
    train_ds, valid_ds = NLBDataset.make_splits(config)
    model = PSTHConfig(kern_sd_ms=0.0).build(
        n_neurons=train_ds.spikes.shape[-1],
        n_time=train_ds.spikes.shape[1],
    )

    result = evaluate_model(
        model,
        DataLoader(valid_ds, batch_size=2),
        train_loader=DataLoader(train_ds, batch_size=2),
    )

    assert np.allclose(result.predictions["rates"], eval_heldout)
    assert result.metrics["co_bps"] > 0.0


def test_metrics_skip_incompatible_nlb_shapes():
    predictions = {"rates": torch.ones(2, 3, 4)}
    targets = {"spikes": torch.ones(2, 3, 2)}

    assert compute_available_metrics(predictions, targets) == {}


def test_gpfa_warns_when_configured_for_em():
    with pytest.warns(RuntimeWarning, match="legacy full-dataset EM adapter"):
        GPFAConfig(optimization={"name": "em"})


def test_gpfa_initialization_methods_are_configurable():
    config = LorenzDatasetConfig(
        neurons=5,
        num_inits=2,
        num_trials=3,
        num_steps=8,
        burn_steps=5,
        seed=0,
    )
    train_ds, _ = LorenzDataset.make_splits(config)
    x = next(iter(DataLoader(train_ds, batch_size=2)))["spikes"]

    normal = GPFAConfig(
        latent_dim=2,
        init_method="normal",
        init_seed=123,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    kaiming = GPFAConfig(
        latent_dim=2,
        init_method="kaiming_normal",
        init_seed=123,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])
    fa = GPFAConfig(
        latent_dim=2,
        init_method="fa",
        init_seed=123,
        fa_max_iters=2,
        learn_kernel_params=False,
    ).build(n_neurons=x.shape[-1], n_time=x.shape[1])

    normal.initialize(x)
    kaiming.initialize(x)
    fa.initialize(x)

    assert normal.initialized
    assert kaiming.initialized
    assert fa.initialized
    assert normal.C.isfinite().all()
    assert kaiming.C.isfinite().all()
    assert fa.C.isfinite().all()
    assert normal._r_diag().isfinite().all()
    assert kaiming._r_diag().isfinite().all()
    assert fa._r_diag().isfinite().all()
    assert not torch.allclose(normal.C, kaiming.C)

    c_before = kaiming.C.detach().clone()
    out = kaiming(x)
    assert "Corth" in out.extras
    assert torch.allclose(kaiming.C, c_before)
