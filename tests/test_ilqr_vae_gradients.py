"""Numerical checks of solver derivatives and the from-scratch training path."""

from types import SimpleNamespace

import pytest
import torch

from nlb2.models.ilqr_vae import ILQRVAEConfig


def _model(**overrides):
    values = dict(latent_dim=4, input_dim=2, max_iter=3, dt=0.2, n_posterior_samples=2)
    values.update(overrides)
    config = ILQRVAEConfig(**values)
    return config, config.build(n_neurons=3, n_time=4)


def _spikes():
    return torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 2.0, 1.0], [2.0, 1.0, 0.0], [1.0, 3.0, 2.0]],
        dtype=torch.float64,
    )


def test_student_prior_derivatives_match_autograd_with_unequal_scales():
    _, model = _model()
    core = model.core
    with torch.no_grad():
        core.spatial_stds.copy_(torch.tensor([0.4, 1.7]))
    control = torch.tensor([[0.7, -0.3]], dtype=torch.float64, requires_grad=True)
    grad, hess = core._prior_grad_hess_t(core.n_beg, control)
    objective = lambda u: core._prior_nll_t(core.n_beg, u)
    expected_grad = torch.autograd.grad(objective(control), control)[0]
    expected_hess = torch.autograd.functional.hessian(objective, control).reshape(2, 2)
    torch.testing.assert_close(grad, expected_grad)
    torch.testing.assert_close(hess, expected_hess)


def test_ilqr_uses_the_full_observation_likelihood_including_last_bin():
    _, model = _model()
    core = model.core
    controls = torch.randn(5, 2, dtype=torch.float64) * 0.2
    for constants in (False, True):
        expected = core.posterior_objective(
            controls, _spikes(), held_in_neurons=3, include_constants=constants
        )
        actual = core.ilqr_objective(
            controls, _spikes(), held_in_neurons=3, include_constants=constants
        )
        torch.testing.assert_close(actual, expected)
    before = core.infer_controls(_spikes(), solver="ilqr", max_iter=3)
    changed = _spikes()
    changed[-1] += 2.0
    after = core.infer_controls(changed, solver="ilqr", max_iter=3)
    assert not torch.allclose(before.controls, after.controls)
    assert before.loss_history[-1] < before.loss_history[0]


@pytest.mark.parametrize("solver", ["ilqr", "adam"])
def test_posterior_control_derivative_matches_centered_finite_difference(solver):
    _, model = _model(solver=solver)
    weights = torch.linspace(0.2, 1.0, 10, dtype=torch.float64).reshape(5, 2)

    def objective():
        result = model.core.infer_controls(
            _spikes(), solver=solver, max_iter=3, differentiable=True
        )
        return torch.sum(weights * result.controls)

    names = ["c", "bias", "gain", "uf", "uh", "wh", "bh", "b", "first_step", "spatial_stds", "nu"]
    parameters = [getattr(model.core, name) for name in names]
    gradients = torch.autograd.grad(objective(), parameters)
    checked_nonzero = 0
    for parameter, gradient in zip(parameters, gradients):
        index = int(gradient.abs().argmax())
        analytic = gradient.flatten()[index]
        assert torch.isfinite(gradient).all()
        value = parameter.detach().flatten()[index].item()
        delta = 1.0e-5
        with torch.no_grad():
            parameter.flatten()[index] = value + delta
        plus = objective().detach()
        with torch.no_grad():
            parameter.flatten()[index] = value - delta
        minus = objective().detach()
        with torch.no_grad():
            parameter.flatten()[index] = value
        numerical = (plus - minus) / (2.0 * delta)
        torch.testing.assert_close(analytic, numerical, rtol=2.0e-3, atol=2.0e-6)
        checked_nonzero += int(abs(analytic.item()) > 1.0e-6)
    assert checked_nonzero >= 8


@pytest.mark.parametrize("force_fallback", [False, True])
def test_sampled_elbo_gradient_matches_finite_difference_including_fallback(force_fallback):
    _, model = _model(ilqr_fallback_max_iter=3)
    if force_fallback:
        infer = model.core.infer_controls

        def fail_ilqr(*args, **kwargs):
            if kwargs["solver"] == "ilqr":
                raise RuntimeError("iLQR backward pass failed")
            return infer(*args, **kwargs)

        model.core.infer_controls = fail_ilqr
    batch = _spikes().unsqueeze(0)

    def objective():
        torch.manual_seed(42)
        output = model(batch)
        assert bool(output.extras["inference_fallbacks"].item()) == force_fallback
        return model.loss(batch, output).total

    names = ["c", "first_step", "uh", "time_cov_d", "space_cov_t"]
    parameters = [getattr(model.core, name) for name in names]
    gradients = torch.autograd.grad(objective(), parameters)
    for parameter, gradient in zip(parameters, gradients):
        index = int(gradient.abs().argmax())
        analytic = gradient.flatten()[index]
        value = parameter.detach().flatten()[index].item()
        delta = 1.0e-5
        with torch.no_grad():
            parameter.flatten()[index] = value + delta
        plus = objective().detach()
        with torch.no_grad():
            parameter.flatten()[index] = value - delta
        minus = objective().detach()
        with torch.no_grad():
            parameter.flatten()[index] = value
        torch.testing.assert_close(analytic, (plus - minus) / (2.0 * delta), rtol=2e-3, atol=2e-6)


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
])
def test_from_scratch_training_is_finite_and_checkpoint_restores_predictions(tmp_path, device):
    config, model = _model(dt=0.005 if device == "cuda" else 0.2)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    batch = _spikes().unsqueeze(0).to(device)
    for _ in range(2 if device == "cuda" else 3):
        optimizer.zero_grad()
        loss = model.loss(batch, model(batch)).total
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        model.project_parameters()
    assert all(not torch.equal(before[name], value) for name, value in model.named_parameters())
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = config.build(n_neurons=3, n_time=4).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        torch.testing.assert_close(restored(batch).rates, model(batch).rates)


def test_build_from_data_derives_neuron_slices_and_dt():
    heldin = torch.zeros(2, 4, 3)
    dataset = SimpleNamespace(
        spikes=heldin, heldin_spikes=heldin, raw_spikes=torch.zeros(2, 4, 2),
        arrays=SimpleNamespace(dt=0.005),
    )
    data = SimpleNamespace(train_dataset=dataset, n_neurons=3, n_time=4)
    config = ILQRVAEConfig(latent_dim=4, input_dim=2)
    model = config.build_from_data(data)
    assert model.core.n_neurons == 5
    assert model.held_in_neurons == model.output_neuron_start == 3
    assert model.output_neurons == 2
    assert model.dt == model.core.dt == 0.005
    with pytest.raises(ValueError, match="disagrees"):
        config.model_copy(update={"held_in_neurons": 99}).build_from_data(data)
    with pytest.raises(ValueError, match="disagrees"):
        config.model_copy(update={"dt": 1.0}).build_from_data(data)


def test_synthetic_raw_spikes_are_not_duplicated_as_heldout_neurons():
    _, model = _model()
    x = _spikes().unsqueeze(0)
    actual = model.loss_forward_observations({"spikes": x, "raw_spikes": x}, x)
    torch.testing.assert_close(actual, x)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float32, torch.float64])
def test_discrete_count_inputs_preserve_floating_predictions_and_gradients(dtype):
    _, model = _model()
    counts = _spikes().unsqueeze(0).to(dtype)
    output = model(counts)
    expected = model(counts.to(torch.float64))
    expected_dtype = dtype if counts.is_floating_point() else model.core.c.dtype
    for actual, reference in (
        (output.rates, expected.rates),
        (output.latents, expected.latents),
        (output.extras["full_rates"], expected.extras["full_rates"]),
    ):
        assert actual.dtype == expected_dtype
        torch.testing.assert_close(actual, reference.to(expected_dtype))
    assert (output.rates != output.rates.round()).any()
    actual_grad = torch.autograd.grad(output.rates.sum(), model.core.c)[0]
    expected_grad = torch.autograd.grad(expected.rates.sum(), model.core.c)[0]
    assert actual_grad.abs().max() > 0
    torch.testing.assert_close(actual_grad, expected_grad)


def test_differentiable_lbfgs_rejects_unsupported_gradient_path():
    _, model = _model()
    with pytest.raises(ValueError, match="does not preserve"):
        model.core.infer_controls(_spikes(), solver="lbfgs", differentiable=True)
