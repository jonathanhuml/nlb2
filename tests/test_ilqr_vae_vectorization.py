"""Parity against the sequential solver before timestep derivative batching."""

import copy
from dataclasses import fields
from types import MethodType

import pytest
import torch

from nlb2.models.ilqr_vae import ILQRVAEConfig, _TapeStep


def sequential_objective(self, controls, spikes, *, held_in_neurons, include_constants=False):
    x = controls.new_zeros(1, self.n_latent)
    loss = controls.new_zeros(())
    for k in range(controls.shape[0]):
        u = controls[k:k + 1]
        loss = loss + self._prior_nll_t(k, u, include_constants=include_constants)
        obs_idx = k - self.n_beg
        if 0 <= obs_idx < spikes.shape[0]:
            loss = loss + self._poisson_nll_t(
                x, spikes[obs_idx:obs_idx + 1], held_in_neurons=held_in_neurons,
                include_constants=include_constants,
            )
        x = self._dynamics_step(k, x, u)
    return loss + self._poisson_nll_t(
        x, spikes[-1:], held_in_neurons=held_in_neurons, include_constants=include_constants,
    )


def sequential_tape(self, controls, spikes, *, held_in_neurons):
    x = controls.new_zeros(1, self.n_latent)
    tape = []
    for k in range(controls.shape[0]):
        u = controls[k:k + 1]
        a = self._dynamics_x(k, x, u)
        b = self._dynamics_u(k, x)
        rlu, rluu = self._prior_grad_hess_t(k, u)
        obs_idx = k - self.n_beg
        if 0 <= obs_idx < spikes.shape[0]:
            rlx, rlxx = self._poisson_grad_hess_t(
                x, spikes[obs_idx:obs_idx + 1], held_in_neurons=held_in_neurons,
            )
        else:
            rlx = controls.new_zeros(1, self.n_latent)
            rlxx = controls.new_zeros(self.n_latent, self.n_latent)
        rlux = controls.new_zeros(self.n_input, self.n_latent)
        tape.append(_TapeStep(x, u, a, b, rlx, rlu, rlxx, rluu, rlux))
        x = self._dynamics_step(k, x, u)
    return tape


@pytest.mark.parametrize("latent_dim,input_dim,n_time", [(4, 2, 1), (4, 4, 1), (4, 2, 8), (20, 5, 120)])
def test_vectorized_tape_preserves_each_derivative(latent_dim, input_dim, n_time):
    core = ILQRVAEConfig(latent_dim=latent_dim, input_dim=input_dim, dt=0.005).build(3, n_time).core
    generator = torch.Generator().manual_seed(4)
    controls = torch.randn(n_time + core.n_beg - 1, input_dim, generator=generator, dtype=torch.float64) * 0.2
    spikes = torch.poisson(torch.full((n_time, 3), 0.2, dtype=torch.float64), generator=generator)
    expected = sequential_tape(core, controls, spikes, held_in_neurons=3)
    actual = core._ilqr_tape(controls, spikes, held_in_neurons=3)
    assert len(actual) == len(expected)
    for original, batched in zip(expected, actual):
        for field in fields(_TapeStep):
            torch.testing.assert_close(getattr(batched, field.name), getattr(original, field.name), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("latent_dim,input_dim,n_time", [(4, 2, 1), (4, 4, 8), (20, 5, 120)])
def test_vectorized_solver_preserves_controls_losses_and_all_parameter_gradients(latent_dim, input_dim, n_time):
    n_neurons = 65 if n_time == 120 else 5
    model = ILQRVAEConfig(
        latent_dim=latent_dim, input_dim=input_dim, dt=0.005, max_iter=5,
        n_posterior_samples=2,
    ).build(n_neurons, n_time)
    reference = copy.deepcopy(model)
    reference.core._ilqr_tape = MethodType(sequential_tape, reference.core)
    reference.core.ilqr_objective = MethodType(sequential_objective, reference.core)
    generator = torch.Generator().manual_seed(7)
    spikes = torch.poisson(torch.full((2, n_time, n_neurons), 0.05, dtype=torch.float64), generator=generator)
    records = []
    for current in (reference, model):
        torch.manual_seed(8)
        output = current(spikes)
        loss = current.loss(spikes, output).total
        loss.backward()
        records.append((output, loss))
    original, actual = records
    for key in ("controls", "ilqr_evaluations", "inference_fallbacks", "posterior_objective"):
        torch.testing.assert_close(actual[0].extras[key], original[0].extras[key], rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(actual[0].rates, original[0].rates, rtol=1e-9, atol=1e-11)
    torch.testing.assert_close(actual[1], original[1], rtol=1e-9, atol=1e-11)
    for name, parameter in model.named_parameters():
        expected = dict(reference.named_parameters())[name].grad
        if expected is None:
            assert parameter.grad is None
            continue
        assert expected is not None and parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, expected, rtol=1e-8, atol=1e-10, msg=name)
