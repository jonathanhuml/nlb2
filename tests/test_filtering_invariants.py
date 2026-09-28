"""Reference Gaussian identities for the Kalman and CASSM filtering cores."""

import pytest
import torch
from torch.distributions import MultivariateNormal, kl_divergence

from nlb2.models import CASSMConfig, KalmanConfig
from nlb2.models._filtering_core import (
    CASSMElboLoss,
    ComputationAwareFilterSmoother,
    _matern32_time_process_cov,
    _matern32_time_stationary_cov,
    _matern32_transition_matrix,
)


@pytest.mark.parametrize("dt,ell,variance", [(0.01, 0.2, 1.3), (0.2, 1.0, 2.0), (1.0, 0.5, 0.7)])
def test_matern_process_noise_preserves_stationary_covariance(dt, ell, variance):
    dt = torch.tensor(dt, dtype=torch.float64)
    ell = torch.tensor(ell, dtype=torch.float64, requires_grad=True)
    variance = torch.tensor(variance, dtype=torch.float64, requires_grad=True)
    transition = _matern32_transition_matrix(dt, ell)
    stationary = _matern32_time_stationary_cov(variance, ell)
    process = _matern32_time_process_cov(dt, variance, ell)
    expected = stationary - transition @ stationary @ transition.T
    torch.testing.assert_close(process, expected, atol=1e-12, rtol=1e-10)
    assert torch.linalg.eigvalsh(process).min() >= 0
    actual_grad = torch.autograd.grad(process.sum(), (ell, variance), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (ell, variance))
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("projection_dim", [-1, 0, 7])
def test_cassm_rejects_invalid_projection_dimensions(projection_dim):
    with pytest.raises(ValueError, match="projection_dim"):
        CASSMConfig(projection_dim=projection_dim).build(n_neurons=6, n_time=3)
    with pytest.raises(ValueError, match="projection_dim"):
        ComputationAwareFilterSmoother(projection_dim, 6, 3, torch.device("cpu"))


def test_sparse_cassm_rejects_nondivisible_neurons():
    with pytest.raises(ValueError, match="divisible"):
        CASSMConfig(projection_dim=2).build(n_neurons=5, n_time=3)


def test_dense_cassm_preserves_nondivisible_neurons():
    model = CASSMConfig(projection_dim=2, use_dense_projection=True).build(5, 3)
    x = torch.randn(2, 3, 5)
    assert torch.isfinite(model(x).extras["loss"])
    model.eval()
    assert model(x).rates.shape == x.shape


@pytest.mark.parametrize("config", [CASSMConfig(projection_dim=2), KalmanConfig()])
def test_legacy_model_saving_does_not_create_hardcoded_directories(tmp_path, monkeypatch, config):
    monkeypatch.chdir(tmp_path)
    config = config.model_copy(update={"save_model": True, "dataset_name": "example"})
    with pytest.raises(ValueError, match="use Experiment output_dir"):
        config.build(4, 3)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("dense", [False, True])
def test_cassm_full_projection_one_step_matches_gaussian_nll_and_gradients(dense):
    torch.manual_seed(1)
    model = CASSMConfig(projection_dim=2, use_dense_projection=dense).build(2, 1).double()
    x = torch.tensor([[[1.0, -1.0]], [[0.5, 0.3]]], dtype=torch.float64)
    actual = model(x).extras["loss"]
    core = model.core
    observation = core.observation_matrix.to_dense()
    prior_cov = core._build_dynamics()[1].to_dense()
    noise = torch.diag(core.softplus(core.obs_noise_values))
    mean = (observation @ core.belief_initial_state).squeeze(-1)
    marginal = MultivariateNormal(mean, covariance_matrix=observation @ prior_cov @ observation.T + noise)
    expected = -marginal.log_prob(x[:, 0]).mean() / core.dim
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    parameters = tuple(model.parameters())
    actual_grad = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad(expected, parameters, allow_unused=True)
    for parameter, grad, reference in zip(parameters, actual_grad, expected_grad):
        if reference is None:
            reference = torch.zeros_like(parameter)
        if grad is None:
            grad = torch.zeros_like(parameter)
        torch.testing.assert_close(grad, reference, rtol=1e-8, atol=1e-9)


def test_projected_cassm_elbo_matches_explicit_gaussian_posterior():
    torch.manual_seed(8)
    dtype = torch.float64
    neurons, projected_dim, states = 3, 2, 6
    root = torch.randn(states, states, dtype=dtype, requires_grad=True)
    covariance = root @ root.T + torch.eye(states, dtype=dtype)
    projection = torch.randn(projected_dim, neurons, dtype=dtype, requires_grad=True)
    log_noise = torch.randn(neurons, dtype=dtype, requires_grad=True)
    noise = log_noise.exp()
    observation = torch.eye(states, dtype=dtype)[0::2]
    projected_observation = projection @ observation
    projected_noise = (projection * noise) @ projection.T
    innovation = projected_observation @ covariance @ projected_observation.T + projected_noise
    chol = torch.linalg.cholesky(innovation)
    prior_mean = torch.randn(4, states, 1, dtype=dtype, requires_grad=True)
    data = torch.randn(4, neurons, 1, dtype=dtype)
    residual = data - observation @ prior_mean
    message = projected_observation.T @ torch.cholesky_solve(projection @ residual, chol)
    update = covariance @ message
    posterior_mean = prior_mean + update
    posterior_cov = covariance - covariance @ projected_observation.T @ torch.cholesky_solve(
        projected_observation @ covariance, chol
    )
    posterior_residual = data - observation @ posterior_mean
    actual = CASSMElboLoss()(
        posterior_residual, posterior_cov, noise, message, update,
        projected_noise, chol, True,
    )
    q = MultivariateNormal(posterior_mean.squeeze(-1), covariance_matrix=posterior_cov)
    p = MultivariateNormal(prior_mean.squeeze(-1), covariance_matrix=covariance)
    expected_likelihood = 0.5 * (
        (posterior_residual.squeeze(-1).square() / noise).sum(-1).mean()
        + ((observation @ posterior_cov @ observation.T).diagonal() / noise).sum()
        + noise.log().sum() + neurons * torch.log(torch.tensor(2 * torch.pi, dtype=dtype))
    )
    expected = expected_likelihood + kl_divergence(q, p).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    inputs = (root, projection, log_noise, prior_mean)
    actual_grad = torch.autograd.grad(actual, inputs, retain_graph=True)
    reference_grad = torch.autograd.grad(expected, inputs)
    for grad, reference in zip(actual_grad, reference_grad):
        torch.testing.assert_close(grad, reference, rtol=1e-8, atol=1e-9)
