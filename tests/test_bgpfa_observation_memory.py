"""Memory-efficient bGPFA variance must preserve predictions and gradients."""

import copy

import pytest
import torch

from nlb2.models.bgpfa import Bvfa, Gaussian


def observation_model(*, neurons=5, dimensions=3, trials=4, time=7, tied=True, dtype=torch.float64):
    samples = 1 if tied else trials
    scale_tril = torch.tril(torch.randn(samples, neurons, dimensions, dimensions) * 0.2)
    scale_tril.diagonal(dim1=-2, dim2=-1).copy_(
        torch.rand(samples, neurons, dimensions) + 0.5
    )
    return Bvfa(
        neurons, dimensions, time, trials, Gaussian(neurons),
        q_mu=torch.randn(samples, neurons, dimensions),
        q_sqrt=scale_tril,
        tied_samples=tied,
        learn_neuron_scale=True,
        ard=True,
        learn_scale=True,
    ).to(dtype=dtype)


def reference_prediction(model, x, *, full_cov=False, sample_idxs=None):
    mean_weights, scale_tril = model.prms
    if not model.tied_samples and sample_idxs is not None:
        mean_weights = mean_weights[sample_idxs]
        scale_tril = scale_tril[sample_idxs]
    scaled = model.scale * model.dim_scale * x
    mean = mean_weights.matmul(scaled)
    projected = scaled[..., None, :, :].transpose(-1, -2).matmul(scale_tril)
    variance = (
        projected.matmul(projected.transpose(-1, -2))
        if full_cov else projected.square().sum(-1)
    )
    return mean, variance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tied,sample_idxs", [(True, None), (False, None), (False, [3, 1])])
def test_variance_and_gradients_match_original_contraction(dtype, tied, sample_idxs):
    torch.manual_seed(23)
    model = observation_model(tied=tied, dtype=dtype)
    reference = copy.deepcopy(model)
    trials = 4 if sample_idxs is None else len(sample_idxs)
    x = torch.randn(3, trials, 3, 7, dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    mean, variance = model.predict(x, full_cov=False, sample_idxs=sample_idxs)
    reference_mean, reference_variance = reference_prediction(
        reference, reference_x, sample_idxs=sample_idxs
    )
    torch.testing.assert_close(mean, reference_mean)
    torch.testing.assert_close(variance, reference_variance)
    names = ("_q_mu", "_q_sqrt", "_scale", "_dim_scale", "_neuron_scale")
    gradients = torch.autograd.grad(
        (mean + 0.37 * variance).square().sum(),
        (x, *(getattr(model, name) for name in names)),
    )
    expected_gradients = torch.autograd.grad(
        (reference_mean + 0.37 * reference_variance).square().sum(),
        (reference_x, *(getattr(reference, name) for name in names)),
    )
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected)


def test_full_covariance_matches_diagonal_prediction():
    torch.manual_seed(5)
    model = observation_model()
    x = torch.randn(2, 4, 3, 7, dtype=torch.float64)
    _, diagonal = model.predict(x, full_cov=False)
    _, full = model.predict(x, full_cov=True)
    torch.testing.assert_close(diagonal, full.diagonal(dim1=-2, dim2=-1))


def test_diagonal_prediction_saves_smaller_backward_activations():
    torch.manual_seed(4)
    model = observation_model(neurons=64, dimensions=4, trials=8, time=32)
    x = torch.randn(3, 8, 4, 32, dtype=torch.float64, requires_grad=True)

    def largest_saved_tensor(predict):
        sizes = []

        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            predict()
        return max(sizes)

    previous = largest_saved_tensor(lambda: reference_prediction(model, x))
    current = largest_saved_tensor(lambda: model.predict(x, full_cov=False))
    assert current < previous / 2


def test_more_latents_than_neurons_keeps_original_contraction():
    torch.manual_seed(1)
    model = observation_model(neurons=2, dimensions=3)
    x = torch.randn(2, 4, 3, 7, dtype=torch.float64)
    actual = model.predict(x, full_cov=False)
    expected = reference_prediction(model, x)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
