"""GPFA must preserve the module's floating-point precision throughout inference."""

import pytest
import torch

from nlb2.models import GPFAConfig


def make_model(dtype, *, init_method="kaiming_normal"):
    return GPFAConfig(
        latent_dim=2,
        init_method=init_method,
        init_seed=3,
        fa_max_iters=3,
        kernel_param_max_iters=2,
    ).build(n_neurons=3, n_time=6).to(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("init_method", ["kaiming_normal", "fa"])
def test_gpfa_forward_and_gradient_use_module_dtype(dtype, init_method):
    torch.manual_seed(2)
    x = torch.randn(4, 6, 3, dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    model = make_model(dtype, init_method=init_method)
    output = model(x)
    loss = model.loss(x, output).total
    assert output.rates.dtype == dtype
    assert output.latents.dtype == dtype
    assert output.reconstruction.dtype == dtype
    assert loss.dtype == dtype
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert parameter.grad.dtype == dtype
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gpfa_em_and_checkpoint_preserve_dtype(dtype):
    torch.manual_seed(4)
    x = torch.randn(4, 6, 3)
    model = make_model(dtype)
    loss = model.fit_em_epoch(x)
    assert loss.total.dtype == dtype
    assert torch.isfinite(loss.total)
    model.eval()
    expected = model(x)
    restored = make_model(dtype)
    restored.load_state_dict(model.state_dict())
    restored.eval()
    actual = restored(x)
    torch.testing.assert_close(actual.rates, expected.rates, rtol=0, atol=0)
    torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)
    assert actual.rates.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpfa_cuda_coerces_cpu_observations_to_module_device():
    torch.manual_seed(2)
    model = make_model(torch.float64).cuda()
    x = torch.randn(4, 6, 3)
    output = model(x)
    assert output.rates.device.type == "cuda"
    assert output.rates.dtype == torch.float64
    model.loss(x, output).total.backward()
    assert torch.isfinite(model.C.grad).all()
