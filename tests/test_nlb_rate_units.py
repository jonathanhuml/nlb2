import numpy as np
import pytest
import torch

from nlb2.models import LFADSConfig, LangevinFlowConfig, NDTConfig, STNDTConfig
from nlb2.models.base import EnsembleDynamicsModel
from nlb2.models.ilqr_vae import ILQRVAEConfig
from nlb2.nlb_eval import _collect_full_rate_parts, nlb_bits_per_spike
from nlb2.types import ModelOutput


def _model(name):
    if name in {"ndt", "stndt"}:
        config_type = NDTConfig if name == "ndt" else STNDTConfig
        config = config_type(output_neurons=5, num_heads=1, num_layers=1, hidden_size=8)
    elif name == "langevin_flow":
        config = LangevinFlowConfig(output_neurons=5, hidden_size=4, transformer_feedforward=8)
    elif name == "lfads":
        config = LFADSConfig(
            generator_dim=4, factor_dim=2, g0_encoder_dim=4,
            controller_encoder_dim=4, controller_dim=4, readout_neurons=5,
            output_neuron_start=3, output_neurons=2, dt=0.005,
        )
    else:
        config = ILQRVAEConfig(
            initialization="random", params_path=None, latent_dim=4, input_dim=2,
            dt=0.005,
            held_in_neurons=3, output_neuron_start=3, output_neurons=2,
            max_iter=1, control_hessian_mode="fisher",
        )
    return config.build(n_neurons=3, n_time=4).eval()


@pytest.mark.parametrize("name", ["ndt", "stndt", "langevin_flow", "lfads", "ilqr_vae"])
def test_full_exports_match_direct_count_predictions(name):
    torch.manual_seed(3)
    model = _model(name)
    batch = {
        "spikes": torch.poisson(torch.full((2, 4, 3), 0.2)),
        "heldin_spikes": torch.ones(2, 4, 3),
        "heldout_spikes": torch.ones(2, 4, 2),
    }
    torch.manual_seed(5)
    with torch.no_grad():
        rates = model(batch["spikes"]).rates
    if name == "ilqr_vae":
        expected = rates.numpy()
    else:
        expected = rates[..., 3:].numpy()
        if name == "lfads":
            expected = expected * 0.005
    torch.manual_seed(5)
    parts = _collect_full_rate_parts(
        model=model, loader=[batch], device=torch.device("cpu"), dt=0.005,
        prediction_floor=1e-9,
    )
    np.testing.assert_allclose(parts["rates_heldout"], expected, rtol=1e-6, atol=1e-9)
    assert nlb_bits_per_spike(parts["rates_heldout"], batch["heldout_spikes"]) == pytest.approx(
        nlb_bits_per_spike(expected, batch["heldout_spikes"]), abs=1e-6
    )


def test_full_rate_units_are_independent_and_required():
    output = ModelOutput(
        rates=torch.full((1, 2, 1), 0.1), rates_unit="counts",
        extras={"full_rates": torch.full((1, 2, 3), 20.0)}, full_rates_unit="hz",
    )
    torch.testing.assert_close(output.count_rates(0.005), torch.full((1, 2, 1), 0.1))
    torch.testing.assert_close(output.count_rates(0.005, full=True), torch.full((1, 2, 3), 0.1))
    output.full_rates_unit = None
    with pytest.raises(ValueError, match="declare units"):
        output.count_rates(0.005, full=True)


def test_ensemble_preserves_hz_units():
    model = EnsembleDynamicsModel([_model("lfads"), _model("lfads")]).eval()
    with torch.no_grad():
        output = model(torch.ones(1, 4, 3))
    assert output.rates_unit == "hz"
    torch.testing.assert_close(output.count_rates(0.005), output.rates * 0.005)
