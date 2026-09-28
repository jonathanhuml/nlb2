import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from nlb2.datasets import ChaoticRNNDataset, ChaoticRNNDatasetConfig, LorenzDataset, LorenzDatasetConfig
from nlb2.datasets.ctd import CTDArrays, CTDDataset, CTDDatasetConfig
from nlb2.metrics import (
    NLBCoSmoothingAdapter,
    SyntheticEvaluationAdapter,
    bits_per_spike,
    evaluate_model,
    poisson_negative_log_likelihood,
)
from nlb2.models import LFADSConfig, LangevinFlowConfig
from nlb2.models.base import BaseDynamicsModel
from nlb2.models.baselines import PSTHConfig
from nlb2.preprocessing import smooth_firing_rate
from nlb2.types import ModelOutput


class FixedRateModel(BaseDynamicsModel):
    def __init__(self, rates, *, unit="counts", output_field="rates"):
        super().__init__()
        self.register_buffer("prediction", rates)
        self.unit = unit
        self.output_field = output_field

    def forward(self, x):
        rates = self.prediction[: x.shape[0]]
        if self.output_field == "predict_rates":
            return ModelOutput(rates_unit=self.unit)
        return ModelOutput(**{self.output_field: rates}, rates_unit=self.unit)

    def predict_rates(self, x):
        return self.prediction[: x.shape[0]]

    def loss(self, batch, output, epoch=0):
        raise NotImplementedError("The fixed model only supplies evaluation predictions.")


def _batch():
    counts = torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.1], [0.2]]])
    dt = torch.tensor([0.005, 0.010])
    spikes = torch.tensor([[[0.0], [1.0], [0.0]], [[1.0], [0.0], [1.0]]])
    return counts, {
        "spikes": spikes,
        "raw_spikes": spikes,
        "rates": counts / dt[:, None, None],
        "rates_unit": ["hz", "hz"],
        "dt": dt,
    }


@pytest.mark.parametrize("unit", ["counts", "hz"])
@pytest.mark.parametrize("output_field", ["rates", "reconstruction", "predict_rates"])
def test_synthetic_metrics_separate_hz_from_poisson_counts(unit, output_field):
    counts, batch = _batch()
    prediction = counts if unit == "counts" else batch["rates"]
    model = FixedRateModel(prediction, unit=unit, output_field=output_field)
    result = SyntheticEvaluationAdapter().evaluate(model, [batch], torch.device("cpu"))
    np.testing.assert_allclose(result.predictions["rates"], batch["rates"], rtol=1e-6)
    np.testing.assert_allclose(result.predictions["count_rates"], counts, rtol=1e-6)
    assert result.metrics["rate_mse"] == pytest.approx(0.0, abs=1e-10)
    assert result.metrics["rate_r2"] == pytest.approx(1.0)
    assert result.metrics["co_bps"] == pytest.approx(bits_per_spike(counts, batch["raw_spikes"]))
    assert result.metrics["poisson_nll"] == pytest.approx(
        poisson_negative_log_likelihood(counts, batch["raw_spikes"]).mean().item()
    )


def test_count_target_rates_are_converted_to_hz():
    counts, batch = _batch()
    batch["rates"] = counts
    batch["rates_unit"] = ["counts", "counts"]
    result = SyntheticEvaluationAdapter().evaluate(
        FixedRateModel(counts), [batch], torch.device("cpu")
    )
    np.testing.assert_allclose(result.targets["rates"], counts / batch["dt"][:, None, None])
    assert result.metrics["rate_mse"] == 0.0


def test_nonunit_bins_reject_unknown_target_rate_units():
    counts, batch = _batch()
    del batch["rates_unit"]
    with pytest.raises(ValueError, match="must declare rates_unit"):
        SyntheticEvaluationAdapter().evaluate(FixedRateModel(counts), [batch], torch.device("cpu"))


@pytest.mark.parametrize("dt", [0.0, -0.1, float("nan"), float("inf")])
def test_rate_conversion_rejects_invalid_bin_widths(dt):
    output = ModelOutput(rates=torch.ones(2, 3, 1), rates_unit="hz")
    with pytest.raises(ValueError, match="finite positive"):
        output.count_rates(torch.tensor([0.005, dt]))


@pytest.mark.parametrize("unit", ["counts", "hz"])
def test_direct_nlb_predictions_use_counts_once(unit):
    counts, batch = _batch()
    batch["heldout_spikes"] = batch["spikes"]
    prediction = counts if unit == "counts" else batch["rates"]
    result = NLBCoSmoothingAdapter().evaluate(
        FixedRateModel(prediction, unit=unit), [batch], torch.device("cpu")
    )
    np.testing.assert_allclose(result.predictions["rates"], counts, rtol=1e-6)
    assert result.metrics["co_bps"] == pytest.approx(bits_per_spike(counts, batch["spikes"]))


def test_nlb_decoder_outputs_already_use_counts():
    counts, batch = _batch()
    batch["heldout_spikes"] = 2.0 * counts
    adapter = NLBCoSmoothingAdapter(feature_source="rates", ridge_alpha=0.0)
    model = FixedRateModel(counts)
    # Force decoder fitting by exposing a target with a different neuron count.
    batch["heldout_spikes"] = torch.cat([2.0 * counts, 3.0 * counts], dim=-1)
    adapter.fit(model, [batch], torch.device("cpu"))
    result = adapter.evaluate(model, [batch], torch.device("cpu"))
    np.testing.assert_allclose(result.predictions["rates"], batch["heldout_spikes"], rtol=1e-6)


def test_smoothed_observations_are_counts_and_poisson_targets_remain_raw():
    raw = torch.zeros(1, 30, 1)
    raw[:, 15] = 1.0
    smoothed = smooth_firing_rate(raw, sampling_precision=5, kern_sd_ms=5)
    torch.testing.assert_close(smoothed.sum(), raw.sum())
    batch = {
        "spikes": smoothed, "raw_spikes": raw, "rates": smoothed / 0.005,
        "rates_unit": "hz", "dt": torch.tensor([0.005]),
    }
    result = SyntheticEvaluationAdapter().evaluate(
        FixedRateModel(smoothed), [batch], torch.device("cpu")
    )
    assert result.metrics["rate_mse"] == 0.0
    assert result.metrics["poisson_nll"] == pytest.approx(
        poisson_negative_log_likelihood(smoothed, raw).mean().item()
    )


def test_psth_custom_adapter_uses_shared_synthetic_units():
    _, batch = _batch()
    model = PSTHConfig().build(n_neurons=1, n_time=3).eval()
    adapter = model.evaluation_adapter("synthetic")
    adapter.fit(model, [batch], torch.device("cpu"))
    counts = model(batch["spikes"]).rates
    batch["rates"] = counts / batch["dt"][:, None, None]
    result = adapter.evaluate(model, [batch], torch.device("cpu"))
    np.testing.assert_allclose(result.predictions["count_rates"], counts)
    assert result.metrics["rate_mse"] == 0.0


@pytest.mark.parametrize("dataset_kind", ["lorenz", "chaotic_rnn"])
def test_generated_dataset_rate_metadata_drives_hz_artifacts(dataset_kind):
    if dataset_kind == "lorenz":
        dataset = LorenzDataset(LorenzDatasetConfig(
            neurons=2, num_inits=1, num_trials=3, num_steps=8,
            burn_steps=5, spike_bin_size=0.005, seed=2,
        ))
        counts = dataset.rates * dataset.arrays.dt
    else:
        dataset = ChaoticRNNDataset(ChaoticRNNDatasetConfig(
            neurons=2, num_trials=3, num_steps=8, dt=0.005, seed=2,
        ))
        counts = dataset.rates
    result = SyntheticEvaluationAdapter().evaluate(
        FixedRateModel(counts), DataLoader(dataset, batch_size=len(dataset)), torch.device("cpu")
    )
    np.testing.assert_allclose(result.targets["rates"], counts / dataset.arrays.dt, rtol=1e-6)
    assert result.metrics["rate_mse"] == pytest.approx(0.0, abs=1e-9)


def test_ctd_imported_rate_units_are_explicitly_configurable(tmp_path):
    value = torch.ones(2, 4, 3)
    arrays = CTDArrays(value, value, value, value, value, value, dt=0.005)
    config = CTDDatasetConfig(data_path=tmp_path / "unused.h5", rates_unit="hz")
    dataset = CTDDataset(config, arrays=arrays)
    assert dataset[0]["rates_unit"] == "hz"
    assert CTDDatasetConfig(data_path=tmp_path / "unused.h5").rates_unit == "counts"


@pytest.mark.parametrize("method", ["lfads", "langevin_flow"])
def test_synthetic_evaluation_honors_prediction_averaging_and_rate_units(method):
    torch.manual_seed(10)
    if method == "lfads":
        config = LFADSConfig(
            generator_dim=4, factor_dim=2, g0_encoder_dim=4,
            controller_encoder_dim=4, controller_dim=4, keep_prob=1.0,
            prediction_samples=4, dt=0.05,
        )
    else:
        config = LangevinFlowConfig(
            hidden_size=4, transformer_feedforward=8, dropout=0.0,
            coordinated_dropout_rate=1.0, prediction_samples=4,
        )
    model = config.build(n_neurons=3, n_time=4).eval()
    x = torch.ones(2, 4, 3)
    with torch.no_grad():
        deterministic = model(x)
        torch.manual_seed(123)
        expected = model.predict_rates(x)
    assert not torch.allclose(expected, deterministic.rates)
    expected_counts = expected * 0.05 if deterministic.rates_unit == "hz" else expected
    expected_hz = expected_counts / 0.05
    batch = {"spikes": x, "rates": torch.ones_like(x), "rates_unit": "hz", "dt": 0.05}

    torch.manual_seed(123)
    result = evaluate_model(model, [batch])

    np.testing.assert_allclose(result.predictions["rates"], expected_hz, rtol=1e-5)
    np.testing.assert_allclose(result.predictions["count_rates"], expected_counts, rtol=1e-5)
    assert result.metrics["rate_mse"] == pytest.approx(float((expected_hz - 1).square().mean()), rel=1e-5)
