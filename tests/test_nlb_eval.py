import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from nlb2.cli import main
from nlb2.datasets import nlb as nlb_module
from nlb2.nlb_eval import nlb_bits_per_spike, score_nlb2_predictions


@pytest.mark.parametrize("missing", [
    ("train_spikes_heldin",), ("train_spikes_heldout",),
    ("train_spikes_heldin", "train_spikes_heldout"),
])
def test_training_rejects_missing_training_tensors(tmp_path, missing):
    path = tmp_path / "incomplete.h5"
    with h5py.File(path, "w") as handle:
        for key in ("train_spikes_heldin", "train_spikes_heldout", "eval_spikes_heldin", "eval_spikes_heldout"):
            if key not in missing:
                handle[key] = np.ones((2, 4, 3), dtype=np.float32)
    config = nlb_module.NLBDatasetConfig(data_path=str(path))
    with pytest.raises(ValueError, match="missing training tensors"):
        nlb_module.NLBDataset.make_splits(config)


def test_evaluation_only_loading_never_creates_training_arrays(tmp_path):
    path = tmp_path / "eval_only.h5"
    with h5py.File(path, "w") as handle:
        handle["eval_spikes_heldin"] = np.ones((2, 4, 3), dtype=np.float32)
        handle["eval_spikes_heldout"] = np.ones((2, 4, 2), dtype=np.float32)
    config = nlb_module.NLBDatasetConfig(data_path=str(path))
    dataset = nlb_module.NLBDataset(config, split="valid")
    assert len(dataset) == 2
    assert dataset.arrays.train_heldin_spikes is None
    assert dataset.arrays.train_heldout_spikes is None
    with pytest.raises(ValueError, match="Training requires"):
        nlb_module.NLBDataset(config, split="train", arrays=dataset.arrays)


def test_nlb_bits_per_spike_matches_manual_poisson_ratio():
    spikes = np.array([[[0.0, 2.0], [1.0, 0.0]], [[3.0, np.nan], [0.0, 1.0]]])
    rates = np.array([[[0.2, 1.8], [0.9, 0.1]], [[2.7, 1.0], [0.2, 1.2]]])

    score = nlb_bits_per_spike(rates, spikes)
    null = np.tile(
        np.nanmean(spikes, axis=(0, 1), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    mask = ~np.isnan(spikes)
    model_ll = np.sum(spikes[mask] * np.log(rates[mask]) - rates[mask])
    null_ll = np.sum(spikes[mask] * np.log(null[mask]) - null[mask])
    expected = (model_ll - null_ll) / np.nansum(spikes) / np.log(2.0)

    assert score == pytest.approx(expected)


def test_score_nlb2_predictions_npz(tmp_path: Path):
    path = tmp_path / "predictions.npz"
    spikes = np.array([[[1.0], [0.0]], [[2.0], [1.0]]])
    rates = np.array([[[1.1], [0.2]], [[1.7], [1.2]]])
    np.savez(path, pred_rates=rates, target_spikes=spikes)

    score = score_nlb2_predictions(path)

    assert score.co_bps == pytest.approx(nlb_bits_per_spike(rates, spikes))
    assert score.prediction_shape == rates.shape
    assert score.target_shape == spikes.shape


def test_score_nlb_cli_for_nlb2_predictions(tmp_path: Path, capsys):
    predictions = tmp_path / "predictions.npz"
    np.savez(
        predictions,
        pred_rates=np.array([[[0.9], [1.2]]]),
        target_spikes=np.array([[[1.0], [1.0]]]),
    )

    assert main(["score-nlb", "--predictions", str(predictions)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert "co_bps" in payload
    assert payload["prediction_shape"] == [1, 2, 1]


def test_synthetic_artifact_scoring_prefers_explicit_count_rates(tmp_path):
    path = tmp_path / "synthetic.npz"
    counts = np.array([[[0.1], [0.2]]])
    spikes = np.array([[[0.0], [1.0]]])
    np.savez(path, pred_rates=counts / 0.005, pred_count_rates=counts,
             target_spikes=spikes)
    assert score_nlb2_predictions(path).co_bps == pytest.approx(
        nlb_bits_per_spike(counts, spikes)
    )


def test_score_nlb_cli_for_evalai_h5_co_bps(tmp_path: Path, capsys):
    target_h5 = tmp_path / "target.h5"
    submission_h5 = tmp_path / "submission.h5"
    spikes = np.array([[[1.0], [0.0]]])
    rates = np.array([[[0.8], [0.2]]])
    with h5py.File(target_h5, "w") as handle:
        group = handle.create_group("mc_maze")
        group.create_dataset("eval_spikes_heldout", data=spikes)
    with h5py.File(submission_h5, "w") as handle:
        group = handle.create_group("mc_maze")
        group.create_dataset("eval_rates_heldout", data=rates)

    assert (
        main(
            [
                "score-nlb",
                "--submission-h5",
                str(submission_h5),
                "--target-h5",
                str(target_h5),
                "--dataset",
                "mc_maze",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["co_bps"] == pytest.approx(nlb_bits_per_spike(rates, spikes))


def test_resolve_target_h5_downloads_when_requested(tmp_path: Path, monkeypatch):
    def fake_urlretrieve(url, filename):
        with h5py.File(filename, "w") as handle:
            handle.create_dataset("target", data=np.array([1.0]))
        return filename, None

    monkeypatch.setattr(nlb_module, "urlretrieve", fake_urlretrieve)
    monkeypatch.chdir(tmp_path)

    output_dir = Path("downloaded")
    path = nlb_module._resolve_target_h5(None, download=True, output_dir=output_dir)

    assert path == output_dir / "eval_data_test.h5"
    assert h5py.is_hdf5(path)


def test_prepare_nlb_validation_split_does_not_resolve_test_target(tmp_path: Path, monkeypatch):
    def fail_resolve_target(*args, **kwargs):
        raise AssertionError("validation splits should not require eval_data_test.h5")

    def fake_prepare_validation_h5(**kwargs):
        return nlb_module.PreparedNLBFile(
            dataset=kwargs["dataset"],
            split="val",
            bin_size_ms=kwargs["bin_size_ms"],
            path=kwargs["output"],
            heldin_shape=(1, 2, 3),
            heldout_shape=(1, 2, 1),
        )

    monkeypatch.setattr(nlb_module, "_resolve_target_h5", fail_resolve_target)
    monkeypatch.setattr(nlb_module, "_prepare_validation_h5", fake_prepare_validation_h5)

    prepared = nlb_module.prepare_nlb_data(
        datasets=["mc_maze"],
        splits=["val"],
        bin_sizes_ms=[5],
        output_dir=tmp_path,
    )

    assert prepared[0].split == "val"
