import h5py
import numpy as np
import pytest

from nlb2.config import ExperimentConfig
from nlb2.datasets.nlb import NLBDatasetConfig
from nlb2.experiment import Experiment
from nlb2.models import GPFAConfig
from nlb2.nlb_eval import prepare_nlb_selection_target
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig


def _config(tmp_path, split="val", live_eval_interval=0):
    path = tmp_path / "prepared.h5"
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as handle:
        for prefix, trials in (("train", 4), ("eval", 2)):
            for partition, neurons in (("heldin", 3), ("heldout", 2)):
                handle[f"{prefix}_spikes_{partition}"] = rng.poisson(
                    0.3, (trials, 4, neurons)
                ).astype("float32")
    return ExperimentConfig(
        dataset=NLBDatasetConfig(data_path=str(path), split=split, max_trials=4),
        model=GPFAConfig(latent_dim=2),
        trainer=TrainerConfig(epochs=1, live_eval_interval=live_eval_interval),
        preprocessing=PreprocessingConfig(),
        batch_size=2, output_dir=str(tmp_path / "runs"),
    )


def test_selection_target_is_copied_from_configured_validation_data(tmp_path):
    config = _config(tmp_path)
    target = prepare_nlb_selection_target(config.dataset, tmp_path / "run")
    with h5py.File(target) as actual, h5py.File(config.dataset.data_path) as source:
        np.testing.assert_array_equal(
            actual["mc_maze/eval_spikes_heldout"], source["eval_spikes_heldout"]
        )
    config.dataset.split = "test"
    with pytest.raises(ValueError, match="split='val'"):
        prepare_nlb_selection_target(config.dataset, tmp_path / "bad")


def test_test_split_never_reaches_training_validation_or_scheduler(tmp_path):
    result = Experiment(_config(tmp_path, split="test")).run()
    assert len(result.history) == 1
    assert result.history[0].valid is None
    assert np.isfinite(result.metrics["co_bps"])


def test_test_split_cannot_select_live_checkpoints(tmp_path):
    experiment = Experiment(_config(tmp_path, split="test", live_eval_interval=1))
    with pytest.raises(ValueError, match="checkpoint selection"):
        experiment.run()
