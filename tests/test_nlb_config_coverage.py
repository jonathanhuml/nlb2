"""Keep the four core 5 ms NLB training recipes complete and validation-only."""

from pathlib import Path

import pytest

from nlb2.config import load_experiment_config


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "real"
DATASETS = ("mc_maze", "area2_bump", "mc_rtt", "dmfc_rsg")
MODELS = (
    "bgpfa", "cassm", "gpfa", "ilqr_vae", "kalman", "langevin_flow",
    "lfads", "mint", "ndt", "psth", "smoothing", "stndt",
)


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("model", MODELS)
def test_core_5ms_validation_recipe_exists_and_can_train(dataset, model):
    paths = sorted((CONFIG_ROOT / dataset / model).glob(f"{model}_{dataset}_nlb_5ms*.yaml"))
    assert paths, f"Missing 5 ms NLB recipe for {model} on {dataset}"
    # The unsuffixed recipe sorts before optional variants; some methods use
    # the established _train or _nlb2 suffix for their canonical recipe.
    config = load_experiment_config(str(paths[0]))
    assert config.dataset.name == dataset
    assert config.dataset.bin_size_ms == 5
    assert config.dataset.split == "val"
    assert config.dataset.data_path is None
    assert config.model.name == model
    if model == "mint":
        assert config.model.optimization.name == "library_fit"
        assert config.trainer.epochs == (config.model.lfads_epochs if config.model.train_source == "lfads" else 1)
    elif model not in {"psth", "smoothing"}:
        assert config.model.optimization.name != "inference_only"
        assert config.trainer.epochs > 0
    if model in {"gpfa", "kalman"}:
        assert config.model.optimization == type(config.model)().optimization
        assert config.trainer.epochs == 20
        assert config.batch_size == 64
    if model == "gpfa":
        assert config.model.init_method == "kaiming_normal"
        assert config.model.learn_kernel_params


@pytest.mark.parametrize("dataset", DATASETS)
def test_core_5ms_recipe_variants_do_not_point_to_test_data(dataset):
    for model in MODELS:
        if model in {"mint", "ilqr_vae"}:
            continue
        for path in (CONFIG_ROOT / dataset / model).glob("*5ms*.yaml"):
            config = load_experiment_config(str(path))
            assert config.dataset.split == "val", str(path)
            assert config.dataset.data_path is None, str(path)
