from pathlib import Path

from nlb2.config import load_experiment_config
from nlb2.models.lfads import LFADSConfig


LFADS_NLB_CONFIGS = [
    Path("configs/experiment/real/area2_bump/lfads/lfads_area2_bump_nlb_5ms_nlb2.yaml"),
    Path("configs/experiment/real/dmfc_rsg/lfads/lfads_dmfc_rsg_nlb_5ms_nlb2.yaml"),
    Path("configs/experiment/real/mc_maze/lfads/lfads_mc_maze_nlb_5ms_nlb2.yaml"),
    Path("configs/experiment/real/mc_rtt/lfads/lfads_mc_rtt_nlb_5ms_nlb2.yaml"),
]


def test_lfads_nlb_experiment_configs_load():
    configs = [load_experiment_config(path) for path in LFADS_NLB_CONFIGS]

    assert all(isinstance(config.model, LFADSConfig) for config in configs)
    assert all(config.model.objective == "lfads_elbo" for config in configs)
    assert all(config.dataset.input_mode == "heldin_full_reconstruction" for config in configs)
    assert all(config.batch_size == 64 for config in configs)


def test_core_nlb_lfads_folders_contain_only_nlb2_configs():
    lfads_paths = sorted(
        path
        for dataset in ("mc_maze", "area2_bump", "mc_rtt", "dmfc_rsg")
        for path in (Path("configs/experiment/real") / dataset / "lfads").glob("*.yaml")
    )

    assert lfads_paths == sorted(LFADS_NLB_CONFIGS)
    assert all(path.name.endswith("_nlb2.yaml") for path in lfads_paths)


def test_allen_lfads_configs_load():
    paths = sorted(Path("configs/experiment/real/allen_vcn/lfads").glob("*.yaml"))
    assert paths
    for path in paths:
        assert isinstance(load_experiment_config(path).model, LFADSConfig)


def test_model_folder_has_single_generic_lfads_preset():
    lfads_model_paths = sorted(Path("configs/model").glob("*lfads*.yaml"))

    assert lfads_model_paths == [Path("configs/model/lfads.yaml")]
