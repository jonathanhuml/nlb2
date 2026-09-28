from pathlib import Path

import numpy as np
import torch

from nlb2.config import load_experiment_config
from nlb2.models.base import BaseModelConfig
from nlb2.models.mint import MINTConfig, get_mint_config


def test_mint_model_is_registered():
    assert BaseModelConfig.registry["mint"] is MINTConfig


def test_mint_nlb_experiment_configs_load():
    paths = [
        Path("configs/experiment/real/area2_bump/mint/mint_area2_bump_nlb_5ms.yaml"),
        Path("configs/experiment/real/dmfc_rsg/mint/mint_dmfc_rsg_nlb_5ms.yaml"),
        Path("configs/experiment/real/dmfc_rsg/mint/mint_dmfc_rsg_nlb_5ms_lfads.yaml"),
        Path("configs/experiment/real/mc_maze/mint/mint_mc_maze_nlb_5ms.yaml"),
        Path("configs/experiment/real/mc_rtt/mint/mint_mc_rtt_nlb_5ms.yaml"),
    ]

    configs = [load_experiment_config(path) for path in paths]

    assert [config.model.dataset for config in configs] == ["area2_bump", "dmfc_rsg", "dmfc_rsg", "mc_maze", "mc_rtt"]
    assert [config.model.train_source for config in configs] == ["h5", "h5", "lfads", "h5", "lfads"]
    assert all(isinstance(config.model, MINTConfig) for config in configs)
    assert all(config.model.nlb_neural_state_defaults for config in configs)
    assert all(config.model.nwb_root is None for config in configs)
    assert all(config.model.mat_data_root is None for config in configs)
    assert all(config.model.target_h5 is None for config in configs)
    assert all(config.model.optimization.name == "library_fit" for config in configs)
    assert [config.trainer.epochs for config in configs] == [1, 1, 100, 1, 100]


def test_mint_lorenz_recipe_trains_trajectories_for_50_epochs():
    config = load_experiment_config("configs/experiment/synthetic/lorenz/mint/mint_lorenz_100.yaml")
    assert config.trainer.epochs == 50
    assert config.model.optimization.name == "library_fit"
    assert config.model.train_source == "lfads"
    assert config.model.lfads_epochs == 50


def test_mint_python_and_yaml_defaults_use_training_spikes():
    from nlb2.utils.yaml import load_yaml

    for config in [MINTConfig(), MINTConfig.model_validate(load_yaml("configs/model/mint.yaml"))]:
        assert config.train_source == "h5"
        assert config.lorenz_library_source == "smoothed_spikes"


def test_mint_dmfc_uses_paper_trajectory_defaults():
    settings, hyperparams = get_mint_config("dmfc_rsg")

    assert (settings.trial_alignment.start, settings.trial_alignment.stop) == (-1950, 750)
    assert (settings.test_alignment.start, settings.test_alignment.stop) == (-1500, 0)
    assert hyperparams.sigma == 55
    assert hyperparams.n_neural_dims == 49
    assert hyperparams.n_cond_dims == 17
    assert hyperparams.n_trial_dims is None
    assert hyperparams.Delta == 20
    assert hyperparams.window_length == 1500
    assert hyperparams.dmfc_section_count == 6


def test_mint_lorenz_library_defaults_to_smoothed_spikes():
    model = MINTConfig(
        dataset="lorenz",
        sigma=1,
        delta=1,
        window_length=3,
        n_candidates=1,
    ).build(n_neurons=2, n_time=12)
    spikes = [
        torch.zeros(2, 12),
        torch.ones(2, 12),
    ]
    oracle_rates = [
        torch.full((2, 12), 999.0),
        torch.full((2, 12), 999.0),
    ]

    model.fit_library(spikes, oracle_rates, np.asarray([0, 0]))

    assert model.config.lorenz_library_source == "smoothed_spikes"
    assert len(model.Omega_plus) == 1
    assert float(model.Omega_plus[0].max()) < 10.0
