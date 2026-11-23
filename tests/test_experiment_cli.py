import json
from pathlib import Path

import pytest
import yaml

from nlb2.cli import CONFIG_ROOT, build_experiment_config, build_parser, main
from nlb2.config import ExperimentConfig, load_experiment_config
from nlb2.data import build_dataset_config
from nlb2.datasets import LorenzDatasetConfig
from nlb2.experiment import Experiment
from nlb2.models import GPFAConfig
from nlb2.models.base import load_model_config
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig
from nlb2.utils.yaml import load_yaml


def test_experiment_writes_run_artifacts(tmp_path: Path):
    config = ExperimentConfig(
        dataset=LorenzDatasetConfig(
            neurons=5,
            num_inits=2,
            num_trials=3,
            num_steps=8,
            burn_steps=5,
            seed=0,
        ),
        model=GPFAConfig(
            latent_dim=2,
            learn_kernel_params=False,
            fa_max_iters=3,
            kernel_param_max_iters=1,
        ),
        trainer=TrainerConfig(epochs=1, device="cpu"),
        preprocessing=PreprocessingConfig(),
        batch_size=2,
        output_dir=str(tmp_path),
        run_name="gpfa-smoke",
    )

    result = Experiment(config).run()

    assert result.run_dir == tmp_path / "gpfa-smoke"
    assert result.config_path.exists()
    assert result.history_path.exists()
    assert result.metrics_path.exists()
    assert result.model_path.exists()
    assert result.predictions_path is not None
    assert result.predictions_path.exists()
    assert result.report_path.exists()
    assert (result.run_dir / "plots" / "train_valid_objective_curves.png").exists()
    assert (result.run_dir / "plots" / "valid_objective_curves.png").exists()
    assert result.plot_paths["train_valid_objective"].exists()
    assert result.plot_paths["valid_objective"].exists()
    assert "co_bps" in result.metrics
    assert "rate_mse" in result.metrics


def test_cli_run_from_config(tmp_path: Path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        """
dataset:
  name: lorenz
  neurons: 5
  num_inits: 2
  num_trials: 3
  num_steps: 8
  burn_steps: 5
  train_fraction: 0.67
  seed: 1
  latent_dt: 0.015
  spike_bin_size: 1.0
  base_rate: 1.0

model:
  name: gpfa
  objective: negative_log_marginal_likelihood
  latent_dim: 2
  bin_width: 20.0
  start_tau: 100.0
  start_eps: 1.0e-3
  min_var_frac: 0.01
  learn_kernel_params: false
  fa_max_iters: 3
  fa_tol: 1.0e-8
  kernel_param_max_iters: 1
  kernel_param_lr: 1.0
  jitter: 1.0e-5
  optimization:
    name: gradient
    optimizer: Adam
    lr: 1.0e-2
    weight_decay: 0.0
    gradient_clip: 100.0

preprocessing:
  observations: null

trainer:
  epochs: 1
  batch_size: 2
  device: cpu
""".lstrip()
    )

    exit_code = main(
        [
            "run",
            "-c",
            str(config_path),
            "--output-dir",
            str(tmp_path),
            "--run-name",
            "cli-run",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli-run" / "metrics.json").exists()


def test_cli_lists_models(capsys):
    assert main(["list", "models"]) == 0
    output = capsys.readouterr().out
    assert "bgpfa" in output
    assert "gpfa" in output
    assert "kalman" in output
    assert "langevin_flow" in output
    assert "lfads" in output
    assert "ndt" in output
    assert "psth" in output
    assert "smoothing" in output


def test_cli_lists_datasets(capsys):
    assert main(["list", "datasets"]) == 0
    output = capsys.readouterr().out
    assert "area2_bump" in output
    assert "chaotic_rnn" in output
    assert "dmfc_rsg" in output
    assert "lorenz" in output
    assert "mc_maze" in output
    assert "mc_rtt" in output


def test_experiment_configs_are_nested_by_dataset_and_model():
    root = Path("configs/experiment")
    paths = sorted(root.glob("*/*/*/*.yaml"))
    assert paths

    for path in paths:
        relative = path.relative_to(root)
        assert len(relative.parts) == 4
        load_experiment_config(str(path))

    shallow_yaml = sorted(root.glob("*/*/*.yaml"))
    assert shallow_yaml == []


def _write_batch_recipe(path: Path, latent_dim: int) -> Path:
    recipe = {
        "dataset": {
            "name": "lorenz", "neurons": 5, "num_inits": 2,
            "num_trials": 3, "num_steps": 8, "burn_steps": 5, "seed": 0,
        },
        "model": {
            "name": "gpfa", "latent_dim": latent_dim, "learn_kernel_params": False,
            "optimization": {"name": "gradient", "optimizer": "Adam", "lr": 2e-4, "weight_decay": 2e-5},
        },
        "trainer": {"epochs": 0, "device": "cpu"},
        "experiment": {"run_name": path.stem, "save_plots": False},
    }
    path.write_text(json.dumps(recipe) if path.suffix == ".json" else yaml.safe_dump(recipe))
    return path


def test_cli_runs_multiple_configs_with_shared_overrides(tmp_path: Path, capsys):
    paths = [
        _write_batch_recipe(tmp_path / "first.yaml", 1),
        _write_batch_recipe(tmp_path / "second.json", 2),
        _write_batch_recipe(tmp_path / "third.yaml", 3),
    ]
    output_dir = tmp_path / "runs"

    assert main([
        "run", "-c", *map(str, paths), "--epochs", "1", "--batch-size", "2",
        "--device", "cpu", "--output-dir", str(output_dir),
    ]) == 0

    output = capsys.readouterr().out
    assert output.count("Wrote NLB2 run:") == 3
    for latent_dim, path in enumerate(paths, 1):
        run_dir = output_dir / path.stem
        saved = load_experiment_config(str(run_dir / "config.json"))
        assert saved.model.latent_dim == latent_dim
        assert saved.model.optimization.kwargs()["weight_decay"] == 2e-5
        assert saved.trainer.epochs == 1
        assert saved.trainer.device == "cpu"
        assert saved.batch_size == 2
        assert json.loads((run_dir / "status.json").read_text())["status"] == "complete"
        assert (run_dir / "predictions.npz").is_file()


def test_cli_validates_all_configs_before_training(tmp_path: Path):
    valid_path = _write_batch_recipe(tmp_path / "valid.yaml", 2)
    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("dataset: {name: lorenz}\nmodel: {name: unknown_model}\n")
    output_dir = tmp_path / "runs"

    with pytest.raises(KeyError):
        main(["run", "-c", str(valid_path), str(invalid_path), "--output-dir", str(output_dir)])

    assert not output_dir.exists()


def test_cli_defaults_run_repository_recipe_from_any_directory(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = build_parser().parse_args(["run", "--epochs", "3", "--batch-size", "8"])
    assert (args.dataset, args.model) == ("lorenz", "lfads")

    config = build_experiment_config(args)
    recipe = load_experiment_config(
        str(CONFIG_ROOT / "experiment" / "synthetic" / "lorenz" / "lfads" / "lfads_lorenz.yaml")
    )
    assert config.dataset == recipe.dataset
    assert config.model == recipe.model
    assert config.preprocessing == recipe.preprocessing
    assert config.trainer.epochs == 3
    assert config.batch_size == 8


def test_cli_defaults_without_recipe_use_model_and_dataset_configs(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = build_experiment_config(build_parser().parse_args(["run", "-d", "lorenz", "-m", "stndt"]))

    assert config.model == load_model_config(str(CONFIG_ROOT / "model" / "stndt.yaml"))
    assert config.dataset == build_dataset_config(
        "lorenz", load_yaml(str(CONFIG_ROOT / "dataset" / "lorenz.yaml"))
    )


def test_cli_requires_one_config_when_resuming():
    with pytest.raises(ValueError, match="single experiment configuration"):
        main(["run", "-c", "first.yaml", "second.yaml", "--resume-from", "training_state.pt"])
