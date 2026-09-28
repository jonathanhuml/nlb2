from copy import deepcopy
import json
import random
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from nlb2 import Experiment, SelectionConfig, Study, StudyConfig, load_experiment_config
from nlb2.config import experiment_config_from_dict
from nlb2.experiment import experiment_config_to_dict
from nlb2.metrics import EvaluationResult
from nlb2.models.base import BaseDynamicsModel
from nlb2.tuning import SearchParameter, load_study_config
from nlb2.tuning.study import write_results
from nlb2.types import LossOutput, ModelOutput


def base_config(tmp_path):
    return experiment_config_from_dict({
        "dataset": {"name": "lorenz", "neurons": 3, "num_inits": 2, "num_trials": 3,
                    "num_steps": 8, "burn_steps": 5, "seed": 41},
        "model": {"name": "gpfa", "latent_dim": 2,
                  "optimization": {"name": "gradient", "lr": 0.01}},
        "trainer": {"epochs": 3, "batch_size": 2, "live_eval_interval": 1},
        "experiment": {"output_dir": str(tmp_path), "save_plots": False,
                       "training_seed": 7, "evaluation_seed": 100, "save_training_state": True},
        "selection": {"metric": "rate_mse", "mode": "min", "checkpoint": "last"},
    })


class ToyModel(BaseDynamicsModel):
    objective = "squared_error"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return ModelOutput(rates=torch.nn.functional.softplus(self.weight).expand_as(x))

    def loss(self, batch, output, epoch=0):
        # Exercise all RNG streams as well as optimizer state on resume.
        noise = torch.rand(()) + float(np.random.random()) + random.random()
        return LossOutput(total=(output.rates - batch["spikes"]).square().mean() + self.weight * noise)


@pytest.fixture
def toy(monkeypatch):
    def build(experiment):
        experiment.model = ToyModel()
        return experiment.model
    monkeypatch.setattr(Experiment, "build_model", build)


def test_config_snapshot_round_trip_preserves_batch_seeds_and_selection(tmp_path):
    original = base_config(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(experiment_config_to_dict(original)))
    restored = load_experiment_config(str(path))
    assert experiment_config_to_dict(restored) == experiment_config_to_dict(original)
    assert restored.batch_size == 2


def test_study_keeps_data_fixed_and_does_not_mutate_base(tmp_path):
    base = base_config(tmp_path)
    snapshot = experiment_config_to_dict(base)
    config = StudyConfig(base=base, objective=base.selection,
                         search_space={"model.latent_dim": {"distribution": "choice", "choices": [1, 2]}},
                         training_seeds=[3, 5])
    first = config.materialize({"model.latent_dim": 1}, 3)
    second = config.materialize({"model.latent_dim": 1}, 5)
    assert first.dataset == second.dataset == base.dataset
    assert first.training_seed == first.model.init_seed == 3
    assert second.training_seed == second.model.init_seed == 5
    assert experiment_config_to_dict(base) == snapshot


@pytest.mark.parametrize("path", ["dataset.seed", "dataset.split", "trainer.epochs", "model.init_seed", "model.typo", "model.optimization.lrr"])
def test_study_rejects_reserved_or_unknown_search_paths(tmp_path, path):
    with pytest.raises(ValueError):
        StudyConfig(base=base_config(tmp_path), objective={"metric": "rate_mse", "mode": "min"},
                    search_space={path: {"distribution": "choice", "choices": [1]}}, include_base=False)


@pytest.mark.parametrize("domain", [
    {"distribution": "loguniform", "low": 0, "high": 1},
    {"distribution": "randint", "low": 0.5, "high": 3},
    {"distribution": "uniform", "low": 3, "high": 1},
    {"distribution": "choice", "choices": []},
])
def test_invalid_domains(domain):
    with pytest.raises(ValueError):
        SearchParameter(**domain)


def test_study_rejects_test_split_and_undersized_candidate_budget(tmp_path):
    base = experiment_config_from_dict({"dataset": {"name": "mc_maze", "split": "test"},
                                        "model": {"name": "smoothing"}})
    kwargs = dict(objective={"metric": "co_bps", "mode": "max"},
                  search_space={"model.kern_sd_ms": {"distribution": "choice", "choices": [20, 50]}})
    with pytest.raises(ValueError, match="validation"):
        StudyConfig(base=base, **kwargs)
    base.dataset.split = "val"
    with pytest.raises(ValueError, match="num_samples"):
        StudyConfig(base=base, **kwargs, num_samples=1, points_to_evaluate=[{"model.kern_sd_ms": 20}])


def test_relative_base_and_coupled_parameter_sets(tmp_path):
    base = base_config(tmp_path)
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(experiment_config_to_dict(base)))
    payload = {"base": "base.yaml", "objective": {"metric": "rate_mse", "mode": "min"},
               "parameter_sets": [{"model.latent_dim": 2, "trainer.batch_size": 2},
                                  {"model.latent_dim": 1, "trainer.batch_size": 4}]}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload))
    config = load_study_config(path)
    config.validate_candidate(payload["parameter_sets"][1])
    with pytest.raises(ValueError, match="parameter set"):
        config.validate_candidate({"model.latent_dim": 2, "trainer.batch_size": 4})


def test_best_selection_matches_checkpoint_predictions_and_epoch(tmp_path, monkeypatch, toy):
    import nlb2.experiment as module
    config = base_config(tmp_path)
    config.selection = SelectionConfig(metric="rate_mse", mode="min", checkpoint="best")
    scores = iter([3.0, 1.0, 4.0])
    states = []

    def evaluate(model, **kwargs):
        states.append(model.weight.detach().item())
        return EvaluationResult(metrics={"rate_mse": next(scores)},
                                predictions={"weight": np.asarray([states[-1]])})

    monkeypatch.setattr(module, "evaluate_model", evaluate)
    reports = []
    result = Experiment(config).run(callback=reports.append)
    assert result.metrics["rate_mse"] == 1.0
    assert result.selected_epoch == 2
    assert result.completed_epochs == 3
    assert torch.load(result.model_path, weights_only=True)["weight"].item() == states[1]
    assert np.load(result.predictions_path)["pred_weight"][0] == states[1]
    assert reports[-1].final and reports[-1].epoch == 2


@pytest.mark.parametrize("strategy", ["gradient", "full_batch_gradient"])
def test_resume_preserves_optimizer_and_rng_trajectory(tmp_path, toy, strategy):
    from nlb2.models.base import OptimizationConfig
    reference_config = base_config(tmp_path / "reference")
    reference_config.model.optimization = OptimizationConfig(name=strategy, lr=0.01)
    reference = Experiment(reference_config).run()
    config = base_config(tmp_path / "interrupted")
    config.model.optimization = OptimizationConfig(name=strategy, lr=0.01)
    checkpoint = None

    def interrupt(report):
        nonlocal checkpoint
        checkpoint = report.checkpoint_path
        raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        Experiment(config).run(callback=interrupt)
    resumed = Experiment(config).run(resume_from=checkpoint)
    expected = torch.load(reference.model_path, weights_only=True)
    actual = torch.load(resumed.model_path, weights_only=True)
    assert torch.equal(actual["weight"], expected["weight"])
    assert [r.train.loss for r in resumed.history] == [r.train.loss for r in reference.history]
    assert resumed.metrics == reference.metrics
    changed = deepcopy(config)
    changed.training_seed += 1
    with pytest.raises(ValueError, match="same experiment"):
        Experiment(changed).run(resume_from=checkpoint)


def test_bgpfa_resume_restores_lazy_optimizer_and_annealing(tmp_path):
    from nlb2.models import BGPFAConfig

    config = base_config(tmp_path)
    config.model = BGPFAConfig(latent_dim=2, ell0=2.0, n_mc_train=1, n_mc_eval=1,
                              nlb_latent_infer_steps=1, nlb_latent_infer_n_mc=1,
                              optimization={"name": "mgplvm_full_batch_gradient", "n_mc": 1, "lr": 0.02})
    reference = Experiment(config).run()
    checkpoint = None

    def interrupt(report):
        nonlocal checkpoint
        checkpoint = report.checkpoint_path
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        Experiment(config).run(callback=interrupt)
    resumed = Experiment(config).run(resume_from=checkpoint)
    expected = torch.load(reference.model_path, weights_only=True)
    actual = torch.load(resumed.model_path, weights_only=True)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert [r.train.loss for r in resumed.history] == [r.train.loss for r in reference.history]


def test_evaluation_frequency_does_not_change_training(tmp_path, toy):
    first = base_config(tmp_path / "frequent")
    second = deepcopy(first)
    second.output_dir = str(tmp_path / "sparse")
    second.trainer.live_eval_interval = 3
    result_a = Experiment(first).run()
    result_b = Experiment(second).run()
    assert torch.equal(torch.load(result_a.model_path, weights_only=True)["weight"],
                       torch.load(result_b.model_path, weights_only=True)["weight"])


def test_callback_stop_retains_artifacts_and_actual_budget(tmp_path, toy):
    result = Experiment(base_config(tmp_path)).run(callback=lambda report: False)
    assert result.completed_epochs == 1
    assert result.model_path.exists()
    assert json.loads((result.run_dir / "status.json").read_text())["status"] == "stopped"


def test_no_candidate_can_win_from_partial_or_failed_scores(tmp_path):
    config = Study.from_config_path("configs/tuning/smoothing_lorenz.yaml").config
    trials = [{"trial_id": "partial", "status": "incomplete", "score": -100,
               "parameters": {"model.kern_sd_ms": 20}, "seeds": []},
              {"trial_id": "failed", "status": "failed", "score": -200,
               "parameters": {"model.kern_sd_ms": 30}, "seeds": []}]
    result = write_results(config, tmp_path, trials)
    assert result.best_config is None and result.best_score is None
    assert (tmp_path / "trials.csv").exists()


def test_public_api_import_does_not_load_ray():
    subprocess.run([sys.executable, "-c", "import sys, nlb2; assert 'ray' not in sys.modules"], check=True)
