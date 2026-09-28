"""Small real Ray studies; skipped when the tuning extra is not installed."""

import json

import numpy as np
import pytest

ray = pytest.importorskip("ray")
pytest.importorskip("optuna")

from nlb2 import Experiment, Study, StudyResult
from nlb2 import StudyConfig
from nlb2.tuning.ray_backend import _search_algorithm


@pytest.fixture(scope="module", autouse=True)
def local_ray():
    ray.init(num_cpus=2, include_dashboard=False)
    yield
    ray.shutdown()


@pytest.mark.parametrize("algorithm", ["random", "tpe"])
def test_real_study_export_repeats_and_restore(tmp_path, algorithm):
    study = Study.from_config_path("configs/tuning/smoothing_lorenz.yaml")
    study.config.algorithm = algorithm
    study.config.output_dir = str(tmp_path)
    study.config.num_samples = 2
    study.config.training_seeds = [2, 3]
    study.config.base["trainer"].update(epochs=2, live_eval_interval=1)
    study.config.objective.checkpoint = "best"
    result = study.run()
    assert len(result.trials) == 2
    assert all(trial["status"] == "complete" for trial in result.trials)
    assert result.best_score == min(trial["score"] for trial in result.trials)
    assert set(result.best_checkpoints) == {2, 3}
    assert all(path.exists() for path in result.best_checkpoints.values())
    for trial in result.trials:
        assert trial["score"] == np.mean([row["score"] for row in trial["seeds"]])
        assert all(row["completed_epochs"] == 2 for row in trial["seeds"])
    replay = Experiment.from_config_path(result.best_config_path).config
    replay.output_dir = str(tmp_path / "replay")
    replay_result = Experiment(replay).run()
    assert replay_result.metrics["rate_mse"] == result.best_score
    restored = Study.from_config_path(result.study_dir / "study.json").run(resume_from=result.study_dir)
    assert restored.best_score == result.best_score
    assert len(restored.trials) == 2
    assert StudyResult.from_path(result.study_dir).best_score == result.best_score
    assert json.loads((result.study_dir / "provenance.json").read_text())["versions"]["ray"] == ray.__version__


def test_random_search_respects_concurrency_and_tpe_ignores_partial_results():
    config = Study.from_config_path("configs/tuning/smoothing_lorenz.yaml").config
    config.max_concurrent_trials = 1
    search = _search_algorithm(config)
    assert search.max_concurrent == 1
    config.algorithm = "tpe"
    search = _search_algorithm(config)
    # A partial report has no aggregate objective; it must not enter Optuna.
    search.on_trial_result("not-yet-complete", {"score": None, "candidate_complete": False})


def test_failed_candidate_is_recorded_and_cannot_win(tmp_path):
    config = StudyConfig(
        base={
            "dataset": {"name": "lorenz", "neurons": 3, "num_inits": 2, "num_trials": 3,
                        "num_steps": 8, "burn_steps": 5, "seed": 41},
            "model": {"name": "gpfa", "latent_dim": 2, "optimization": {"name": "gradient", "lr": 0.01}},
            "trainer": {"epochs": 1, "batch_size": 2},
            "experiment": {"save_plots": False},
        },
        objective={"metric": "rate_mse", "mode": "min"},
        search_space={"model.optimization.lr": {"distribution": "choice", "choices": [0.01, -1.0]}},
        points_to_evaluate=[{"model.optimization.lr": -1.0}],
        num_samples=2, algorithm="tpe", output_dir=str(tmp_path),
    )
    result = Study(config).run()
    assert [trial["status"] for trial in result.trials] == ["complete", "failed"]
    assert result.best_score == result.trials[0]["score"]
    assert result.trials[1]["error"]
    assert result.trials[1]["score"] is None
