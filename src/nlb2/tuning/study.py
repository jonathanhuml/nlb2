"""Public study orchestration and portable result summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from nlb2.config import ExperimentConfig, experiment_config_from_dict, load_experiment_config
from nlb2.experiment import _create_unique_dir, _slugify, _write_json
from nlb2.tuning.config import StudyConfig, canonical_config, load_study_config


@dataclass
class StudyResult:
    """Completed and failed candidates, plus the selection across complete candidates.

    best_config uses the first declared training seed. With multiple seeds,
    best_checkpoints contains one corresponding model per seed; there is no
    single checkpoint whose score is the candidate mean.
    """

    study_dir: Path
    trials: list[dict[str, Any]]
    best_config: ExperimentConfig | None
    best_config_path: Path | None
    best_score: float | None
    best_checkpoints: dict[int, Path]

    @classmethod
    def from_path(cls, path: str | Path) -> "StudyResult":
        path = Path(path).resolve()
        selection = json.loads((path / "selection.json").read_text())
        config_path = path / "best_config.yaml" if selection["best_trial"] is not None else None
        return cls(
            study_dir=path, trials=json.loads((path / "trials.json").read_text()),
            best_config=None if config_path is None else load_experiment_config(str(config_path)),
            best_config_path=config_path, best_score=selection["score"],
            best_checkpoints={int(seed): path / value for seed, value in selection["checkpoints"].items()},
        )


class Study:
    """Run a fixed-budget Ray Tune study over NLB2 experiments."""

    def __init__(self, config: StudyConfig):
        # Revalidate after any programmatic mutations, and isolate caller state.
        self.config = StudyConfig.model_validate(config.model_dump())
        self.result: StudyResult | None = None

    @classmethod
    def from_config_path(cls, path: str | Path) -> "Study":
        return cls(load_study_config(path))

    def run(self, *, resume_from: str | Path | None = None) -> StudyResult:
        from nlb2.tuning.ray_backend import require_ray, run_study

        require_ray()
        config = _resolve_paths(self.config)
        identity = provenance(config)
        if resume_from is None:
            root = Path(config.output_dir).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            name = config.name or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            directory = _create_unique_dir(root / _slugify(name))
            _write_json(directory / "study.json", config.model_dump(mode="json"))
            _write_json(directory / "provenance.json", identity)
        else:
            directory = Path(resume_from).resolve()
            saved = StudyConfig.model_validate(json.loads((directory / "study.json").read_text()))
            if saved.model_dump() != config.model_dump():
                raise ValueError("Resume requires the saved study configuration; use its study.json recipe.")
            previous = json.loads((directory / "provenance.json").read_text())
            if previous != identity:
                raise ValueError("Study data, source code, or software versions changed since the saved run.")
        self.result = run_study(config, directory, resume=resume_from is not None)
        return self.result


def _resolve_paths(config: StudyConfig) -> StudyConfig:
    payload = config.model_dump()
    base = experiment_config_from_dict(payload["base"])
    dataset = base.dataset
    path = getattr(dataset, "resolved_data_path", getattr(dataset, "data_path", None))
    if path is not None:
        payload["base"]["dataset"]["data_path"] = str(Path(path).expanduser().resolve())

    def resolve(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.endswith("_path") and value:
                    node[key] = str(Path(value).expanduser().resolve())
                else:
                    resolve(value)
        elif isinstance(node, list):
            for value in node:
                resolve(value)

    resolve(payload["base"]["model"])
    payload["output_dir"] = str(Path(config.output_dir).expanduser().resolve())
    return StudyConfig.model_validate(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(config: StudyConfig) -> dict[str, Any]:
    """Fingerprint data and source content once on the driver, before trials start."""
    files = {}

    def collect(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key.endswith("_path") and isinstance(value, str) and value:
                    path = Path(value)
                    if not path.is_file():
                        raise FileNotFoundError(f"Study input does not exist: {path}")
                    files[str(path)] = _sha256(path)
                else:
                    collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(config.base)
    versions = {"python": platform.python_version()}
    for package in ("nlb2", "torch", "numpy", "pydantic", "ray", "optuna"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    source_root = Path(__file__).resolve().parents[3]
    source_hash = hashlib.sha256()
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml"}:
            source_hash.update(str(path.relative_to(source_root)).encode())
            source_hash.update(path.read_bytes())
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    dataset_hash = hashlib.sha256(json.dumps(config.base["dataset"], sort_keys=True).encode()).hexdigest()
    return {"files": files, "dataset_config_sha256": dataset_hash,
            "source_sha256": source_hash.hexdigest(), "git_commit": commit, "versions": versions}


def write_results(config: StudyConfig, directory: Path, trials: list[dict]) -> StudyResult:
    import csv
    import yaml

    _write_json(directory / "trials.json", trials)
    with (directory / "trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial_id", "status", "score", "score_std", "elapsed_seconds", "parameters", "error"])
        writer.writeheader()
        for trial in trials:
            row = {key: trial.get(key) for key in writer.fieldnames}
            row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            writer.writerow(row)
    complete = [trial for trial in trials if trial["status"] == "complete"]
    best = None
    checkpoints = {}
    if complete:
        # Stable tie breaking follows Ray's returned trial order.
        best = (min if config.objective.mode == "min" else max)(complete, key=lambda row: row["score"])
        candidate = config.materialize(best["parameters"], config.training_seeds[0])
        candidate.output_dir = "runs"
        candidate.run_name = f"{directory.name}_selected"
        candidate.trainer.device = "cuda" if config.resources.gpu else "cpu"
        (directory / "best_config.yaml").write_text(yaml.safe_dump(canonical_config(candidate), sort_keys=False))
        checkpoints = {str(row["seed"]): row["model_path"] for row in best["seeds"]}
    _write_json(directory / "selection.json", {
        "best_trial": None if best is None else best["trial_id"],
        "score": None if best is None else best["score"],
        "score_std": None if best is None else best["score_std"],
        "objective": config.objective.model_dump(), "aggregation": "mean",
        "training_seeds": config.training_seeds, "checkpoints": checkpoints,
    })
    return StudyResult.from_path(directory)
