"""Command-line interface for NLB2 experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys

from nlb2.config import ExperimentConfig, load_experiment_config
from nlb2.data import available_datasets, build_dataset_config
from nlb2.datasets.nlb import NLB_BIN_SIZES_MS, NLB_DATASETS, prepare_nlb_data
from nlb2.experiment import Experiment
from nlb2.mint_nlb import run_mint_nlb
from nlb2.models.mint import MINTConfig
from nlb2.models.base import BaseModelConfig, load_model_config
from nlb2.nlb_eval import (
    evaluate_nlb_submission,
    read_submission_co_bps,
    score_nlb2_predictions,
    score_run_dir,
    score_to_json,
)
from nlb2.preprocessing import PreprocessingConfig
from nlb2.training import TrainerConfig
from nlb2.utils.yaml import load_yaml


def main(argv: list[str] | None = None) -> int:
    """Run the NLB2 CLI."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0].startswith("-") and not _is_help_request(argv):
        argv = ["run", *argv]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nlb2",
        description="Run and inspect NLB2 latent dynamics experiments.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="run one or more experiments",
        description="Run NLB2 experiments sequentially, saving a separate folder for each run.",
    )
    run_parser.add_argument("-d", "--dataset", default="lorenz", help="dataset name")
    run_parser.add_argument("-m", "--model", default="cassm", help="model name")
    run_parser.add_argument(
        "-c", "--config", nargs="+", metavar="CONFIG",
        help="one or more experiment YAML recipes or saved config.json files",
    )
    run_parser.add_argument("--dataset-config", help="dataset YAML config")
    run_parser.add_argument("--model-config", help="model YAML config")
    run_parser.add_argument("--epochs", type=int, help="number of training epochs")
    run_parser.add_argument("--batch-size", type=int, help="training batch size")
    run_parser.add_argument("--device", help="PyTorch device")
    run_parser.add_argument("--training-seed", type=int, help="training RNG seed, independent of the dataset seed")
    run_parser.add_argument("--resume-from", help="resume a training_state.pt checkpoint into a new run folder")
    run_parser.add_argument(
        "--live-eval-interval",
        type=int,
        help="evaluate and write live metrics every N epochs; 0 disables live evaluation",
    )
    run_parser.add_argument("--output-dir", help="directory where run folders are written")
    run_parser.add_argument("--run-name", help="run folder name")
    run_parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="skip predictions.npz artifact",
    )
    run_parser.set_defaults(handler=run_command)

    tune_parser = subparsers.add_parser("tune", help="tune hyperparameters with Ray")
    tune_parser.add_argument("-c", "--config", help="study YAML or resolved study.json")
    tune_parser.add_argument("--resume-from", help="resume a study directory using its saved recipe")
    tune_parser.set_defaults(handler=tune_command)

    list_parser = subparsers.add_parser("list", help="list registered components")
    list_parser.add_argument("kind", choices=["datasets", "models"])
    list_parser.set_defaults(handler=list_command)

    prepare_nlb_parser = subparsers.add_parser(
        "prepare-nlb",
        help="prepare Neural Latents Benchmark H5 files",
        description=(
            "Build NLB2-ready NLB held-in/held-out H5 files from DANDI NWB files "
            "and the public nlb_tools test target H5."
        ),
    )
    prepare_nlb_parser.add_argument("--datasets", nargs="+", default=list(NLB_DATASETS))
    prepare_nlb_parser.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    prepare_nlb_parser.add_argument(
        "--bin-sizes-ms",
        nargs="+",
        type=int,
        default=list(NLB_BIN_SIZES_MS),
        choices=list(NLB_BIN_SIZES_MS),
    )
    prepare_nlb_parser.add_argument("--output-dir", default="data/real/nlb")
    prepare_nlb_parser.add_argument("--target-h5", help="path to nlb_tools/data/eval_data_test.h5")
    prepare_nlb_parser.add_argument("--nwb-root", default="data/real/nlb/dandi")
    prepare_nlb_parser.add_argument(
        "--search-root",
        action="append",
        dest="search_roots",
        help="additional root to search for DANDI-style NWB paths",
    )
    prepare_nlb_parser.add_argument(
        "--download",
        action="store_true",
        help="download the NLB target H5 and missing NWB files with dandi",
    )
    prepare_nlb_parser.add_argument("--overwrite", action="store_true")
    prepare_nlb_parser.add_argument("--include-psth", action="store_true", help="include PSTHs for val targets")
    prepare_nlb_parser.set_defaults(handler=prepare_nlb_command)

    score_nlb_parser = subparsers.add_parser(
        "score-nlb",
        help="score NLB held-out predictions",
        description=(
            "Score NLB2 predictions.npz artifacts with NLB co-bps, or run the "
            "full nlb_tools evaluator on an EvalAI-style H5 submission."
        ),
    )
    source = score_nlb_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", help="NLB2 run directory containing predictions.npz")
    source.add_argument("--predictions", help="NLB2 predictions.npz artifact")
    source.add_argument("--submission-h5", help="EvalAI-style NLB submission H5")
    score_nlb_parser.add_argument("--target-h5", help="NLB target H5 for --submission-h5")
    score_nlb_parser.add_argument("--dataset", choices=list(NLB_DATASETS), help="dataset for H5 co-bps-only scoring")
    score_nlb_parser.add_argument(
        "--bin-size-ms",
        type=int,
        default=5,
        choices=list(NLB_BIN_SIZES_MS),
        help="bin size for H5 co-bps-only scoring",
    )
    score_nlb_parser.add_argument("--output-json", help="optional path to write JSON metrics")
    score_nlb_parser.set_defaults(handler=score_nlb_command)

    return parser


def run_command(args: argparse.Namespace) -> int:
    paths = args.config if isinstance(args.config, list) else [args.config]
    if len(paths) > 1 and getattr(args, "resume_from", None):
        raise ValueError("Use --resume-from with a single experiment configuration.")
    # Validate every recipe before starting any training.
    configs = [
        build_experiment_config(argparse.Namespace(**{**vars(args), "config": path}))
        for path in paths
    ]
    for config in configs:
        _run_experiment(config, resume_from=getattr(args, "resume_from", None))
    return 0


def _run_experiment(config: ExperimentConfig, *, resume_from: str | None = None) -> None:
    if isinstance(config.model, MINTConfig) and config.model.train_source in {"mat", "nwb"}:
        result = run_mint_nlb(config)
        print(f"Wrote NLB2 run: {result.run_dir}")
        print("Metrics:")
        for key, value in sorted(result.metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            print(f"  {key}: {display}")
        print(f"  predictions: {result.predictions_path}")
        print(f"  submission_h5: {result.submission_path}")
        return

    result = Experiment(config).run(resume_from=resume_from)
    print(f"Wrote NLB2 run: {result.run_dir}")
    if result.metrics:
        print("Metrics:")
        for key, value in sorted(result.metrics.items()):
            display = "nan" if not math.isfinite(value) else f"{value:.6g}"
            print(f"  {key}: {display}")
    else:
        print("Metrics: none available")


def tune_command(args: argparse.Namespace) -> int:
    from nlb2.tuning import Study

    if args.config and args.resume_from:
        raise ValueError("Use either --config or --resume-from; resume uses the saved study recipe.")
    if not args.config and not args.resume_from:
        raise ValueError("Provide a study with --config or --resume-from.")
    path = args.config or str(Path(args.resume_from) / "study.json")
    result = Study.from_config_path(path).run(resume_from=args.resume_from)
    print(f"Wrote NLB2 study: {result.study_dir}")
    if result.best_config is None:
        print("No candidate completed all seed runs; inspect trials.csv and Ray logs.")
        return 1
    print(f"Selected mean validation score: {result.best_score:.6g}")
    print(f"Selected config: {result.best_config_path}")
    return 0


def list_command(args: argparse.Namespace) -> int:
    if args.kind == "datasets":
        for name in available_datasets():
            print(name)
        return 0

    from nlb2 import models as _models  # noqa: F401

    for name in sorted(BaseModelConfig.registry):
        print(name)
    return 0


def prepare_nlb_command(args: argparse.Namespace) -> int:
    prepared = prepare_nlb_data(
        datasets=args.datasets,
        splits=args.splits,
        bin_sizes_ms=args.bin_sizes_ms,
        output_dir=Path(args.output_dir),
        target_h5=Path(args.target_h5) if args.target_h5 else None,
        nwb_root=Path(args.nwb_root),
        search_roots=[Path(item) for item in args.search_roots] if args.search_roots else None,
        download=bool(args.download),
        overwrite=bool(args.overwrite),
        include_psth=bool(args.include_psth),
    )
    for item in prepared:
        print(
            f"{item.dataset} {item.split} {item.bin_size_ms}ms: "
            f"{item.path} heldin={item.heldin_shape} heldout={item.heldout_shape}"
        )
    return 0


def score_nlb_command(args: argparse.Namespace) -> int:
    if args.run_dir:
        score = score_run_dir(Path(args.run_dir))
    elif args.predictions:
        score = score_nlb2_predictions(Path(args.predictions))
    else:
        if not args.target_h5:
            raise ValueError("--target-h5 is required with --submission-h5.")
        if args.dataset:
            score = read_submission_co_bps(
                Path(args.target_h5),
                Path(args.submission_h5),
                dataset=args.dataset,
                bin_size_ms=args.bin_size_ms,
            )
        else:
            score = evaluate_nlb_submission(Path(args.target_h5), Path(args.submission_h5))

    text = score_to_json(score)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n")
    return 0


def build_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    config_path = args.config
    if isinstance(config_path, list):
        if len(config_path) != 1:
            raise ValueError("build_experiment_config requires a single configuration.")
        config_path = config_path[0]
    if config_path:
        config = load_experiment_config(config_path)
    else:
        config = ExperimentConfig(
            dataset=_load_dataset_config(args.dataset, args.dataset_config),
            model=_load_model_config(args.model, args.model_config),
            trainer=TrainerConfig(),
            preprocessing=_load_default_preprocessing(args.dataset, args.model),
        )

    trainer = config.trainer
    if args.epochs is not None:
        trainer = replace(trainer, epochs=args.epochs)
    if args.device is not None:
        trainer = replace(trainer, device=args.device)
    if args.live_eval_interval is not None:
        trainer = replace(trainer, live_eval_interval=args.live_eval_interval)
    training_seed = getattr(args, "training_seed", None)

    return replace(
        config,
        trainer=trainer,
        batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
        output_dir=args.output_dir if args.output_dir is not None else config.output_dir,
        run_name=args.run_name if args.run_name is not None else config.run_name,
        save_predictions=False if args.no_save_predictions else config.save_predictions,
        training_seed=training_seed if training_seed is not None else config.training_seed,
    )


def _load_dataset_config(name: str, path: str | None) -> object:
    if path is None:
        return build_dataset_config(name)
    data = load_yaml(path)
    dataset_name = data.get("name", name)
    return build_dataset_config(dataset_name, data)


def _load_model_config(name: str, path: str | None) -> BaseModelConfig:
    from nlb2 import models as _models  # noqa: F401

    if path is not None:
        return load_model_config(path)
    if name not in BaseModelConfig.registry:
        known = ", ".join(sorted(BaseModelConfig.registry)) or "<none>"
        raise KeyError(f"Unknown model '{name}'. Registered models: {known}.")
    return BaseModelConfig.registry[name]()


def _load_default_preprocessing(dataset: str, model: str) -> PreprocessingConfig:
    path = _default_experiment_config_path(dataset, model)
    if not path.exists():
        return PreprocessingConfig()
    data = load_yaml(path)
    return PreprocessingConfig.model_validate(data.get("preprocessing", {}))


def _default_experiment_config_path(dataset: str, model: str) -> Path:
    root = Path("configs") / "experiment"
    candidates = [
        root / "synthetic" / dataset / model / f"{model}_{dataset}.yaml",
        root / "real" / dataset / model / f"{model}_{dataset}.yaml",
        root / f"{model}_{dataset}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _is_help_request(argv: list[str]) -> bool:
    return len(argv) == 1 and argv[0] in {"-h", "--help"}


if __name__ == "__main__":
    raise SystemExit(main())
