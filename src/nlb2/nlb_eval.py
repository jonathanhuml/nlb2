"""Neural Latents Benchmark scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any
import warnings

import h5py
import numpy as np
from scipy.special import gammaln
import torch

from nlb2.types import move_batch_to_device, observations_from_batch


@dataclass
class NLBScore:
    """NLB co-smoothing score for held-out spike-count predictions."""

    co_bps: float
    spike_count: float
    prediction_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "co_bps": self.co_bps,
            "spike_count": self.spike_count,
            "prediction_shape": list(self.prediction_shape),
            "target_shape": list(self.target_shape),
        }
        data.update(self.metrics)
        return data


@dataclass
class NLBFullEvaluation:
    """Artifacts and metrics from an EvalAI-style NLB evaluation."""

    metrics: dict[str, float]
    raw_result: list[dict[str, dict[str, float]]]
    submission_path: Path
    target_path: Path


def nlb_negative_log_likelihood(
    rates: np.ndarray,
    spikes: np.ndarray,
    *,
    zero_floor: float = 1e-9,
) -> float:
    """NLB Poisson negative log-likelihood for spike-count rates."""

    rates = np.asarray(rates, dtype=np.float64).copy()
    spikes = np.asarray(spikes, dtype=np.float64)
    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")

    if np.any(np.isnan(spikes)):
        mask = ~np.isnan(spikes)
        rates = rates[mask]
        spikes = spikes[mask]

    if np.any(np.isnan(rates)):
        raise ValueError("NaN rate predictions found")
    if np.any(rates < 0):
        raise ValueError("negative rate predictions found")
    rates[rates == 0] = zero_floor
    return float(np.sum(rates - spikes * np.log(rates) + gammaln(spikes + 1.0)))


def nlb_bits_per_spike(rates: np.ndarray, spikes: np.ndarray) -> float:
    """NLB/EvalAI co-smoothing bits per spike for count-valued predictions."""

    rates = np.asarray(rates, dtype=np.float64)
    spikes = np.asarray(spikes, dtype=np.float64)
    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")

    nll_model = nlb_negative_log_likelihood(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = nlb_negative_log_likelihood(null_rates, spikes)
    spike_count = float(np.nansum(spikes))
    if spike_count <= 0.0:
        raise ValueError("cannot compute bits/spike with zero held-out spikes")
    return float((nll_null - nll_model) / spike_count / np.log(2.0))


def score_count_predictions(rates: np.ndarray, spikes: np.ndarray) -> NLBScore:
    """Score held-out count predictions against held-out spikes."""

    rates = np.asarray(rates)
    spikes = np.asarray(spikes)
    score = nlb_bits_per_spike(rates, spikes)
    return NLBScore(
        co_bps=score,
        spike_count=float(np.nansum(spikes)),
        prediction_shape=tuple(rates.shape),
        target_shape=tuple(spikes.shape),
        metrics={"co-bps": score},
    )


def score_nlb2_predictions(path: Path | str) -> NLBScore:
    """Score a NLB2 ``predictions.npz`` artifact with the NLB co-bps metric."""

    path = Path(path)
    with np.load(path) as data:
        rate_key = "pred_count_rates" if "pred_count_rates" in data else "pred_rates"
        if rate_key not in data or "target_spikes" not in data:
            keys = ", ".join(data.files)
            raise KeyError(
                f"{path} must contain pred_rates and target_spikes arrays. Found: {keys}"
            )
        return score_count_predictions(data[rate_key], data["target_spikes"])


def score_run_dir(path: Path | str) -> NLBScore:
    """Score ``predictions.npz`` inside a NLB2 run directory."""

    return score_nlb2_predictions(Path(path) / "predictions.npz")


def evaluate_nlb_submission(
    target_h5: Path | str,
    submission_h5: Path | str,
) -> list[dict[str, dict[str, float]]]:
    """Run the full nlb_tools EvalAI-style evaluator on an H5 submission."""

    try:
        from nlb_tools.evaluation import evaluate
    except ImportError as exc:
        raise RuntimeError("nlb_tools is required for full NLB H5 evaluation.") from exc
    return evaluate(str(target_h5), str(submission_h5))


def evaluate_model_nlb_submission(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    valid_loader: Any,
    dataset_config: Any,
    device: torch.device | str,
    output_dir: Path | str,
    prediction_floor: float = 1e-9,
    progress: bool = False,
) -> NLBFullEvaluation | None:
    """Write and evaluate an EvalAI-style NLB submission for a full-rate model.

    The lightweight NLB2 adapter reports held-out co-smoothing directly. This
    helper additionally exports train/eval held-in, held-out, and forward rate
    tensors so ``nlb_tools`` can compute behavioral, PSTH, and forward metrics
    for NLB H5 files that include those targets.
    """

    if not _looks_like_nlb_config(dataset_config):
        return None
    if getattr(dataset_config, "max_trials", None) is not None:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    group_name = _nlb_group_name(dataset_config)
    target_path = output_dir / "nlb_eval_target.h5"
    submission_path = output_dir / "nlb_submission.h5"
    _write_grouped_target_h5(dataset_config, target_path, group_name)

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    dt = float(getattr(dataset_config, "bin_size", None) or getattr(dataset_config, "bin_size_ms") / 1000.0)

    train_rates = _collect_full_rate_parts(
        model=model,
        loader=train_loader,
        device=torch_device,
        dt=dt,
        prediction_floor=prediction_floor,
        progress_label="train" if progress else None,
    )
    eval_rates = _collect_full_rate_parts(
        model=model,
        loader=valid_loader,
        device=torch_device,
        dt=dt,
        prediction_floor=prediction_floor,
        progress_label="eval" if progress else None,
    )
    if train_rates is None or eval_rates is None:
        return None

    with h5py.File(submission_path, "w") as handle:
        group = handle.create_group(group_name)
        _write_rate_parts(group, "train", train_rates)
        _write_rate_parts(group, "eval", eval_rates)

    with h5py.File(target_path, "r") as target:
        target_group = target[group_name]
        has_behavior = all(key in target_group for key in ("train_behavior", "eval_behavior"))
        if not has_behavior:
            co_bps = nlb_bits_per_spike(
                eval_rates["rates_heldout"], target_group["eval_spikes_heldout"][()]
            )
    if has_behavior:
        raw_result = evaluate_nlb_submission(target_path, submission_path)
    else:
        warnings.warn(
            "NLB data lacks train/eval behavior metadata: exported neural predictions "
            "and scored co-bps only; full NLB metrics are unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw_result = [{f"{group_name}_split": {"co-bps": co_bps}}]
    result_path = output_dir / "nlb_full_metrics.json"
    result_path.write_text(score_to_json(raw_result) + "\n")
    return NLBFullEvaluation(
        metrics=_flatten_nlb_result(raw_result),
        raw_result=raw_result,
        submission_path=submission_path,
        target_path=target_path,
    )


def read_submission_co_bps(
    target_h5: Path | str,
    submission_h5: Path | str,
    dataset: str,
    bin_size_ms: int = 5,
) -> NLBScore:
    """Score one EvalAI-style H5 group when only co-bps is required."""

    group_name = dataset if bin_size_ms == 5 else f"{dataset}_{bin_size_ms}"
    with h5py.File(target_h5, "r") as target, h5py.File(submission_h5, "r") as pred:
        spikes = target[group_name]["eval_spikes_heldout"][()]
        rates = pred[group_name]["eval_rates_heldout"][()]
    return score_count_predictions(rates, spikes)


def _looks_like_nlb_config(config: Any) -> bool:
    return hasattr(config, "resolved_data_path") and hasattr(config, "resolved_group")


def _nlb_group_name(config: Any) -> str:
    group = getattr(config, "resolved_group")
    return str(group() if callable(group) else group)


def _write_grouped_target_h5(config: Any, output: Path, group_name: str) -> None:
    source_path = Path(getattr(config, "resolved_data_path"))
    with h5py.File(source_path, "r") as source, h5py.File(output, "w") as target:
        source_group = source[group_name] if group_name in source else source
        target_group = target.create_group(group_name)
        for key in source_group.keys():
            source_group.copy(key, target_group, name=key)


def prepare_nlb_selection_target(config: Any, output_dir: Path | str) -> Path:
    """Build checkpoint-selection targets from the configured validation data."""

    if getattr(config, "split", None) != "val":
        raise ValueError("NLB checkpoint selection requires a dataset with split='val'.")
    output = Path(output_dir) / "nlb_validation_target.h5"
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_grouped_target_h5(config, output, _nlb_group_name(config))
    return output


def _collect_full_rate_parts(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    dt: float,
    prediction_floor: float,
    progress_label: str | None = None,
) -> dict[str, np.ndarray] | None:
    heldin_rates: list[np.ndarray] = []
    heldout_rates: list[np.ndarray] = []
    heldin_forward_rates: list[np.ndarray] = []
    heldout_forward_rates: list[np.ndarray] = []
    any_forward = False

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if not isinstance(batch, dict):
                return None
            if progress_label is not None:
                if total_batches is None:
                    print(f"{progress_label} batch {batch_index}", flush=True)
                else:
                    print(
                        f"{progress_label} batch {batch_index}/{total_batches}",
                        flush=True,
                    )
            batch = move_batch_to_device(batch, device)
            output = model(observations_from_batch(batch))
            full_counts = output.count_rates(dt, full=True)
            if full_counts is None:
                return None
            full_counts = full_counts.clamp_min(prediction_floor)

            heldin = batch.get("heldin_spikes", batch.get("spikes"))
            heldout = batch.get("heldout_spikes", batch.get("raw_spikes"))
            if heldin is None or heldout is None:
                return None
            main_steps = int(heldout.shape[1])
            heldin_neurons = int(heldin.shape[-1])
            heldout_neurons = int(heldout.shape[-1])
            heldout_start = int(getattr(model, "output_neuron_start", heldin_neurons) or heldin_neurons)
            heldout_stop = heldout_start + heldout_neurons
            if full_counts.shape[1] < main_steps or full_counts.shape[-1] < heldout_stop:
                return None

            heldin_rates.append(
                full_counts[:, :main_steps, :heldin_neurons].detach().cpu().numpy()
            )
            heldout_rates.append(
                full_counts[:, :main_steps, heldout_start:heldout_stop].detach().cpu().numpy()
            )

            heldin_forward = batch.get("heldin_forward_spikes")
            heldout_forward = batch.get("heldout_forward_spikes")
            if heldin_forward is not None and heldout_forward is not None:
                forward_steps = int(heldout_forward.shape[1])
                if full_counts.shape[1] >= main_steps + forward_steps:
                    any_forward = True
                    heldin_forward_rates.append(
                        full_counts[
                            :,
                            main_steps : main_steps + forward_steps,
                            :heldin_neurons,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    heldout_forward_rates.append(
                        full_counts[
                            :,
                            main_steps : main_steps + forward_steps,
                            heldout_start:heldout_stop,
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

    if not heldin_rates or not heldout_rates:
        return None

    parts = {
        "rates_heldin": np.concatenate(heldin_rates, axis=0),
        "rates_heldout": np.concatenate(heldout_rates, axis=0),
    }
    if any_forward and heldin_forward_rates and heldout_forward_rates:
        parts["rates_heldin_forward"] = np.concatenate(heldin_forward_rates, axis=0)
        parts["rates_heldout_forward"] = np.concatenate(heldout_forward_rates, axis=0)
    return parts


def _write_rate_parts(group: h5py.Group, prefix: str, parts: dict[str, np.ndarray]) -> None:
    for suffix, array in parts.items():
        group.create_dataset(f"{prefix}_{suffix}", data=np.asarray(array, dtype=np.float32))


def _flatten_nlb_result(result: list[dict[str, dict[str, float]]]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for split_result in result:
        for _, metrics in split_result.items():
            for key, value in metrics.items():
                normalized = key.lower().replace("-", "_").replace(" ", "_")
                if np.isfinite(value):
                    flattened[f"nlb_{normalized}"] = float(value)
                else:
                    flattened[f"nlb_{normalized}"] = float("nan")
    return flattened


def score_to_json(score: NLBScore | list[dict[str, dict[str, float]]]) -> str:
    """Serialize NLB scorer output for CLI use."""

    if isinstance(score, NLBScore):
        payload: Any = score.to_dict()
    else:
        payload = score
    return json.dumps(_json_ready(payload), indent=2, sort_keys=True)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value
