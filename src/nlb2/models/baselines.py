"""Simple statistical baselines used by NLB and synthetic benchmarks."""

from __future__ import annotations

from typing import Iterable, Literal

import h5py
import torch
from pydantic import Field
from torch import Tensor
import torch.nn.functional as F

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


@BaseModelConfig.register
class SmoothingConfig(BaseModelConfig):
    """Config for Gaussian spike smoothing plus an NLB held-out decoder."""

    name: Literal["smoothing"] = "smoothing"
    objective: str = "smoothed_poisson_nll"
    kern_sd_ms: float = 50.0
    bin_size_ms: float = 5.0
    log_offset: float = 1e-4
    nlb_decoder_alpha: float = 0.01
    nlb_poisson_max_iter: int = 500
    prediction_floor: float = 1e-9
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="inference_only")
    )

    def build(self, n_neurons: int, n_time: int) -> "Smoothing":
        return Smoothing(
            n_neurons=n_neurons,
            n_time=n_time,
            kern_sd_ms=self.kern_sd_ms,
            bin_size_ms=self.bin_size_ms,
            log_offset=self.log_offset,
            nlb_decoder_alpha=self.nlb_decoder_alpha,
            nlb_poisson_max_iter=self.nlb_poisson_max_iter,
            prediction_floor=self.prediction_floor,
            objective=self.objective,
        )


class Smoothing(BaseDynamicsModel):
    """Gaussian-smoothed spike-count baseline.

    ## When to use

    Use Smoothing as a lightweight statistical baseline for binned spike
    counts. It convolves each neuron's spike train with a Gaussian kernel and
    returns nonnegative smoothed count rates. For NLB co-smoothing, the model
    mirrors the public NLB smoothing baseline: log-smoothed held-in counts are
    used as features for a Poisson decoder fitted to training held-out neurons.

    ## Assumptions

    Inputs are nonnegative binned spike counts. `kern_sd_ms` and `bin_size_ms`
    define the Gaussian kernel in the same units as the NLB baseline scripts.
    The default `kern_sd_ms=50` and `bin_size_ms=5` match the public MCMaze
    smoothing defaults.

    ## Outputs

    `forward` returns smoothed counts as `rates` and log-smoothed counts as
    `latents`. On synthetic datasets the rates are scored directly. On NLB
    datasets, the NLB adapter fits a Poisson readout from `latents` to held-out
    spike counts and scores the decoded held-out rates.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        kern_sd_ms: float = 50.0,
        bin_size_ms: float = 5.0,
        log_offset: float = 1e-4,
        nlb_decoder_alpha: float = 0.01,
        nlb_poisson_max_iter: int = 500,
        prediction_floor: float = 1e-9,
        objective: str = "smoothed_poisson_nll",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.kern_sd_ms = float(kern_sd_ms)
        self.bin_size_ms = float(bin_size_ms)
        self.log_offset = float(log_offset)
        self.nlb_decoder_alpha = float(nlb_decoder_alpha)
        self.nlb_poisson_max_iter = int(nlb_poisson_max_iter)
        self.prediction_floor = float(prediction_floor)
        self.objective = objective
        self.register_buffer("kernel", _gaussian_kernel(self.kern_sd_ms, self.bin_size_ms))

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("Smoothing expects input shape (batch, time, neurons).")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")

        rates = self._smooth(x.float()).clamp_min(self.prediction_floor)
        latents = torch.log(rates + self.log_offset)
        return ModelOutput(rates=rates, latents=latents, reconstruction=rates)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        x = _training_count_observations(batch).to(device=output.rates.device, dtype=output.rates.dtype)
        total = _poisson_nll(output.rates, x).mean()
        return LossOutput(
            total=total,
            named_terms={"poisson_nll": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            from nlb2.metrics import NLBCoSmoothingAdapter

            return NLBCoSmoothingAdapter(
                feature_source="latents",
                decoder="poisson",
                ridge_alpha=self.nlb_decoder_alpha,
                poisson_max_iter=self.nlb_poisson_max_iter,
                prediction_floor=self.prediction_floor,
            )
        if task == "synthetic":
            from nlb2.metrics import SyntheticEvaluationAdapter

            return SyntheticEvaluationAdapter(use_raw_spikes=True)
        return None

    def _smooth(self, x: Tensor) -> Tensor:
        kernel = self.kernel.to(device=x.device, dtype=x.dtype)
        if kernel.numel() <= 1:
            return x

        batch, time, neurons = x.shape
        weight = kernel.flip(0).view(1, 1, -1).repeat(neurons, 1, 1)
        padded = F.pad(x.transpose(1, 2), (kernel.numel() - 1, kernel.numel() - 1))
        full = F.conv1d(padded, weight, groups=neurons)
        start = (kernel.numel() - 1) // 2
        return full[..., start : start + time].transpose(1, 2).reshape(batch, time, neurons)


@BaseModelConfig.register
class PSTHConfig(BaseModelConfig):
    """Config for the peri-stimulus time histogram baseline."""

    name: Literal["psth"] = "psth"
    objective: str = "psth_poisson_nll"
    kern_sd_ms: float = 70.0
    bin_size_ms: float = 5.0
    prediction_floor: float = 1e-9
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="library_fit")
    )

    def build(self, n_neurons: int, n_time: int) -> "PSTH":
        return PSTH(
            n_neurons=n_neurons,
            n_time=n_time,
            kern_sd_ms=self.kern_sd_ms,
            bin_size_ms=self.bin_size_ms,
            prediction_floor=self.prediction_floor,
            objective=self.objective,
        )


class PSTH(BaseDynamicsModel):
    """Peri-stimulus time histogram baseline.

    ## When to use

    Use PSTH as the simplest condition-averaged firing-rate baseline. On NLB
    files prepared with condition indices, the adapter smooths training
    held-out spikes, averages them within each condition, and maps those
    condition means onto eval trials. On synthetic datasets without condition
    metadata, it fits a time-varying mean rate from the training loader and
    repeats it for every validation trial.

    ## Assumptions

    NLB condition-index tensors are expected to use the same ordering as
    `train_spikes_heldout` and `eval_spikes_heldout`. `kern_sd_ms` and
    `bin_size_ms` define smoothing applied before training-trial averaging; the
    default 70 ms kernel matches NLB's MC_Maze PSTH construction. The
    target-side `psth` tensor written by `nlb_tools` is not used for prediction
    because it is evaluation metadata. Generic datasets are treated as one
    condition, so this is a deliberately weak time-only baseline.

    ## Outputs

    `forward` requires a fitted training time-varying mean rate. NLB
    evaluation bypasses `forward` and returns condition-matched held-out
    training PSTH rates directly from the prepared H5 tensors.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        kern_sd_ms: float = 70.0,
        bin_size_ms: float = 5.0,
        prediction_floor: float = 1e-9,
        objective: str = "psth_poisson_nll",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.kern_sd_ms = float(kern_sd_ms)
        self.bin_size_ms = float(bin_size_ms)
        self.prediction_floor = float(prediction_floor)
        self.objective = objective
        self.register_buffer("psth_rates", torch.empty(0))
        self.register_buffer("kernel", _gaussian_kernel(self.kern_sd_ms, self.bin_size_ms))

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("PSTH expects input shape (batch, time, neurons).")
        if self.psth_rates.numel() == 0:
            raise RuntimeError("PSTH must be fitted on training data before prediction.")
        if tuple(x.shape[1:]) != (self.n_time, self.n_neurons):
            raise ValueError(f"PSTH expects time/neuron dimensions {(self.n_time, self.n_neurons)}, got {tuple(x.shape[1:])}.")
        dtype = x.dtype if x.is_floating_point() else self.psth_rates.dtype
        rates = self.psth_rates.to(device=x.device, dtype=dtype)
        rates = rates.unsqueeze(0).expand(x.shape[0], -1, -1)
        rates = rates.clamp_min(self.prediction_floor)
        return ModelOutput(rates=rates, reconstruction=rates)

    def fit_training_data(self, loader, *, device) -> None:
        if self.psth_rates.numel() == 0:
            self.fit_psth_from_loader(loader, torch.device(device))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved = state_dict.get(prefix + "psth_rates")
        if saved is not None:
            expected = (self.n_time, self.n_neurons)
            if tuple(saved.shape) not in {(0,), expected}:
                error_msgs.append(f"PSTH checkpoint dimensions must be {expected}, got {tuple(saved.shape)}.")
            elif not bool(torch.isfinite(saved).all()) or bool((saved < 0).any()):
                error_msgs.append("PSTH checkpoint rates must be finite and nonnegative.")
            else:
                self.psth_rates = self.psth_rates.new_empty(saved.shape)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        x = _training_count_observations(batch).to(device=output.rates.device, dtype=output.rates.dtype)
        total = _poisson_nll(output.rates, x).mean()
        return LossOutput(
            total=total,
            named_terms={"poisson_nll": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task in {"synthetic", "nlb"}:
            return PSTHEvaluationAdapter(task=task)
        return None

    def fit_psth_from_loader(self, loader: Iterable | None, device: torch.device) -> None:
        if loader is None:
            raise ValueError("PSTH fitting requires a training loader.")
        sums: Tensor | None = None
        count = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = _training_count_observations(batch).float()
                if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_time, self.n_neurons):
                    raise ValueError("PSTH training data does not match its configured time/neuron dimensions.")
                if not bool(torch.isfinite(x).all()) or bool((x < 0).any()):
                    raise ValueError("PSTH requires finite nonnegative training spike counts.")
                x = _smooth_trials(x, self.kernel)
                batch_sum = x.sum(dim=0)
                sums = batch_sum if sums is None else sums + batch_sum
                count += int(x.shape[0])
        if sums is None or count == 0:
            raise ValueError("PSTH fitting requires nonempty training data.")
        self.psth_rates = (sums / count).clamp_min(self.prediction_floor)


class PSTHEvaluationAdapter:
    """Evaluation adapter for PSTH baselines."""

    def __init__(self, task: str) -> None:
        self.task = task
        self.fallback_rates: Tensor | None = None

    def fit(
        self,
        model: BaseDynamicsModel,
        loader: Iterable | None,
        device: torch.device,
    ) -> None:
        if not isinstance(model, PSTH):
            raise TypeError("PSTHEvaluationAdapter requires a PSTH model.")
        model.fit_training_data(loader, device=device)
        if self.task != "nlb" or loader is None:
            return

        targets = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                if isinstance(batch, dict) and "heldout_spikes" in batch:
                    targets.append(batch["heldout_spikes"].float())
        if targets:
            target_rates = _smooth_trials(torch.cat(targets, dim=0), model.kernel)
            self.fallback_rates = target_rates.mean(dim=0).clamp_min(
                model.prediction_floor
            )

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ):
        if self.task == "nlb":
            return self._evaluate_nlb(model, loader, device)
        return self._evaluate_synthetic(model, loader, device)

    def _evaluate_synthetic(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ):
        from nlb2.metrics import SyntheticEvaluationAdapter

        return SyntheticEvaluationAdapter(use_raw_spikes=True).evaluate(model, loader, device)

    def _evaluate_nlb(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ):
        from nlb2.metrics import EvaluationResult, compute_available_metrics

        dataset = _unwrap_dataset(getattr(loader, "dataset", None))
        targets = _collect_nlb_targets(loader, device)
        rates = None
        if dataset is not None:
            rates = _rates_from_nlb_condition_psth(
                dataset,
                targets,
                model.kernel,
                float(model.prediction_floor),
            )
        if rates is None:
            if self.fallback_rates is None:
                raise RuntimeError("PSTH could not find NLB PSTH tensors or fallback rates.")
            rates = self.fallback_rates.to(device=targets.device, dtype=targets.dtype)
            rates = rates.unsqueeze(0).expand_as(targets)

        pred_dict = {"rates": rates.detach().cpu()}
        target_dict = {"spikes": targets.detach().cpu()}
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )


def _gaussian_kernel(kern_sd_ms: float, bin_size_ms: float) -> Tensor:
    if kern_sd_ms <= 0.0 or bin_size_ms <= 0.0:
        return torch.ones(1, dtype=torch.float32)
    width = max(int(6.0 * kern_sd_ms / bin_size_ms), 1)
    sd = max(float(int(kern_sd_ms / bin_size_ms)), 1.0)
    coords = torch.arange(width, dtype=torch.float32) - 0.5 * (width - 1)
    kernel = torch.exp(-0.5 * (coords / sd).pow(2))
    return kernel / kernel.sum().clamp_min(1e-12)


def _poisson_nll(rates: Tensor | None, spikes: Tensor) -> Tensor:
    if rates is None:
        raise RuntimeError("Model output is missing rates.")
    rates = rates.clamp_min(1e-9)
    return rates - spikes * torch.log(rates)


def _unwrap_dataset(dataset):
    while dataset is not None and hasattr(dataset, "dataset"):
        dataset = getattr(dataset, "dataset")
    return dataset


def _collect_nlb_targets(loader: Iterable, device: torch.device) -> Tensor:
    targets = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if not isinstance(batch, dict) or "heldout_spikes" not in batch:
                raise TypeError("PSTH NLB evaluation requires heldout_spikes batches.")
            targets.append(batch["heldout_spikes"].float())
    if not targets:
        raise RuntimeError("PSTH NLB evaluation received an empty loader.")
    return torch.cat(targets, dim=0)


def _rates_from_nlb_condition_psth(
    dataset,
    targets: Tensor,
    kernel: Tensor,
    prediction_floor: float,
) -> Tensor | None:
    config = getattr(dataset, "config", None)
    if config is None or not hasattr(config, "resolved_data_path"):
        return None

    path = config.resolved_data_path
    if not path.exists():
        return None

    with h5py.File(path, "r") as handle:
        group_name = getattr(config, "resolved_group", None)
        group = handle[group_name] if isinstance(group_name, str) and group_name in handle else handle
        if "train_cond_idx" not in group or "eval_cond_idx" not in group:
            return None
        train_cond_idx = group["train_cond_idx"][()]
        cond_idx = group["eval_cond_idx"][()]

    arrays = getattr(dataset, "arrays", None)
    train_heldout = getattr(arrays, "train_heldout_spikes", None)
    if train_heldout is None:
        return None
    train_heldout = train_heldout.to(device=targets.device, dtype=targets.dtype)
    train_heldout = train_heldout[:, : targets.shape[1], : targets.shape[-1]]
    train_heldout = _smooth_trials(train_heldout, kernel)

    fallback = train_heldout.mean(dim=0).clamp_min(prediction_floor)
    rates = fallback.unsqueeze(0).expand_as(targets).clone()
    for condition, trial_indices in enumerate(cond_idx):
        if condition >= len(train_cond_idx) or len(trial_indices) == 0:
            continue
        train_idx = torch.as_tensor(
            train_cond_idx[condition],
            dtype=torch.long,
            device=targets.device,
        )
        train_valid = train_idx[(train_idx >= 0) & (train_idx < train_heldout.shape[0])]
        if train_valid.numel() == 0:
            condition_rate = fallback
        else:
            condition_rate = train_heldout.index_select(0, train_valid).mean(dim=0)
        idx = torch.as_tensor(trial_indices, dtype=torch.long, device=targets.device)
        valid = idx[(idx >= 0) & (idx < targets.shape[0])]
        if valid.numel() == 0:
            continue
        rates[valid] = condition_rate.clamp_min(prediction_floor)
    return torch.nan_to_num(rates, nan=prediction_floor).clamp_min(prediction_floor)


def _smooth_trials(x: Tensor, kernel: Tensor) -> Tensor:
    kernel = kernel.to(device=x.device, dtype=x.dtype)
    if kernel.numel() <= 1:
        return x
    batch, time, neurons = x.shape
    weight = kernel.flip(0).view(1, 1, -1).repeat(neurons, 1, 1)
    padded = F.pad(x.transpose(1, 2), (kernel.numel() - 1, kernel.numel() - 1))
    full = F.conv1d(padded, weight, groups=neurons)
    start = (kernel.numel() - 1) // 2
    return full[..., start : start + time].transpose(1, 2).reshape(batch, time, neurons)


def _training_count_observations(batch) -> Tensor:
    if isinstance(batch, dict) and "heldout_spikes" not in batch:
        return batch.get("raw_spikes", batch["spikes"])
    return observations_from_batch(batch)
