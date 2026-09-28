"""Evaluation metrics for NLB2 model outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Literal

import numpy as np
import torch
from torch import Tensor

from nlb2.models.base import BaseDynamicsModel
from nlb2.types import ModelOutput, move_batch_to_device, observations_from_batch


EPS = 1e-8


@dataclass
class EvaluationResult:
    """Metrics and arrays collected from model evaluation."""

    metrics: dict[str, float]
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)


EvaluationTaskName = Literal["synthetic", "nlb"]


class EvaluationAdapter:
    """Task-specific model-output adapter used by benchmark evaluation."""

    task: EvaluationTaskName

    def fit(
        self,
        model: BaseDynamicsModel,
        loader: Iterable | None,
        device: torch.device,
    ) -> None:
        """Fit task-level readouts using training data when required."""

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ) -> EvaluationResult:
        """Collect predictions, targets, and task-specific metrics."""

        raise NotImplementedError


class SyntheticEvaluationAdapter(EvaluationAdapter):
    """Evaluate rate errors in Hz and Poisson scores in expected counts per bin.

    Saved `rates` are Hz; `count_rates` are expected counts. Target batches with
    non-unit bin widths must declare `rates_unit` as `hz` or `counts`.
    `use_predict_rates` honors model-specific prediction averaging while keeping
    forward-pass latents available for latent reconstruction metrics.
    """

    task: EvaluationTaskName = "synthetic"

    def __init__(self, use_raw_spikes: bool = False, use_predict_rates: bool = False) -> None:
        self.use_raw_spikes = use_raw_spikes
        self.use_predict_rates = use_predict_rates

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ) -> EvaluationResult:
        rate_predictions: list[Tensor] = []
        count_predictions: list[Tensor] = []
        latent_predictions: list[Tensor] = []
        observed_spikes: list[Tensor] = []
        true_rates: list[Tensor] = []
        true_latents: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                if self.use_raw_spikes and isinstance(batch, dict):
                    x = batch.get("raw_spikes", x)
                output = model(x)
                dt = _batch_bin_widths(batch, x, model=model)
                if self.use_predict_rates:
                    count_rates = ModelOutput(
                        rates=model.predict_rates(x), rates_unit=output.rates_unit
                    ).count_rates(dt)
                else:
                    count_rates = output.count_rates(dt)
                if count_rates is None:
                    rates = output.reconstruction
                    if rates is None:
                        rates = model.predict_rates(x)
                    count_rates = ModelOutput(
                        rates=rates, rates_unit=output.rates_unit
                    ).count_rates(dt)
                count_predictions.append(count_rates.detach().cpu())
                rate_predictions.append((count_rates / dt).detach().cpu())

                if output.latents is not None:
                    latent_predictions.append(output.latents.detach().cpu())
                if isinstance(batch, dict):
                    observed = batch.get("raw_spikes", batch.get("spikes"))
                    if observed is not None:
                        observed_spikes.append(observed.detach().cpu())
                    if "rates" in batch:
                        unit = _target_rate_unit(batch, dt)
                        rates = batch["rates"]
                        if unit == "counts":
                            rates = rates / dt
                        true_rates.append(rates.detach().cpu())
                    if "latents" in batch:
                        true_latents.append(batch["latents"].detach().cpu())

        predictions: dict[str, Tensor] = {}
        targets: dict[str, Tensor] = {}
        if rate_predictions:
            predictions["rates"] = torch.cat(rate_predictions, dim=0)
            predictions["count_rates"] = torch.cat(count_predictions, dim=0)
        if latent_predictions:
            predictions["latents"] = torch.cat(latent_predictions, dim=0)
        if observed_spikes:
            targets["spikes"] = torch.cat(observed_spikes, dim=0)
        if true_rates:
            targets["rates"] = torch.cat(true_rates, dim=0)
        if true_latents:
            targets["latents"] = torch.cat(true_latents, dim=0)

        metrics = compute_available_metrics(predictions, targets)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in predictions.items()},
            targets={key: value.numpy() for key, value in targets.items()},
        )


class NLBCoSmoothingAdapter(EvaluationAdapter):
    """Evaluate held-in to held-out NLB co-smoothing predictions."""

    task: EvaluationTaskName = "nlb"

    def __init__(
        self,
        feature_source: Literal[
            "auto",
            "latents",
            "rates",
            "reconstruction",
            "predict_rates",
        ] = "auto",
        decoder: Literal["ridge", "poisson"] = "ridge",
        ridge_alpha: float = 1e-4,
        poisson_max_iter: int = 100,
        poisson_eta_clip: float = 20.0,
        prediction_floor: float = 1e-9,
        target: Literal["heldout", "heldin"] = "heldout",
    ) -> None:
        self.feature_source = feature_source
        self.decoder = decoder
        self.ridge_alpha = float(ridge_alpha)
        self.poisson_max_iter = int(poisson_max_iter)
        self.poisson_eta_clip = float(poisson_eta_clip)
        self.prediction_floor = float(prediction_floor)
        self.target = target
        self.decoder_weight: Tensor | None = None
        self.direct_prediction = False

    def fit(
        self,
        model: BaseDynamicsModel,
        loader: Iterable | None,
        device: torch.device,
    ) -> None:
        if loader is None:
            return

        features: list[Tensor] = []
        targets: list[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                output = None if self.feature_source == "predict_rates" else model(x)
                target = _nlb_target_from_batch(batch, target=self.target)
                if (
                    self._allow_direct_prediction()
                    and output is not None
                    and output.rates is not None
                    and output.rates.shape == target.shape
                ):
                    self.direct_prediction = True
                    return
                feature = self._features_from_output(output, model=model, x=x)
                features.append(feature.detach())
                targets.append(target.detach())

        if not features:
            return

        x_flat = torch.cat([item.reshape(-1, item.shape[-1]) for item in features], dim=0)
        y_flat = torch.cat([item.reshape(-1, item.shape[-1]) for item in targets], dim=0)
        finite = torch.isfinite(y_flat).all(dim=1) & torch.isfinite(x_flat).all(dim=1)
        x_flat = x_flat[finite].double()
        y_flat = y_flat[finite].double()
        if x_flat.numel() == 0:
            raise ValueError("No finite NLB training targets available for decoder fitting.")

        if self.decoder == "poisson":
            self.decoder_weight = self._fit_poisson_decoder(x_flat, y_flat)
            return

        ones = torch.ones(x_flat.shape[0], 1, device=x_flat.device, dtype=x_flat.dtype)
        design = torch.cat([x_flat, ones], dim=1)
        penalty = self.ridge_alpha * torch.eye(
            design.shape[1],
            device=design.device,
            dtype=design.dtype,
        )
        penalty[-1, -1] = 0.0
        lhs = design.T @ design + penalty
        rhs = design.T @ y_flat
        self.decoder_weight = torch.linalg.solve(lhs, rhs).float()

    def _fit_poisson_decoder(self, features: Tensor, targets: Tensor) -> Tensor:
        ones = torch.ones(features.shape[0], 1, device=features.device, dtype=features.dtype)
        design = torch.cat([features, ones], dim=1)
        weight = torch.zeros(
            design.shape[1],
            targets.shape[1],
            device=features.device,
            dtype=features.dtype,
            requires_grad=True,
        )
        with torch.no_grad():
            mean_rates = targets.mean(dim=0).clamp_min(self.prediction_floor)
            weight[-1].copy_(torch.log(mean_rates))

        optimizer = torch.optim.LBFGS(
            [weight],
            lr=1.0,
            max_iter=max(self.poisson_max_iter, 1),
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            optimizer.zero_grad(set_to_none=True)
            eta = (design @ weight).clamp(max=self.poisson_eta_clip)
            loss = (torch.exp(eta) - targets * eta).mean()
            if self.ridge_alpha > 0.0:
                loss = loss + self.ridge_alpha * weight[:-1].pow(2).mean()
            loss.backward()
            return loss

        optimizer.step(closure)
        return weight.detach().float()

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        device: torch.device,
    ) -> EvaluationResult:
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        latents: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                output = None if self.feature_source == "predict_rates" else model(x)
                target = _nlb_target_from_batch(batch, target=self.target)
                prediction = self._predict_from_output(
                    output, target, model=model, x=x,
                    dt=_batch_bin_widths(batch, target, model=model),
                )
                predictions.append(prediction.detach().cpu())
                targets.append(target.detach().cpu())
                if output is not None and output.latents is not None:
                    latents.append(output.latents.detach().cpu())

        pred_rates = torch.cat(predictions, dim=0)
        spikes = torch.cat(targets, dim=0)
        pred_dict: dict[str, Tensor] = {"rates": pred_rates}
        target_dict: dict[str, Tensor] = {"spikes": spikes}
        if latents:
            pred_dict["latents"] = torch.cat(latents, dim=0)
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )

    def _predict_from_output(
        self,
        output,
        target: Tensor,
        model: BaseDynamicsModel | None = None,
        x: Tensor | None = None,
        dt: float | Tensor = 1.0,
    ) -> Tensor:
        if (
            self._allow_direct_prediction()
            and output is not None
            and output.rates is not None
            and output.rates.shape == target.shape
        ):
            return output.count_rates(dt).clamp_min(self.prediction_floor)
        if self.decoder_weight is None:
            raise RuntimeError(
                "NLB decoder was not fitted and the model did not directly return held-out rates."
            )
        feature = self._features_from_output(output, model=model, x=x)
        flat = feature.reshape(-1, feature.shape[-1]).float()
        ones = torch.ones(flat.shape[0], 1, device=flat.device, dtype=flat.dtype)
        design = torch.cat([flat, ones], dim=1)
        weight = self.decoder_weight.to(device=design.device, dtype=design.dtype)
        decoded = design @ weight
        if self.decoder == "poisson":
            decoded = torch.exp(decoded.clamp(max=self.poisson_eta_clip))
        return decoded.reshape(*feature.shape[:-1], weight.shape[1]).clamp_min(
            self.prediction_floor
        )

    def _features_from_output(
        self,
        output,
        model: BaseDynamicsModel | None = None,
        x: Tensor | None = None,
    ) -> Tensor:
        if self.feature_source == "predict_rates":
            if model is None or x is None:
                raise RuntimeError("predict_rates features require the model and input batch.")
            return model.predict_rates(x)
        if output is None:
            raise RuntimeError("Model output is required for this NLB feature source.")
        if self.feature_source == "latents":
            if output.latents is None:
                raise RuntimeError("Model did not return latents for NLB decoder fitting.")
            return output.latents
        if self.feature_source == "rates":
            if output.rates is None:
                raise RuntimeError("Model did not return rates for NLB decoder fitting.")
            return output.rates
        if self.feature_source == "reconstruction":
            if output.reconstruction is None:
                raise RuntimeError("Model did not return reconstruction for NLB decoder fitting.")
            return output.reconstruction
        if output.latents is not None:
            return output.latents
        if output.rates is not None:
            return output.rates
        if output.reconstruction is not None:
            return output.reconstruction
        if model is not None and x is not None:
            return model.predict_rates(x)
        raise RuntimeError("Model output has no usable NLB decoder features.")

    def _allow_direct_prediction(self) -> bool:
        return self.feature_source in {"auto", "rates"}


def poisson_negative_log_likelihood(
    rates: Tensor,
    spikes: Tensor,
    eps: float = EPS,
) -> Tensor:
    """Poisson negative log likelihood without the spike factorial constant."""

    rates = rates.clamp_min(eps)
    return rates - spikes * torch.log(rates)


def bits_per_spike(
    rates: Tensor,
    spikes: Tensor,
    eps: float = 1e-9,
) -> float:
    """NLB-style improvement over a per-neuron mean-rate Poisson baseline."""

    if rates.shape != spikes.shape:
        raise ValueError(
            f"rates and spikes shapes differ: {tuple(rates.shape)} != {tuple(spikes.shape)}"
        )

    rates = rates.float()
    spikes = spikes.float()
    valid = torch.isfinite(spikes)
    if torch.isnan(rates[valid]).any():
        raise ValueError("NaN rate predictions found")
    if (rates[valid] < 0).any():
        raise ValueError("negative rate predictions found")

    reduce_dims = tuple(range(max(spikes.ndim - 1, 0)))
    null_rates = torch.nanmean(spikes, dim=reduce_dims, keepdim=True).expand_as(spikes)
    model_rates = rates[valid].clamp_min(eps)
    null_rates = null_rates[valid].clamp_min(eps)
    valid_spikes = spikes[valid]
    model_nll = poisson_negative_log_likelihood(model_rates, valid_spikes, eps=eps).sum()
    null_nll = poisson_negative_log_likelihood(null_rates, valid_spikes, eps=eps).sum()
    total_spikes = torch.nansum(spikes)
    if float(total_spikes) <= 0.0:
        return float("nan")
    return float(((null_nll - model_nll) / total_spikes / math.log(2.0)).detach().cpu())


def r2_score(prediction: Tensor, target: Tensor, eps: float = EPS) -> float:
    """Coefficient of determination over all samples and timesteps."""

    target_mean = target.mean(dim=tuple(range(max(target.ndim - 1, 0))), keepdim=True)
    ss_res = torch.sum((target - prediction) ** 2)
    ss_tot = torch.sum((target - target_mean) ** 2)
    if float(ss_tot) <= eps:
        return float("nan")
    return float((1.0 - ss_res / ss_tot).detach().cpu())


def linear_r2_score(prediction: Tensor, target: Tensor, eps: float = EPS) -> float:
    """R2 after fitting the best affine map from prediction to target."""

    if prediction.shape[:-1] != target.shape[:-1]:
        return float("nan")

    x = prediction.reshape(-1, prediction.shape[-1]).double()
    y = target.reshape(-1, target.shape[-1]).double()
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    design = torch.cat([x, ones], dim=1)
    solution = torch.linalg.lstsq(design, y).solution
    fitted = design @ solution
    return r2_score(fitted, y, eps=eps)


def evaluate_model(
    model: BaseDynamicsModel,
    loader: Iterable,
    device: torch.device | str = "cpu",
    train_loader: Iterable | None = None,
) -> EvaluationResult:
    """Evaluate a trained model on a dataloader using available outputs."""

    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    task = _infer_task_from_loader(loader)
    adapter = model.evaluation_adapter(task)
    if adapter is None:
        adapter = _default_adapter(task)
    adapter.fit(model, train_loader, torch_device)
    return adapter.evaluate(model, loader, torch_device)


def _infer_task_from_loader(loader: Iterable) -> EvaluationTaskName:
    for batch in loader:
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            return "nlb"
        return "synthetic"
    return "synthetic"


def _default_adapter(task: EvaluationTaskName) -> EvaluationAdapter:
    if task == "nlb":
        return NLBCoSmoothingAdapter()
    return SyntheticEvaluationAdapter()


def _nlb_target_from_batch(
    batch: Tensor | dict[str, Tensor],
    target: Literal["heldout", "heldin"] = "heldout",
) -> Tensor:
    if not isinstance(batch, dict):
        raise TypeError("NLB evaluation requires dict batches with heldout_spikes.")
    if target == "heldin":
        if "heldin_spikes" in batch:
            return batch["heldin_spikes"]
        if "spikes" in batch:
            return batch["spikes"]
        raise KeyError("NLB batch is missing heldin_spikes/spikes.")
    if "heldout_spikes" in batch:
        return batch["heldout_spikes"]
    if "raw_spikes" in batch:
        return batch["raw_spikes"]
    raise KeyError("NLB batch is missing heldout_spikes/raw_spikes.")


def compute_available_metrics(
    predictions: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> dict[str, float]:
    """Compute metrics from rate targets and expected-count spike predictions.

    `count_rates` supplies Poisson means when `rates` uses physical rate units.
    Existing count-only callers may continue supplying just `rates`.
    """

    metrics: dict[str, float] = {}
    pred_rates = predictions.get("rates")
    pred_counts = predictions.get("count_rates", pred_rates)
    spikes = targets.get("spikes")
    rates = targets.get("rates")
    pred_latents = predictions.get("latents")
    latents = targets.get("latents")

    if pred_counts is not None and spikes is not None and pred_counts.shape == spikes.shape:
        metrics["co_bps"] = bits_per_spike(pred_counts, spikes)
        valid = torch.isfinite(spikes)
        nll = poisson_negative_log_likelihood(pred_counts[valid], spikes[valid]).mean()
        metrics["poisson_nll"] = float(nll.detach().cpu())
    if pred_rates is not None and rates is not None and pred_rates.shape == rates.shape:
        metrics["rate_mse"] = float(torch.mean((pred_rates - rates) ** 2).detach().cpu())
        metrics["rate_r2"] = r2_score(pred_rates, rates)
    if pred_latents is not None and latents is not None and pred_latents.shape[:-1] == latents.shape[:-1]:
        metrics["latent_linear_r2"] = linear_r2_score(pred_latents, latents)

    return metrics


def _batch_bin_widths(batch, reference: Tensor, *, model: BaseDynamicsModel) -> Tensor:
    value = batch.get("dt") if isinstance(batch, dict) else None
    if value is None:
        value = getattr(model, "dt", None)
    dt = torch.as_tensor(
        1.0 if value is None else value,
        device=reference.device,
        dtype=reference.dtype if reference.is_floating_point() else torch.get_default_dtype(),
    )
    if not bool((torch.isfinite(dt) & (dt > 0)).all()):
        raise ValueError("Evaluation requires finite positive bin widths (dt).")
    while dt.ndim < reference.ndim:
        dt = dt.unsqueeze(-1)
    if torch.broadcast_shapes(dt.shape, reference.shape) != reference.shape:
        raise ValueError("dt must broadcast to the observation shape.")
    return dt


def _target_rate_unit(batch: dict, dt: Tensor) -> Literal["counts", "hz"]:
    unit = batch.get("rates_unit")
    if unit is None:
        if not bool((dt == 1).all()):
            raise ValueError("Synthetic rate targets with dt != 1 must declare rates_unit.")
        return "hz"
    units = set(unit) if isinstance(unit, (list, tuple)) else {unit}
    if len(units) != 1 or not units.issubset({"counts", "hz"}):
        raise ValueError("Each batch must declare one rates_unit: 'counts' or 'hz'.")
    return next(iter(units))
