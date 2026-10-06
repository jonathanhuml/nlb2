"""Optimization strategy abstraction.

The benchmark trainer owns epochs, logging, timing, and dataloaders. Strategies
own the parameter update procedure so gradient, variational, and EM-style
methods can share the same outer training/reporting contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from nlb2.models.base import BaseDynamicsModel, OptimizationConfig
from nlb2.types import LossOutput, StepResult, move_batch_to_device, observations_from_batch


class OptimizationStrategy(ABC):
    name: str
    requires_ordered_training_data = False
    fits_training_data_in_epoch = False
    max_epochs: int | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        """Initialize optimizer state."""

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer/scheduler state for resumable training."""

        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore optimizer/scheduler state after setup."""

    def advance_to_epoch(self, epoch: int) -> None:
        """Advance epoch-scoped scheduler state when no saved state exists."""

    def on_epoch_start(self, model: BaseDynamicsModel, epoch: int) -> None:
        """Hook before an epoch starts."""

        set_epoch = getattr(model, "set_training_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

    def on_epoch_end(self, model: BaseDynamicsModel, epoch: int) -> None:
        """Hook after an epoch ends."""

    def on_validation_end(
        self,
        model: BaseDynamicsModel,
        epoch: int,
        valid_result: StepResult | None,
    ) -> None:
        """Hook after validation finishes."""

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        results = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            results.append(self.step(model, batch, epoch))
        return results

    @abstractmethod
    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        """Run one training update."""

    def validation_step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        with torch.no_grad():
            x = observations_from_batch(batch)
            x = _loss_forward_observations(model, batch, x)
            output = model(x)
            loss = model.loss(batch, output, epoch=epoch)
            return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class GradientStrategy(OptimizationStrategy):
    name = "gradient"

    def __init__(
        self,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip: float | None = None,
        lr_scheduler: str | None = None,
        scheduler_factor: float = 0.95,
        scheduler_patience: int = 10,
        scheduler_threshold: float = 0.0,
        scheduler_min_lr: float = 1e-5,
        sqrt_decay_scale: float = 1.0,
        warmup_steps: int = 0,
        total_steps: int | None = None,
        scheduler_step: str = "batch",
    ) -> None:
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.lr_scheduler_name = lr_scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.scheduler_min_lr = scheduler_min_lr
        self.sqrt_decay_scale = float(sqrt_decay_scale)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.scheduler_step = str(scheduler_step)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: ReduceLROnPlateau | LambdaLR | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        self.optimizer = optimizer_cls(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = None
        if self.lr_scheduler_name is None:
            return
        if self.lr_scheduler_name == "sqrt_decay":
            scale = max(self.sqrt_decay_scale, 1.0e-12)
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0 / (1.0 + np.sqrt(float(step) / scale)),
            )
            return
        if self.lr_scheduler_name == "warmup_cosine":
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=self._warmup_cosine_factor)
            return
        if self.lr_scheduler_name != "ReduceLROnPlateau":
            raise ValueError(f"Unsupported lr_scheduler '{self.lr_scheduler_name}'.")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
            min_lr=self.scheduler_min_lr,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])

    def advance_to_epoch(self, epoch: int) -> None:
        if (
            self.optimizer is None
            or not isinstance(self.scheduler, LambdaLR)
            or self.scheduler_step != "epoch"
        ):
            return
        epoch = max(int(epoch), 0)
        for group, base_lr, lr_lambda in zip(
            self.optimizer.param_groups,
            self.scheduler.base_lrs,
            self.scheduler.lr_lambdas,
        ):
            group["lr"] = float(base_lr) * float(lr_lambda(epoch))
        self.scheduler.last_epoch = epoch
        self.scheduler._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        if self.optimizer is None:
            raise RuntimeError("GradientStrategy.setup() must be called before training.")

        model.train()
        x = observations_from_batch(batch)
        x = _loss_forward_observations(model, batch, x)
        output = model(x)
        loss = model.loss(batch, output, epoch=epoch)

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        if hasattr(model, "on_before_optimizer_step"):
            model.on_before_optimizer_step(self.optimizer, epoch)
        self.optimizer.step()
        if isinstance(self.scheduler, LambdaLR):
            self._step_lambda_scheduler("batch")
        if hasattr(model, "project_parameters"):
            model.project_parameters()

        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))

    def on_epoch_end(self, model: BaseDynamicsModel, epoch: int) -> None:
        del model, epoch
        self._step_lambda_scheduler("epoch")

    def on_validation_end(
        self,
        model: BaseDynamicsModel,
        epoch: int,
        valid_result: StepResult | None,
    ) -> None:
        del model, epoch
        if isinstance(self.scheduler, ReduceLROnPlateau) and valid_result is not None:
            self.scheduler.step(valid_result.loss)

    def _step_lambda_scheduler(self, step_kind: str) -> None:
        if isinstance(self.scheduler, LambdaLR) and self.scheduler_step == step_kind:
            self.scheduler.step()

    def _warmup_cosine_factor(self, step: int) -> float:
        step = max(int(step), 1)
        warmup = max(self.warmup_steps, 0)
        total = max(self.total_steps or warmup + 1, warmup + 1)
        if warmup > 0 and step <= warmup:
            return float(step) / float(warmup)
        progress = min(1.0, max(0.0, float(step - warmup) / float(total - warmup)))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


class FullBatchGradientStrategy(OptimizationStrategy):
    """Gradient strategy for models with dataset-shaped variational parameters."""

    name = "full_batch_gradient"

    def __init__(
        self,
        optimizer: str = "AdamW",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip: float | None = None,
    ) -> None:
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip
        self.optimizer: torch.optim.Optimizer | None = None
        self._pending_optimizer_state: Mapping[str, Any] | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        self.optimizer = None
        self._pending_optimizer_state = None

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict() if self.optimizer is not None else self._pending_optimizer_state}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._pending_optimizer_state = state.get("optimizer")

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        batches = [move_batch_to_device(batch, device) for batch in loader]
        if not batches:
            return []

        batch = _concat_batches(batches)
        return [self.step(model, batch, epoch)]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        model.train()
        x = observations_from_batch(batch)
        x = _loss_forward_observations(model, batch, x)
        output = model(x)
        loss = model.loss(batch, output, epoch=epoch)

        if self.optimizer is None:
            optimizer_cls = getattr(torch.optim, self.optimizer_name)
            params = [param for param in model.parameters() if param.requires_grad]
            if not params:
                raise RuntimeError(
                    f"{type(model).__name__} has no trainable parameters after forward()."
                )
            self.optimizer = optimizer_cls(
                params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            if self._pending_optimizer_state is not None:
                self.optimizer.load_state_dict(self._pending_optimizer_state)
                self._pending_optimizer_state = None

        self.optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
        if hasattr(model, "on_before_optimizer_step"):
            model.on_before_optimizer_step(self.optimizer, epoch)
        self.optimizer.step()
        if hasattr(model, "project_parameters"):
            model.project_parameters()

        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class MgplvmFullBatchGradientStrategy(OptimizationStrategy):
    """Full-batch gradient strategy matching mgplvm's SVGP optimizer loop."""

    name = "mgplvm_full_batch_gradient"
    requires_ordered_training_data = True

    def __init__(
        self,
        optimizer: str = "Adam",
        lr: float = 1e-1,
        steps_per_epoch: int = 1,
        burnin: int = 150,
        n_mc: int = 3,
        batch_mc: int | None = None,
        prior_m: int | None = None,
        analytic_kl: bool = True,
        weight_decay: float = 0.0,
        gradient_clip: float | None = None,
    ) -> None:
        self.optimizer_name = optimizer
        self.lr = float(lr)
        if int(steps_per_epoch) < 1:
            raise ValueError("steps_per_epoch must be >= 1.")
        self.steps_per_epoch = int(steps_per_epoch)
        self.burnin = int(burnin)
        self.n_mc = int(n_mc)
        self.batch_mc = batch_mc
        self.prior_m = prior_m
        self.analytic_kl = bool(analytic_kl)
        self.weight_decay = float(weight_decay)
        self.gradient_clip = gradient_clip
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: LambdaLR | None = None
        self._hooks = []
        self._step_index = 0
        self._pending_state: Mapping[str, Any] | None = None

    def setup(self, model: BaseDynamicsModel) -> None:
        self.optimizer = None
        self.scheduler = None
        self._hooks = []
        self._step_index = 0
        self._pending_state = None

    def state_dict(self) -> dict[str, Any]:
        if self._pending_state is not None:
            return dict(self._pending_state)
        return {
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "step_index": self._step_index,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._pending_state = state
        self._step_index = int(state["step_index"])

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        batches = [move_batch_to_device(batch, device) for batch in loader]
        if not batches:
            return []
        batch = _concat_batches(batches)
        result = None
        for _ in range(self.steps_per_epoch):
            result = self.step(model, batch, epoch)
        if result is None:
            return []
        return [result]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        if not hasattr(model, "mgplvm_training_model") or not hasattr(
            model, "mgplvm_observations"
        ):
            raise TypeError(
                f"{type(model).__name__} does not expose mgplvm training hooks."
            )

        model.train()
        x = observations_from_batch(batch)
        mod = model.mgplvm_training_model(x)
        y = model.mgplvm_observations(x)

        if self.optimizer is None:
            self._setup_optimizer(mod)

        if self.optimizer is None:
            raise RuntimeError("MgplvmFullBatchGradientStrategy optimizer was not initialized.")

        self.optimizer.zero_grad(set_to_none=True)
        ramp = self._kl_ramp()
        mc_batches = self._mc_batches()
        loss_values = []
        kl_values = []
        elbo_values = []

        for mc in mc_batches:
            mc_weight = mc / self.n_mc
            svgp_elbo, latent_kl = mod(
                y,
                mc,
                m=self.prior_m,
                analytic_kl=self.analytic_kl,
            )
            loss = -svgp_elbo + ramp * latent_kl
            loss_values.append(float(loss.detach().cpu()) * mc_weight)
            kl_values.append(float(latent_kl.detach().cpu()) * mc_weight)
            elbo_values.append(float(svgp_elbo.detach().cpu()) * mc_weight)
            (loss * mc_weight).backward()

        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(mod.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        self._step_index += 1

        normalizer = max(int(mod.n) * int(mod.m) * int(mod.n_samples), 1)
        reported = y.new_tensor(float(np.sum(loss_values)) / normalizer)
        loss = LossOutput(
            total=reported,
            named_terms={
                "negative_elbo": reported,
                "svgp_elbo": y.new_tensor(float(np.sum(elbo_values))),
                "latent_kl": y.new_tensor(float(np.sum(kl_values))),
                "kl_weight": ramp,
                "optimizer_step": float(self._step_index),
            },
            objective=getattr(model, "objective", "negative_elbo"),
        )
        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))

    def _setup_optimizer(self, mod: torch.nn.Module) -> None:
        from nlb2.models.bgpfa_core.optim import sort_params

        def no_op_hook(grad):
            return grad

        params, self._hooks = sort_params(mod, no_op_hook)
        optimizer_cls = getattr(torch.optim, self.optimizer_name)
        self.optimizer = optimizer_cls(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        def fburn(step):
            return 1.0 - np.exp(-step / (3.0 * max(self.burnin, 1)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda step: 1.0, fburn])
        if self._pending_state is not None:
            if self._pending_state.get("optimizer") is not None:
                self.optimizer.load_state_dict(self._pending_state["optimizer"])
            if self._pending_state.get("scheduler") is not None:
                self.scheduler.load_state_dict(self._pending_state["scheduler"])
            self._pending_state = None

    def _kl_ramp(self) -> float:
        return float(1.0 - np.exp(-self._step_index / max(self.burnin, 1)))

    def _mc_batches(self) -> list[int]:
        batch_mc = self.n_mc if self.batch_mc is None else int(self.batch_mc)
        batches = [batch_mc for _ in range(self.n_mc // batch_mc)]
        remainder = self.n_mc % batch_mc
        if remainder > 0:
            batches.append(remainder)
        if not batches:
            raise ValueError("n_mc must be positive.")
        return batches


class EMStrategy(OptimizationStrategy):
    """Full-dataset EM strategy.

    One benchmark epoch corresponds to one call to `model.fit_em_epoch(x)`.
    """

    name = "em"

    def setup(self, model: BaseDynamicsModel) -> None:
        if not hasattr(model, "fit_em_epoch") and not hasattr(model, "fit_em_epoch_from_loader"):
            raise TypeError(
                f"{type(model).__name__} does not implement fit_em_epoch() or "
                "fit_em_epoch_from_loader()."
            )

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        if hasattr(model, "fit_em_epoch_from_loader"):
            loss = model.fit_em_epoch_from_loader(loader, device=device, epoch=epoch)
            return [StepResult.from_loss(loss, batch_size=_loader_batch_size(loader))]

        batches = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            batches.append(observations_from_batch(batch))
        x = torch.cat(batches, dim=0)
        loss = model.fit_em_epoch(x, epoch=epoch)
        return [StepResult.from_loss(loss, batch_size=int(x.shape[0]))]

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        x = observations_from_batch(batch)
        loss = model.fit_em_epoch(x, epoch=epoch)
        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class InferenceOnlyStrategy(OptimizationStrategy):
    """No-op strategy for fixed-parameter inference-only models."""

    name = "inference_only"

    def setup(self, model: BaseDynamicsModel) -> None:
        return None

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        return []

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        x = observations_from_batch(batch)
        x = _loss_forward_observations(model, batch, x)
        output = model(x)
        loss = model.loss(batch, output, epoch=epoch)
        return StepResult.from_loss(loss, batch_size=int(x.shape[0]))


class LibraryFitStrategy(OptimizationStrategy):
    """Train a trajectory estimator or fit a statistical library directly."""

    name = "library_fit"
    fits_training_data_in_epoch = True

    def setup(self, model: BaseDynamicsModel) -> None:
        factory = getattr(model, "create_library_trainer", None)
        self.library_trainer = factory() if callable(factory) else None
        self.max_epochs = 1 if self.library_trainer is None else None

    def state_dict(self) -> dict[str, Any]:
        if self.library_trainer is None:
            return {}
        return {"library_trainer": self.library_trainer.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.library_trainer is not None:
            if "library_trainer" not in state:
                raise ValueError("Resuming iterative library training requires its estimator and optimizer state.")
            self.library_trainer.load_state_dict(state["library_trainer"])

    def train_epoch(
        self,
        model: BaseDynamicsModel,
        loader: Iterable,
        epoch: int,
        device: torch.device | str,
    ) -> list[StepResult]:
        model.train()
        metrics = {}
        if self.library_trainer is None:
            model.fit_training_data(loader, device=torch.device(device))
        else:
            metrics = self.library_trainer.train_epoch(loader, epoch, torch.device(device))
        model.eval()
        results = [
            self.step(model, move_batch_to_device(batch, device), epoch)
            for batch in loader
        ]
        for result in results:
            result.metrics.update(metrics)
        return results

    def step(
        self,
        model: BaseDynamicsModel,
        batch: Tensor | dict[str, Tensor],
        epoch: int,
    ) -> StepResult:
        return self.validation_step(model, batch, epoch)


def build_strategy(config: OptimizationConfig) -> OptimizationStrategy:
    kwargs = config.kwargs()
    if config.name == "gradient":
        return GradientStrategy(**kwargs)
    if config.name == "full_batch_gradient":
        return FullBatchGradientStrategy(**kwargs)
    if config.name == "mgplvm_full_batch_gradient":
        return MgplvmFullBatchGradientStrategy(**kwargs)
    if config.name == "em":
        return EMStrategy(**kwargs)
    if config.name == "inference_only":
        return InferenceOnlyStrategy(**kwargs)
    if config.name == "library_fit":
        return LibraryFitStrategy(**kwargs)
    raise KeyError(f"Unknown optimization strategy '{config.name}'.")


def _loss_forward_observations(
    model: BaseDynamicsModel,
    batch: Tensor | dict[str, Tensor],
    x: Tensor,
) -> Tensor:
    hook = getattr(model, "loss_forward_observations", None)
    if callable(hook):
        return hook(batch, x)
    return x


def _concat_batches(batches: list[Tensor | dict[str, Tensor]]) -> Tensor | dict[str, Tensor]:
    first = batches[0]
    if isinstance(first, Tensor):
        return torch.cat([batch for batch in batches if isinstance(batch, Tensor)], dim=0)
    if isinstance(first, dict):
        out: dict[str, Tensor] = {}
        for key, value in first.items():
            if isinstance(value, Tensor):
                out[key] = torch.cat([batch[key] for batch in batches], dim=0)
        return out
    raise TypeError(f"Unsupported batch type {type(first).__name__}.")


def _loader_batch_size(loader: Iterable) -> int:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except TypeError:
            pass
    return 0
