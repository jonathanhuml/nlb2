"""Minimal trainer that enforces a common benchmark lifecycle."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

import torch

from nlb2.models.base import BaseDynamicsModel
from nlb2.training.strategies import OptimizationStrategy
from nlb2.types import StepResult, move_batch_to_device


@dataclass
class TrainerConfig:
    epochs: int = 10
    device: str = "cpu"
    live_eval_interval: int = 0


@dataclass
class EpochReport:
    epoch: int
    train: StepResult
    valid: StepResult | None
    seconds: float
    metrics: dict[str, float] = field(default_factory=dict)


class Trainer:
    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.history: list[EpochReport] = []

    def fit(
        self,
        model: BaseDynamicsModel,
        strategy: OptimizationStrategy,
        train_loader: Iterable,
        valid_loader: Iterable | None = None,
        epoch_metrics: Mapping[str, Callable[[BaseDynamicsModel], float]] | None = None,
        epoch_callback: Callable[[EpochReport], bool | None] | None = None,
        start_epoch: int = 0,
        strategy_state: Mapping[str, object] | None = None,
        on_ready: Callable[[], None] | None = None,
    ) -> list[EpochReport]:
        device = torch.device(self.config.device)
        model.to(device)
        if strategy.fits_training_data_in_epoch:
            if self.config.epochs < 1:
                raise ValueError(f"{strategy.name} requires epochs >= 1 to train from data.")
        else:
            model.fit_training_data(train_loader, device=device)
        strategy.setup(model)
        if strategy_state is not None:
            strategy.load_state_dict(strategy_state)
        elif int(start_epoch) > 0:
            strategy.advance_to_epoch(int(start_epoch))
        if on_ready is not None:
            on_ready()

        stop_epoch = self.config.epochs
        if strategy.max_epochs is not None:
            stop_epoch = min(stop_epoch, strategy.max_epochs)
        for epoch in range(int(start_epoch), stop_epoch):
            start = time.perf_counter()
            strategy.on_epoch_start(model, epoch)
            train_results = strategy.train_epoch(model, train_loader, epoch, device)
            strategy.on_epoch_end(model, epoch)
            seconds = time.perf_counter() - start
            valid_result = self.validate(model, strategy, valid_loader, epoch, device)
            strategy.on_validation_end(model, epoch, valid_result)
            metrics = _compute_epoch_metrics(model, epoch_metrics)

            report = EpochReport(
                epoch=epoch,
                train=_aggregate_results(train_results),
                valid=valid_result,
                seconds=seconds,
                metrics=metrics,
            )
            self.history.append(report)
            if epoch_callback is not None:
                if epoch_callback(report) is False:
                    break

        return self.history

    def validate(
        self,
        model: BaseDynamicsModel,
        strategy: OptimizationStrategy,
        loader: Iterable | None,
        epoch: int,
        device: torch.device,
    ) -> StepResult | None:
        if loader is None:
            return None

        model.eval()
        results = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            results.append(strategy.validation_step(model, batch, epoch))
        return _aggregate_results(results)


def _aggregate_results(results: list[StepResult]) -> StepResult:
    if not results:
        return StepResult(loss=float("nan"), batch_size=0)

    total_batch = sum(max(result.batch_size, 1) for result in results)
    loss = sum(result.loss * max(result.batch_size, 1) for result in results) / total_batch
    metrics: dict[str, float] = {}
    keys = set().union(*(result.metrics.keys() for result in results))
    for key in keys:
        metrics[key] = (
            sum(result.metrics.get(key, 0.0) * max(result.batch_size, 1) for result in results)
            / total_batch
        )

    return StepResult(
        loss=loss,
        metrics=metrics,
        batch_size=total_batch,
        objective=results[0].objective,
    )


def _compute_epoch_metrics(
    model: BaseDynamicsModel,
    metric_fns: Mapping[str, Callable[[BaseDynamicsModel], float]] | None,
) -> dict[str, float]:
    if metric_fns is None:
        return {}
    return {name: float(fn(model)) for name, fn in metric_fns.items()}
