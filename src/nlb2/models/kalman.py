"""Adapter for the bundled dense Kalman filter baseline."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import Field
from torch import Tensor

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.models._filtering_core import DenseKalmanFilterSmoother
from nlb2.types import LossOutput, ModelOutput


@BaseModelConfig.register
class KalmanConfig(BaseModelConfig):
    """Config for the bundled dense Kalman filter baseline."""

    name: Literal["kalman"] = "kalman"
    objective: str = "negative_log_marginal_likelihood"
    dt: float = 0.01
    dataset_name: Optional[str] = None
    save_model: bool = False
    nlb_ridge_alpha: float = 500.0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=1e-2,
            weight_decay=0.0,
            gradient_clip=300.0,
        )
    )

    def build(self, n_neurons: int, n_time: int) -> "Kalman":
        return Kalman(
            n_neurons=n_neurons,
            n_time=n_time,
            dt=self.dt,
            dataset_name=self.dataset_name,
            save_model=self.save_model,
            nlb_ridge_alpha=self.nlb_ridge_alpha,
            objective=self.objective,
        )


class Kalman(BaseDynamicsModel):
    """Dense Kalman filter baseline adapted from the CASSM source.

    ## When to use

    Use Kalman as a dense Bayesian filtering baseline alongside sparse CASSM
    and GPFA. The method uses the full observation update rather than CASSM's
    sparse projection, so it is useful for comparing accuracy and runtime
    against the computation-aware approximation.

    ## Assumptions

    Observations are modeled with Gaussian noise and Matern temporal dynamics.

    ## Outputs

    The training path returns the Kalman marginal-likelihood objective in
    `extras["loss"]`. `predict_rates` returns nonnegative filtered rate
    predictions shaped like the input observations.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        dt: float = 0.01,
        dataset_name: Optional[str] = None,
        save_model: bool = False,
        nlb_ridge_alpha: float = 500.0,
        objective: str = "negative_log_marginal_likelihood",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.dt = float(dt)
        self.dataset_name = dataset_name
        self.save_model = bool(save_model)
        self.nlb_ridge_alpha = float(nlb_ridge_alpha)
        self.objective = objective

        self.core = DenseKalmanFilterSmoother(
            nneurons=self.n_neurons,
            timesteps=self.n_time,
            device=torch.device("cpu"),
            dt=self.dt,
            dataset_name=self.dataset_name,
            save_model=self.save_model,
        )

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("Kalman expects input shape (batch, time, neurons).")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")

        self._sync_core_device(self.device)
        x = x.to(device=self.device, dtype=self.core.obs_noise_values.dtype)
        loss = self.core(x)
        return ModelOutput(extras={"loss": loss})

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        total = output.extras["loss"]
        return LossOutput(
            total=total,
            named_terms={"kalman_nll": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            from nlb2.metrics import NLBCoSmoothingAdapter

            return NLBCoSmoothingAdapter(
                feature_source="predict_rates",
                decoder="ridge",
                ridge_alpha=self.nlb_ridge_alpha,
            )
        return None

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("Kalman expects input shape (batch, time, neurons).")
        self._sync_core_device(self.device)
        x = x.to(device=self.device, dtype=self.core.obs_noise_values.dtype)
        state_means, _ = self.core.filter(x, return_type="for_prediction")
        return state_means[..., 0::2].clamp_min(0.0)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._sync_core_device(self.device)
        return module

    def _sync_core_device(self, device: torch.device) -> None:
        self.core.device = device
        if hasattr(self.core, "dt"):
            self.core.dt = self.core.dt.to(device)
