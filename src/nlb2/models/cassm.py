"""Adapter for the bundled sparse CASSM implementation."""

from __future__ import annotations

from typing import Literal, Optional

import torch
from pydantic import Field
from torch import Tensor

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.models._filtering_core import ComputationAwareFilterSmoother
from nlb2.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class CASSMConfig(BaseModelConfig):
    """Config for the bundled sparse CASSM adapter."""

    name: Literal["cassm"] = "cassm"
    objective: str = "cassm_elbo"
    projection_dim: int = Field(default=20, ge=1)
    dt: float = 0.01
    dataset_name: Optional[str] = None
    save_model: bool = False
    use_dense_projection: bool = False
    health_checks: bool = True
    nlb_feature_source: Literal["latents", "rates", "predict_rates"] = "latents"
    nlb_decoder: Literal["ridge", "poisson"] = "ridge"
    nlb_ridge_alpha: float = 500.0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=5e-2,
            weight_decay=0.0,
            gradient_clip=300.0,
        )
    )

    def build(self, n_neurons: int, n_time: int) -> "CASSM":
        return CASSM(
            n_neurons=n_neurons,
            n_time=n_time,
            projection_dim=self.projection_dim,
            dt=self.dt,
            dataset_name=self.dataset_name,
            save_model=self.save_model,
            use_dense_projection=self.use_dense_projection,
            health_checks=self.health_checks,
            nlb_feature_source=self.nlb_feature_source,
            nlb_decoder=self.nlb_decoder,
            nlb_ridge_alpha=self.nlb_ridge_alpha,
            objective=self.objective,
        )


class CASSM(BaseDynamicsModel):
    """Thin wrapper around the bundled CASSM sparse filter/smoother.

    ## When to use

    Use CASSM when benchmarking computation-aware sparse state-space models
    against latent dynamics baselines. NLB2 keeps the compact filtering core
    inside `nlb2.models` and maps it onto the shared model, loss, prediction,
    and device contracts.

    ## Inputs

    `forward` expects observations shaped `(batch, time, neurons)`.

    ## Outputs

    The training path returns the CASSM ELBO-style loss in `extras["loss"]`.
    `predict_rates` calls CASSM's native filtering path and returns nonnegative
    rate predictions shaped like the input observations.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        projection_dim: int = 20,
        dt: float = 0.01,
        dataset_name: Optional[str] = None,
        save_model: bool = False,
        use_dense_projection: bool = False,
        health_checks: bool = True,
        nlb_feature_source: str = "latents",
        nlb_decoder: str = "ridge",
        nlb_ridge_alpha: float = 500.0,
        objective: str = "cassm_elbo",
    ) -> None:
        super().__init__()
        if not 1 <= projection_dim <= n_neurons:
            raise ValueError("projection_dim must be positive and <= n_neurons for CASSM.")

        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.projection_dim = int(projection_dim)
        self.dt = float(dt)
        self.dataset_name = dataset_name
        self.save_model = bool(save_model)
        self.use_dense_projection = bool(use_dense_projection)
        self.health_checks = bool(health_checks)
        self.nlb_feature_source = str(nlb_feature_source)
        self.nlb_decoder = str(nlb_decoder)
        self.nlb_ridge_alpha = float(nlb_ridge_alpha)
        self.objective = objective

        self.core = ComputationAwareFilterSmoother(
            projection_dim=self.projection_dim,
            nneurons=self.n_neurons,
            timesteps=self.n_time,
            device=torch.device("cpu"),
            dt=self.dt,
            dataset_name=self.dataset_name,
            save_model=self.save_model,
            use_dense_projection=self.use_dense_projection,
        )

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("CASSM expects input shape (batch, time, neurons).")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")

        self._sync_core_device(self.device)
        x = x.to(device=self.device, dtype=self.core.obs_noise_values.dtype)
        if not self.training:
            state_means, obs_vars = self.core.filter(x, return_type="prediction")
            return ModelOutput(
                rates=state_means[..., 0::2].clamp_min(0.0),
                latents=state_means,
                extras={"obs_vars": obs_vars},
            )
        loss = self.core(x)
        return ModelOutput(extras={"loss": loss})

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        total = output.extras.get("loss")
        if total is None:
            x = observations_from_batch(batch).to(
                device=self.device,
                dtype=self.core.obs_noise_values.dtype,
            )
            total = self.core(x)
        return LossOutput(
            total=total,
            named_terms={"cassm_elbo": total},
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            from nlb2.metrics import NLBCoSmoothingAdapter

            return NLBCoSmoothingAdapter(
                feature_source=self.nlb_feature_source,
                decoder=self.nlb_decoder,
                ridge_alpha=self.nlb_ridge_alpha,
            )
        return None

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("CASSM expects input shape (batch, time, neurons).")
        self._sync_core_device(self.device)
        x = x.to(device=self.device, dtype=self.core.obs_noise_values.dtype)
        state_means, _ = self.core.filter(x, return_type="prediction")
        return state_means[..., 0::2].clamp_min(0.0)

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._sync_core_device(self.device)
        return module

    def _sync_core_device(self, device: torch.device) -> None:
        """Synchronize CASSM runtime-only tensors with module device and dtype."""

        self.core.device = device
        if hasattr(self.core, "dt"):
            self.core.dt = self.core.dt.to(device)
        if hasattr(self.core, "projection_indices"):
            self.core.projection_indices = self.core.projection_indices.to(device)
        if hasattr(self.core, "observation_matrix"):
            from linear_operator.operators import (
                IdentityLinearOperator,
                KroneckerProductLinearOperator,
            )

            dtype = self.core.obs_noise_values.dtype
            self.core.observation_matrix = KroneckerProductLinearOperator(
                IdentityLinearOperator(self.core.dim, device=device, dtype=dtype),
                torch.tensor([[1.0, 0.0]], device=device, dtype=dtype),
            )
