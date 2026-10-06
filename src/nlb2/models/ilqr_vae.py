"""Self-contained iLQR-VAE with posterior-control inference and ELBO training.

Includes the PyTorch solver, parameter initialization, and the small OCaml
Marshal reader needed to load optional tutorial checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import struct
from typing import Any, Literal, Optional

import numpy as np
from pydantic import Field, model_validator
from scipy.special import gammaln
import torch
from torch import Tensor

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.metrics import EvaluationAdapter, SyntheticEvaluationAdapter, poisson_negative_log_likelihood
from nlb2.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class ILQRVAEConfig(BaseModelConfig):
    """Config for the PyTorch iLQR-VAE adapter.

    The default ELBO objective trains a randomly initialized model. Loading a
    fixed tutorial checkpoint is opt-in through `initialization="pretrained"`,
    an explicit `params_path`, `objective="posterior_control"`, and
    `optimization.name="inference_only"`.

    `latent_dim` is the recurrent latent state dimension and `input_dim` is the
    dimensionality of the inferred control input. The current trainable NLB2
    path uses the translated Student prior, Mini-GRU-IO dynamics, Poisson
    likelihood, and shared Kronecker posterior covariance. For co-smoothing
    datasets, `held_in_neurons` selects the neurons used by the inner posterior
    solve, while `output_neuron_start` and `output_neurons` select the decoded
    prediction slice returned to the benchmark metrics. `build_from_data`
    derives those slices and `dt` from the dataset; explicit inconsistent
    values are rejected. Direct `build` uses unit-width bins if `dt` is omitted.
    """

    name: Literal["ilqr_vae"] = "ilqr_vae"
    objective: Literal["posterior_control", "ilqr_vae_elbo"] = "ilqr_vae_elbo"
    params_path: Optional[str] = None
    initialization: Literal["pretrained", "random", "checkpoint_transfer"] = "random"
    template_params_path: Optional[str] = None
    random_init_profile: Literal["default", "tutorial_mc_maze"] = "default"
    readout_bias_initialization: Literal["none", "empirical_rates"] = "none"
    empirical_rate_floor_hz: float = Field(default=1.0e-3, gt=0.0, allow_inf_nan=False)
    latent_dim: int = Field(default=20, gt=0)
    input_dim: int = Field(default=5, gt=0)
    init_seed: int = 0
    solver: Literal["ilqr", "lbfgs", "adam"] = "ilqr"
    max_iter: int = Field(default=5, ge=0)
    lr: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    control_hessian_mode: Literal["true", "fisher", "clamped"] = "true"
    ilqr_failure_fallback: Literal["none", "adam", "lbfgs"] = "adam"
    ilqr_fallback_max_iter: int = Field(default=25, ge=0)
    ilqr_fallback_lr: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    differentiate_controls: bool = True
    trainable_parameters: bool = True
    n_posterior_samples: int = Field(default=1, gt=0)
    include_elbo_constants: bool = True
    dynamics_regularizer: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    held_in_neurons: Optional[int] = Field(default=None, gt=0)
    output_neuron_start: Optional[int] = Field(default=None, ge=0)
    output_neurons: Optional[int] = Field(default=None, gt=0)
    rate_mode: Literal["likelihood", "pre_sample"] = "likelihood"
    dt: Optional[float] = Field(default=None, gt=0.0, allow_inf_nan=False)
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(name="gradient", lr=1.0e-3)
    )

    @model_validator(mode="after")
    def validate_training_config(self) -> "ILQRVAEConfig":
        if self.latent_dim % self.input_dim != 0:
            raise ValueError("latent_dim must be divisible by input_dim.")
        if self.objective == "ilqr_vae_elbo" and self.trainable_parameters:
            if not self.include_elbo_constants:
                raise ValueError(
                    "ELBO training requires include_elbo_constants=True: "
                    "the prior normalization depends on learned parameters."
                )
            if self.differentiate_controls and (
                self.solver == "lbfgs"
                or (self.solver == "ilqr" and self.ilqr_failure_fallback == "lbfgs")
            ):
                raise ValueError("Differentiable ELBO training does not support LBFGS inference.")
        return self

    def build(self, n_neurons: int, n_time: int) -> "ILQRVAE":
        model_neurons = n_neurons
        if (
            self.initialization in {"random", "checkpoint_transfer"}
            and self.output_neuron_start is not None
        ):
            output_stop = self.output_neuron_start + (self.output_neurons or 0)
            model_neurons = max(model_neurons, output_stop)
        return ILQRVAE(
            n_neurons=model_neurons,
            n_time=n_time,
            params_path=self.params_path,
            initialization=self.initialization,
            template_params_path=self.template_params_path,
            random_init_profile=self.random_init_profile,
            latent_dim=self.latent_dim,
            input_dim=self.input_dim,
            init_seed=self.init_seed,
            solver=self.solver,
            max_iter=self.max_iter,
            lr=self.lr,
            control_hessian_mode=self.control_hessian_mode,
            ilqr_failure_fallback=self.ilqr_failure_fallback,
            ilqr_fallback_max_iter=self.ilqr_fallback_max_iter,
            ilqr_fallback_lr=self.ilqr_fallback_lr,
            differentiate_controls=self.differentiate_controls,
            trainable_parameters=self.trainable_parameters,
            n_posterior_samples=self.n_posterior_samples,
            include_elbo_constants=self.include_elbo_constants,
            dynamics_regularizer=self.dynamics_regularizer,
            held_in_neurons=self.held_in_neurons,
            output_neuron_start=self.output_neuron_start,
            output_neurons=self.output_neurons,
            rate_mode=self.rate_mode,
            dt=1.0 if self.dt is None else self.dt,
            objective=self.objective,
        )

    def build_from_data(self, data: Any) -> "ILQRVAE":
        train_dataset = getattr(data, "train_dataset", None)
        dataset = getattr(train_dataset, "dataset", train_dataset)
        updates: dict[str, Any] = {}
        heldin = getattr(dataset, "heldin_spikes", None)
        heldout = getattr(dataset, "raw_spikes", None)
        if isinstance(heldin, Tensor) and isinstance(heldout, Tensor):
            dimensions = {
                "held_in_neurons": int(heldin.shape[-1]),
                "output_neuron_start": int(heldin.shape[-1]),
                "output_neurons": int(heldout.shape[-1]),
            }
            for name, value in dimensions.items():
                configured = getattr(self, name)
                if configured is not None and configured != value:
                    raise ValueError(f"{name}={configured} disagrees with dataset value {value}")
            updates.update(dimensions)
        dataset_config = getattr(data, "config", None)
        arrays = getattr(dataset, "arrays", None)
        data_dt = getattr(arrays, "dt", None)
        if data_dt is None:
            data_dt = getattr(dataset_config, "spike_bin_size", None)
        if data_dt is None:
            data_dt = getattr(dataset_config, "bin_size", None)
        if data_dt is not None:
            if self.dt is not None and abs(self.dt - float(data_dt)) > 1.0e-8:
                raise ValueError(f"model dt={self.dt} disagrees with dataset bin size {data_dt}")
            updates["dt"] = float(data_dt)
        model = self.model_copy(update=updates).build(
            n_neurons=data.n_neurons, n_time=data.n_time
        )
        if self.readout_bias_initialization == "empirical_rates":
            full_spikes = _full_spikes_from_dataset(train_dataset)
            if full_spikes is None:
                raise ValueError(
                    "readout_bias_initialization='empirical_rates' requires a train dataset "
                    "with held-in spikes and held-out/raw spikes."
                )
            model.initialize_readout_bias_from_counts(
                full_spikes,
                rate_floor_hz=self.empirical_rate_floor_hz,
            )
        return model


class ILQRVAE(BaseDynamicsModel):
    """Optimization-based iLQR-VAE with posterior-control inference and ELBO training.

    ## Method

    iLQR-VAE is a latent dynamical model with no amortized recognition network
    for the posterior mean. For each trial and current generative parameter
    setting, the model solves an inner optimal-control problem over a sequence
    of latent inputs `u`. Those inputs drive a recurrent dynamical system to
    produce latent states `z`, and a likelihood readout decodes observations
    from `z`.

    The NLB2 implementation ports the tutorial Student input prior,
    Mini-GRU-IO dynamics, Poisson spike likelihood, and iLQR posterior-control
    solver into PyTorch. The posterior covariance is shared across trials as a
    Kronecker product of learned time and input-space factors, matching the
    structure used by the original tutorial code.

    ## Inference-only checkpoint mode

    With `objective="posterior_control"` and `initialization="pretrained"`, the
    model loads the supplied `params_path` and uses iLQR only to infer posterior
    controls for each evaluation trial. It can register checkpoint tensors as
    buffers (`trainable_parameters=false`) or as `nn.Parameter`s
    (`trainable_parameters=true`) for regression checks; with zero training
    epochs both paths are numerically identical. The corrected solver includes
    the terminal observation likelihood and is not an exact reproduction of
    historical predictions made before that correction.

    For NLB-style co-smoothing, the inner solve can be restricted to held-in
    neurons by setting `held_in_neurons`, and the returned rates can be sliced to
    held-out neurons with `output_neuron_start` and `output_neurons`. Returned
    rates are expected spike counts per bin, not Hz, so they can be consumed
    directly by NLB2/NLB bits-per-spike metrics.

    ## ELBO training mode

    With `objective="ilqr_vae_elbo"` and a gradient optimizer, training follows
    the original iLQR-VAE outer objective:

    ```text
    ELBO = H[q(u | o)] + E_q[log p(u) + log p(o | z(u))]
    loss = -ELBO / num_observations + regularizer
    ```

    The inner iLQR solve provides the posterior mean controls. By default the
    outer backward pass differentiates through the executed solver updates as
    well as the sampled ELBO, including the Student prior, dynamics, likelihood,
    and shared posterior covariance. This unrolled finite-iteration derivative
    differs from the upstream implicit adjoint at an optimum. Adam fallback
    also preserves its control gradient; a detached-control approximation is
    available only by explicitly setting `differentiate_controls=false`.
    Bounded parameters such as prior
    scales, degrees of freedom, gains, and covariance diagonals are projected
    back into their valid domains after optimizer steps.

    ## Current scope

    The trainable path is designed for spike-count datasets and currently uses
    the Poisson/Mini-GRU-IO variant that was translated for MC_Maze. The
    original repository's standalone Lorenz example uses MGU2 dynamics with a
    3D Gaussian observation model; exact parity with that example would require
    adding that dynamics/likelihood variant as a separate model option. The
    provided `configs/experiment/synthetic/lorenz/ilqr_vae/ilqr_vae_lorenz_100.yaml`
    trains the Poisson/Mini-GRU-IO variant on the NLB2 Lorenz-100 spike
    population for comparison with LFADS and NDT.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        params_path: Optional[str] = None,
        initialization: str = "random",
        template_params_path: Optional[str] = None,
        random_init_profile: str = "default",
        latent_dim: int = 20,
        input_dim: int = 5,
        init_seed: int = 0,
        solver: str = "ilqr",
        max_iter: int = 5,
        lr: Optional[float] = None,
        control_hessian_mode: str = "true",
        ilqr_failure_fallback: str = "adam",
        ilqr_fallback_max_iter: int = 25,
        ilqr_fallback_lr: Optional[float] = None,
        differentiate_controls: bool = True,
        trainable_parameters: bool = True,
        n_posterior_samples: int = 1,
        include_elbo_constants: bool = True,
        dynamics_regularizer: float = 0.0,
        held_in_neurons: Optional[int] = None,
        output_neuron_start: Optional[int] = None,
        output_neurons: Optional[int] = None,
        rate_mode: str = "likelihood",
        dt: float = 1.0,
        objective: str = "ilqr_vae_elbo",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.params_path = None if params_path is None else str(params_path)
        self.initialization = initialization
        self.template_params_path = (
            None if template_params_path is None else str(template_params_path)
        )
        self.random_init_profile = random_init_profile
        self.latent_dim = int(latent_dim)
        self.input_dim = int(input_dim)
        self.init_seed = int(init_seed)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.lr = lr
        self.control_hessian_mode = str(control_hessian_mode)
        self.ilqr_failure_fallback = str(ilqr_failure_fallback)
        self.ilqr_fallback_max_iter = int(ilqr_fallback_max_iter)
        self.ilqr_fallback_lr = ilqr_fallback_lr
        self.differentiate_controls = bool(differentiate_controls)
        self.trainable_parameters = bool(trainable_parameters)
        self.n_posterior_samples = int(n_posterior_samples)
        self.include_elbo_constants = bool(include_elbo_constants)
        self.dynamics_regularizer = float(dynamics_regularizer)
        self.held_in_neurons = held_in_neurons
        self.output_neuron_start = output_neuron_start
        self.output_neurons = output_neurons
        self.rate_mode = rate_mode
        self.dt = float(dt)
        self.objective = objective
        if self.n_neurons < 1 or self.n_time < 1:
            raise ValueError("n_neurons and n_time must be positive.")
        if not math.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        if self.n_posterior_samples < 1:
            raise ValueError("n_posterior_samples must be positive.")
        if self.objective == "ilqr_vae_elbo" and self.trainable_parameters and not self.include_elbo_constants:
            raise ValueError("ELBO training requires parameter-dependent prior normalization constants.")

        if initialization == "pretrained":
            if params_path is None:
                raise ValueError("params_path is required for pretrained iLQR-VAE initialization.")
            params = load_tutorial_params(Path(params_path))
        elif initialization in {"random", "checkpoint_transfer"}:
            params = make_random_params(
                latent_dim=self.latent_dim,
                input_dim=self.input_dim,
                n_neurons=self.n_neurons,
                n_time=self.n_time,
                seed=self.init_seed,
                **_random_init_profile_kwargs(random_init_profile),
            )
            if initialization == "checkpoint_transfer":
                if template_params_path is None:
                    raise ValueError(
                        "template_params_path is required for checkpoint_transfer initialization."
                    )
                template = load_tutorial_params(Path(template_params_path))
                params = _transfer_checkpoint_parameters(params, template)
        else:
            raise ValueError(f"unknown iLQR-VAE initialization {initialization!r}")
        self.core = TutorialILQRVAE(
            params,
            dt=self.dt,
            trainable=self.trainable_parameters,
            control_hessian_mode=self.control_hessian_mode,
        )

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError(f"expected input shape batch x time x neurons, got {tuple(x.shape)}")

        held_in = self._observed_neurons_for_input(x)
        output_start = self.output_neuron_start if self.output_neuron_start is not None else 0
        output_stop = (
            self.core.n_neurons
            if self.output_neurons is None
            else output_start + int(self.output_neurons)
        )
        output_dtype = x.dtype if x.is_floating_point() else self.core.c.dtype

        rates = []
        full_rates = []
        latents = []
        controls = []
        eval_counts = []
        fallback_counts = []
        objectives = []
        for trial in x:
            differentiable = (
                self.objective == "ilqr_vae_elbo"
                and self.differentiate_controls
                and torch.is_grad_enabled()
            )
            result, used_fallback = self._infer_controls_with_fallback(
                trial.detach(),
                held_in,
                differentiable=differentiable,
            )
            observed_latents = self.core.observation_latents(
                result.latents,
                n_observed_steps=int(trial.shape[0]),
            )
            rates_hz = self.core.firing_rates(observed_latents, mode=self.rate_mode)
            full_counts = (self.dt * rates_hz).to(output_dtype)
            full_rates.append(rates_hz.to(output_dtype))
            rates.append(full_counts[:, output_start:output_stop])
            latents.append(observed_latents.to(output_dtype))
            controls.append(result.controls)
            eval_counts.append(len(result.loss_history))
            fallback_counts.append(1.0 if used_fallback else 0.0)
            objectives.append(result.loss_history[-1] if result.loss_history else float("nan"))

        return ModelOutput(
            rates=torch.stack(rates, dim=0),
            latents=torch.stack(latents, dim=0),
            full_rates_unit="hz",
            extras={
                "controls": torch.stack(controls, dim=0),
                "full_rates": torch.stack(full_rates, dim=0),
                "ilqr_evaluations": torch.tensor(eval_counts, dtype=torch.float32, device=x.device),
                "inference_fallbacks": torch.tensor(
                    fallback_counts,
                    dtype=torch.float32,
                    device=x.device,
                ),
                "posterior_objective": torch.tensor(objectives, dtype=torch.float32, device=x.device),
            },
        )

    def _infer_controls_with_fallback(
        self,
        trial: Tensor,
        held_in: int,
        *,
        differentiable: bool = False,
    ):
        try:
            return (
                self.core.infer_controls(
                    trial,
                    held_in_neurons=held_in,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    lr=self.lr,
                    differentiable=differentiable,
                ),
                False,
            )
        except RuntimeError as exc:
            message = str(exc)
            is_ilqr_failure = (
                "iLQR backward pass" in message
                or "iLQR line search" in message
            )
            if (
                self.solver != "ilqr"
                or self.ilqr_failure_fallback == "none"
                or not is_ilqr_failure
            ):
                raise
            return (
                self.core.infer_controls(
                    trial,
                    held_in_neurons=held_in,
                    solver=self.ilqr_failure_fallback,
                    max_iter=self.ilqr_fallback_max_iter,
                    lr=self.ilqr_fallback_lr,
                    differentiable=differentiable,
                ),
                True,
            )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        x = observations_from_batch(batch)
        if self.objective == "ilqr_vae_elbo":
            return self._elbo_loss(self._training_observations(batch, x), output)

        target = batch.get("raw_spikes", x) if isinstance(batch, dict) else x
        if output.rates is None:
            raise RuntimeError("ILQRVAE.forward did not return rates.")
        total = poisson_negative_log_likelihood(output.rates, target).mean()
        return LossOutput(
            total=total,
            named_terms={
                "poisson_nll": total,
                "mean_ilqr_evaluations": output.extras["ilqr_evaluations"].mean(),
                "mean_inference_fallbacks": output.extras["inference_fallbacks"].mean(),
                "mean_posterior_objective": output.extras["posterior_objective"].mean(),
            },
            objective=self.objective,
        )

    def _elbo_loss(self, x: Tensor, output: ModelOutput) -> LossOutput:
        controls = output.extras.get("controls")
        if not isinstance(controls, Tensor):
            raise RuntimeError("ILQR-VAE ELBO requires posterior controls from forward().")
        held_in = self._observed_neurons_for_input(x)
        losses = []
        elbos = []
        for trial, trial_controls in zip(x, controls):
            trial = trial.to(dtype=self.core.c.dtype, device=self.core.c.device)
            if not self.differentiate_controls:
                trial_controls = trial_controls.detach()
            trial_controls = trial_controls.to(dtype=self.core.c.dtype, device=self.core.c.device)
            elbo = self.core.elbo_from_controls(
                trial_controls,
                trial,
                held_in_neurons=held_in,
                n_posterior_samples=self.n_posterior_samples,
                include_constants=self.include_elbo_constants,
            )
            normalizer = max(int(trial[:, :held_in].numel()), 1)
            elbos.append(elbo)
            losses.append(-elbo / float(normalizer))

        negative_elbo = torch.stack(losses).mean()
        regularizer = self._dynamics_regularizer()
        total = negative_elbo + regularizer
        return LossOutput(
            total=total,
            named_terms={
                "negative_elbo": negative_elbo,
                "elbo": torch.stack(elbos).mean(),
                "dynamics_regularizer": regularizer,
                "mean_ilqr_evaluations": output.extras["ilqr_evaluations"].mean(),
                "mean_inference_fallbacks": output.extras["inference_fallbacks"].mean(),
                "mean_posterior_objective": output.extras["posterior_objective"].mean(),
            },
            objective=self.objective,
        )

    def loss_forward_observations(
        self,
        batch: Tensor | dict[str, Tensor],
        x: Tensor,
    ) -> Tensor:
        if self.objective != "ilqr_vae_elbo":
            return x
        return self._training_observations(batch, x)

    def evaluation_adapter(self, task: str) -> EvaluationAdapter | None:
        if task == "synthetic":
            return SyntheticEvaluationAdapter(use_raw_spikes=True)
        return None

    def _training_observations(self, batch: Tensor | dict[str, Tensor], x: Tensor) -> Tensor:
        if not isinstance(batch, dict):
            return x
        reconstruction = batch.get("reconstruction_spikes")
        if reconstruction is not None:
            if int(reconstruction.shape[-1]) != self.core.n_neurons:
                raise ValueError("reconstruction_spikes channels disagree with the model readout")
            return reconstruction.to(device=x.device, dtype=x.dtype)
        if int(x.shape[-1]) == self.core.n_neurons:
            raw = batch.get("raw_spikes")
            return raw if raw is not None and raw.shape == x.shape else x
        x = batch.get("heldin_spikes", x)
        heldout = batch.get("heldout_spikes")
        if heldout is None:
            heldout = batch.get("raw_spikes")
        if heldout is None:
            return x
        if x.shape[:-1] != heldout.shape[:-1]:
            return x
        return torch.cat([x, heldout.to(device=x.device, dtype=x.dtype)], dim=-1)

    def _observed_neurons_for_input(self, x: Tensor) -> int:
        configured = self.held_in_neurons or int(x.shape[-1])
        if self.training and self.objective == "ilqr_vae_elbo" and int(x.shape[-1]) > configured:
            return int(x.shape[-1])
        return configured

    def initialize_readout_bias_from_counts(
        self,
        spikes: Tensor,
        *,
        rate_floor_hz: float = 1.0e-3,
    ) -> None:
        if spikes.ndim != 3:
            raise ValueError(
                f"expected spikes with shape trials x time x neurons, got {tuple(spikes.shape)}"
            )
        if int(spikes.shape[-1]) != self.core.n_neurons:
            raise ValueError(
                f"empirical readout initialization expected {self.core.n_neurons} neurons, "
                f"got {int(spikes.shape[-1])}."
            )
        with torch.no_grad():
            mean_counts = spikes.to(
                dtype=self.core.bias.dtype,
                device=self.core.bias.device,
            ).mean(dim=(0, 1))
            gain = self.core._positive(self.core.gain.reshape(-1))
            rate_hz = torch.clamp(mean_counts / self.dt, min=float(rate_floor_hz))
            bias = torch.log(torch.clamp(rate_hz / gain - 1.0e-3, min=float(rate_floor_hz)))
            self.core.bias.copy_(bias.reshape_as(self.core.bias))

    def _dynamics_regularizer(self) -> Tensor:
        if self.dynamics_regularizer <= 0.0:
            return self.core.c.new_zeros(())
        scale = self.dynamics_regularizer / float(self.core.n_latent * self.core.n_latent)
        return scale * (torch.sum(self.core.uh**2) + torch.sum(self.core.uf**2))

    def project_parameters(self) -> None:
        self.core.project_parameters()


def _random_init_profile_kwargs(profile: str) -> dict[str, float]:
    if profile == "default":
        return {}
    if profile == "tutorial_mc_maze":
        return {
            "spatial_std": 1.0,
            "nu": 20.0,
            "first_step_std": 1.0,
            "uf_sigma": 0.0035,
            "dynamics_sigma": 0.01,
            "bh_sigma": 0.01,
            "input_sigma": 1.0 / (15.0**0.5),
            "readout_sigma": 0.01,
            "bias_mean": 1.0,
            "bias_sigma": 0.01,
            "gain_mean": 1.0,
            "gain_sigma": 0.01,
            "covariance_jitter_sigma": 0.01,
        }
    raise ValueError(f"unknown iLQR-VAE random_init_profile {profile!r}")


def _transfer_checkpoint_parameters(params: TutorialParams, template: TutorialParams) -> TutorialParams:
    """Copy shape-compatible non-readout parameters from a trained checkpoint."""

    replacements: dict[str, Any] = {}
    for name in (
        "spatial_stds",
        "nu",
        "first_step",
        "uf",
        "wh",
        "uh",
        "bh",
        "b",
        "space_cov_d",
        "space_cov_t",
        "time_cov_d",
        "time_cov_t",
    ):
        source = getattr(template, name)
        target = getattr(params, name)
        if name == "nu" or tuple(source.shape) == tuple(target.shape):
            replacements[name] = source.copy() if hasattr(source, "copy") else source
    return replace(params, **replacements)


def _full_spikes_from_dataset(dataset: Any) -> Tensor | None:
    if dataset is None:
        return None
    base = getattr(dataset, "dataset", dataset)
    heldin = getattr(base, "heldin_spikes", None)
    heldout = getattr(base, "raw_spikes", None)
    if heldin is None:
        return heldout if heldout is not None else getattr(base, "spikes", None)
    if heldout is None:
        return heldin
    if heldin.shape[:-1] != heldout.shape[:-1]:
        return heldin
    return torch.cat([heldin, heldout.to(device=heldin.device, dtype=heldin.dtype)], dim=-1)


# OCaml Marshal reader for optional tutorial checkpoints

SMALL_MAGIC = 0x8495A6BE
BIGARRAY_IDENT = "_bigarr02"
BIGARRAY_FLOAT64 = 1


@dataclass(frozen=True)
class Block:
    tag: int
    fields: tuple[Any, ...]


@dataclass(frozen=True)
class Bigarray:
    flags: int
    shape: tuple[int, ...]
    data: np.ndarray


class MarshalReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    @property
    def pos(self) -> int:
        return self._pos

    def read_value(self) -> Any:
        code = self._read_u8()

        if code >= 0x80:
            tag = code & 0x0F
            size = (code >> 4) & 0x07
            return self._read_block(tag, size)
        if code >= 0x40:
            return code - 0x40
        if code >= 0x20:
            size = code - 0x20
            return self._read(size).decode("latin1")

        if code == 0x00:
            return struct.unpack("b", self._read(1))[0]
        if code == 0x01:
            return self._read_i16()
        if code == 0x02:
            return self._read_i32()
        if code == 0x03:
            return self._read_i64()
        if code == 0x08:
            header = self._read_u32()
            return self._read_block(header & 0xFF, header >> 10)
        if code == 0x13:
            header = self._read_u64()
            return self._read_block(header & 0xFF, header >> 10)
        if code == 0x09:
            return self._read(self._read_u8()).decode("latin1")
        if code == 0x0A:
            return self._read(self._read_u32()).decode("latin1")
        if code == 0x15:
            return self._read(self._read_u64()).decode("latin1")
        if code == 0x0B:
            return struct.unpack(">d", self._read(8))[0]
        if code == 0x0C:
            return struct.unpack("<d", self._read(8))[0]
        if code == 0x0D:
            return self._read_float_array(self._read_u8(), ">f8")
        if code == 0x0E:
            return self._read_float_array(self._read_u8(), "<f8")
        if code == 0x0F:
            return self._read_float_array(self._read_u32(), ">f8")
        if code == 0x07:
            return self._read_float_array(self._read_u32(), "<f8")
        if code == 0x16:
            return self._read_float_array(self._read_u64(), ">f8")
        if code == 0x17:
            return self._read_float_array(self._read_u64(), "<f8")
        if code == 0x18:
            ident = self._read_cstring()
            self._read_u32()
            self._read_u64()
            return self._read_custom(ident)
        if code == 0x19:
            return self._read_custom(self._read_cstring())

        if code in {0x04, 0x05, 0x06, 0x14}:
            raise NotImplementedError("shared Marshal references are not expected")
        raise ValueError(f"unsupported OCaml Marshal code 0x{code:02x} at byte {self._pos - 1}")

    def _read_block(self, tag: int, size: int) -> Block:
        return Block(tag=tag, fields=tuple(self.read_value() for _ in range(size)))

    def _read_custom(self, ident: str) -> Bigarray:
        if ident != BIGARRAY_IDENT:
            raise ValueError(f"unsupported OCaml custom block {ident!r}")

        n_dims = self._read_u32()
        flags = self._read_u32()
        shape = []
        for _ in range(n_dims):
            size = self._read_u16()
            if size == 0xFFFF:
                size = self._read_u64()
            shape.append(size)

        n_items = math.prod(shape)
        kind = flags & 0xFF
        if kind != BIGARRAY_FLOAT64:
            raise ValueError(f"unsupported Bigarray kind {kind}; expected float64")

        data = self._read_float_array(n_items, ">f8").reshape(tuple(shape))
        return Bigarray(flags=flags, shape=tuple(shape), data=data)

    def _read_float_array(self, n_items: int, dtype: str) -> np.ndarray:
        return np.frombuffer(self._read(8 * n_items), dtype=dtype).astype(np.float64)

    def _read_cstring(self) -> str:
        end = self._data.index(0, self._pos)
        value = self._data[self._pos:end].decode()
        self._pos = end + 1
        return value

    def _read(self, size: int) -> bytes:
        if self._pos + size > len(self._data):
            raise EOFError("unexpected end of OCaml Marshal data")
        out = self._data[self._pos : self._pos + size]
        self._pos += size
        return out

    def _read_u8(self) -> int:
        return self._read(1)[0]

    def _read_u16(self) -> int:
        return struct.unpack(">H", self._read(2))[0]

    def _read_i16(self) -> int:
        return struct.unpack(">h", self._read(2))[0]

    def _read_u32(self) -> int:
        return struct.unpack(">I", self._read(4))[0]

    def _read_i32(self) -> int:
        return struct.unpack(">i", self._read(4))[0]

    def _read_u64(self) -> int:
        return struct.unpack(">Q", self._read(8))[0]

    def _read_i64(self) -> int:
        return struct.unpack(">q", self._read(8))[0]


def load_marshal(path: str | Path) -> Any:
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError("file is too short to be an OCaml Marshal stream")

    magic, data_len, _object_count, _size_32, _size_64 = struct.unpack(">IIIII", raw[:20])
    if magic != SMALL_MAGIC:
        raise ValueError(f"unsupported OCaml Marshal magic 0x{magic:08x}")

    payload = raw[20 : 20 + data_len]
    reader = MarshalReader(payload)
    value = reader.read_value()
    if reader.pos != data_len:
        raise ValueError(f"did not consume full Marshal payload: {reader.pos} != {data_len}")
    return value


# Parameter loading and random initialization

@dataclass(frozen=True)
class TutorialParams:
    spatial_stds: np.ndarray
    nu: float
    first_step: np.ndarray
    uf: np.ndarray
    wh: np.ndarray
    uh: np.ndarray
    bh: np.ndarray
    b: np.ndarray
    c: np.ndarray
    bias: np.ndarray
    gain: np.ndarray
    space_cov_d: np.ndarray
    space_cov_t: np.ndarray
    time_cov_d: np.ndarray
    time_cov_t: np.ndarray


def load_tutorial_params(path: str | Path) -> TutorialParams:
    """Load ``final_params.bin`` or ``progress_*.params.bin``.

    The expected parameter layout is the model defined in ``demo.ipynb``:
    Student prior, Mini_GRU_IO dynamics, Poisson likelihood, and iLQR
    recognition covariance.
    """

    root = _block(load_marshal(path), tag=0, size=2, label="model")
    generative = _block(root.fields[0], tag=0, label="generative")
    recognition = _block(root.fields[1], tag=0, label="recognition")
    if len(generative.fields) not in {3, 4}:
        raise ValueError(f"generative: expected 3 or 4 fields, got {len(generative.fields)}")

    prior = _block(generative.fields[0], tag=0, size=3, label="prior")
    dynamics = _block(generative.fields[1], tag=0, size=5, label="dynamics")
    likelihood = _block(generative.fields[2], tag=0, size=4, label="likelihood")

    if len(recognition.fields) == 2:
        space_cov = _block(recognition.fields[0], tag=0, size=2, label="space_cov")
        time_cov = _block(recognition.fields[1], tag=0, size=2, label="time_cov")
    elif len(recognition.fields) == 4:
        space_cov = _block(recognition.fields[1], tag=0, size=2, label="space_cov")
        time_cov = _block(recognition.fields[2], tag=0, size=2, label="time_cov")
    else:
        raise ValueError(f"recognition: expected 2 or 4 fields, got {len(recognition.fields)}")

    c_mask = likelihood.fields[1]
    if c_mask != 0:
        raise ValueError("masked likelihood readouts are not supported")

    b = _option_param(dynamics.fields[4], "dynamics.b")
    if b is None:
        raise ValueError("tutorial Mini_GRU_IO parameters should include dynamics.b")

    return TutorialParams(
        spatial_stds=_param_array(prior.fields[0], "prior.spatial_stds"),
        nu=float(_param_array(prior.fields[1], "prior.nu")),
        first_step=_param_array(prior.fields[2], "prior.first_step"),
        uf=_param_array(dynamics.fields[0], "dynamics.uf"),
        wh=_param_array(dynamics.fields[1], "dynamics.wh"),
        uh=_param_array(dynamics.fields[2], "dynamics.uh"),
        bh=_param_array(dynamics.fields[3], "dynamics.bh"),
        b=b,
        c=_param_array(likelihood.fields[0], "likelihood.c"),
        bias=_param_array(likelihood.fields[2], "likelihood.bias"),
        gain=_param_array(likelihood.fields[3], "likelihood.gain"),
        space_cov_d=_param_array(space_cov.fields[0], "recognition.space_cov.d"),
        space_cov_t=_param_array(space_cov.fields[1], "recognition.space_cov.t"),
        time_cov_d=_param_array(time_cov.fields[0], "recognition.time_cov.d"),
        time_cov_t=_param_array(time_cov.fields[1], "recognition.time_cov.t"),
    )


def make_random_params(
    *,
    latent_dim: int,
    input_dim: int,
    n_neurons: int,
    n_time: int,
    seed: int = 0,
    spatial_std: float = 1.0,
    nu: float = 20.0,
    first_step_std: float | None = None,
    uf_sigma: float | None = None,
    dynamics_sigma: float | None = None,
    bh_sigma: float = 0.0,
    input_sigma: float | None = None,
    readout_sigma: float | None = None,
    bias_mean: float = 0.0,
    bias_sigma: float = 0.0,
    gain_mean: float = 1.0,
    gain_sigma: float = 0.0,
    covariance_jitter_sigma: float = 0.0,
) -> TutorialParams:
    """Initialize iLQR-VAE parameters for a new dataset.

    The defaults mirror the original Lorenz example's dimensions and Student
    prior initialization, while using the tutorial Poisson likelihood expected
    by the NLB2 spike-count datasets.
    """

    if latent_dim % input_dim != 0:
        raise ValueError("latent_dim must be divisible by input_dim.")
    rng = np.random.default_rng(seed)
    n_beg = latent_dim // input_dim
    n_controls = n_time + n_beg - 1
    first_step_std = spatial_std if first_step_std is None else first_step_std
    uf_sigma = 0.0 if uf_sigma is None else uf_sigma
    dyn_sigma = 0.1 / np.sqrt(float(latent_dim)) if dynamics_sigma is None else dynamics_sigma
    input_sigma = 1.0 / np.sqrt(float(input_dim)) if input_sigma is None else input_sigma
    readout_sigma = (
        1.0 / np.sqrt(float(latent_dim)) if readout_sigma is None else readout_sigma
    )
    return TutorialParams(
        spatial_stds=np.full((1, input_dim), spatial_std, dtype=np.float64),
        nu=float(nu),
        first_step=np.full((1, input_dim), first_step_std, dtype=np.float64),
        uf=rng.normal(scale=uf_sigma, size=(latent_dim, latent_dim)).astype(np.float64),
        wh=rng.normal(scale=dyn_sigma, size=(latent_dim, latent_dim)).astype(np.float64),
        uh=rng.normal(scale=dyn_sigma, size=(latent_dim, latent_dim)).astype(np.float64),
        bh=rng.normal(scale=bh_sigma, size=(1, latent_dim)).astype(np.float64),
        b=rng.normal(scale=input_sigma, size=(input_dim, latent_dim)).astype(np.float64),
        c=rng.normal(scale=readout_sigma, size=(n_neurons, latent_dim)).astype(np.float64),
        bias=rng.normal(loc=bias_mean, scale=bias_sigma, size=(1, n_neurons)).astype(np.float64),
        gain=rng.normal(loc=gain_mean, scale=gain_sigma, size=(1, n_neurons)).astype(np.float64),
        space_cov_d=np.ones((input_dim, 1), dtype=np.float64),
        space_cov_t=rng.normal(
            scale=covariance_jitter_sigma,
            size=(input_dim, input_dim),
        ).astype(np.float64),
        time_cov_d=np.ones((n_controls, 1), dtype=np.float64),
        time_cov_t=rng.normal(
            scale=covariance_jitter_sigma,
            size=(n_controls, n_controls),
        ).astype(np.float64),
    )


def _param_array(value: Any, label: str) -> np.ndarray:
    """Extract an ``Owl_parameters.tag`` value."""

    block = _block(value, label=label)
    if block.tag in {0, 1}:  # Pinned | Learned
        _check_size(block, 1, label)
        return _ad_value(block.fields[0], label)
    if block.tag == 2:  # Learned_bounded
        _check_size(block, 3, label)
        return _ad_value(block.fields[0], label)
    raise ValueError(f"{label}: unsupported Owl_parameters tag {block.tag}")


def _option_param(value: Any, label: str) -> np.ndarray | None:
    if value == 0:
        return None
    some = _block(value, tag=0, size=1, label=label)
    return _param_array(some.fields[0], label)


def _ad_value(value: Any, label: str) -> np.ndarray:
    block = _block(value, label=label)
    if block.tag == 0:
        _check_size(block, 1, label)
        return np.asarray(block.fields[0], dtype=np.float64)
    if block.tag == 1:
        _check_size(block, 1, label)
        bigarray = block.fields[0]
        if not isinstance(bigarray, Bigarray):
            raise TypeError(f"{label}: expected Bigarray, got {type(bigarray).__name__}")
        return bigarray.data.copy()
    raise ValueError(f"{label}: unsupported Owl Algodiff value tag {block.tag}")


def _block(value: Any, *, tag: int | None = None, size: int | None = None, label: str) -> Block:
    if not isinstance(value, Block):
        raise TypeError(f"{label}: expected OCaml block, got {type(value).__name__}")
    if tag is not None and value.tag != tag:
        raise ValueError(f"{label}: expected tag {tag}, got {value.tag}")
    if size is not None:
        _check_size(value, size, label)
    return value


def _check_size(block: Block, size: int, label: str) -> None:
    if len(block.fields) != size:
        raise ValueError(f"{label}: expected {size} fields, got {len(block.fields)}")


# Generative model and posterior-control solver

Solver = Literal["adam", "lbfgs", "ilqr"]
RateMode = Literal["likelihood", "pre_sample"]
ControlHessianMode = Literal["true", "fisher", "clamped"]
_MAX_LOG_RATE = 12.0


@dataclass(frozen=True)
class InferenceResult:
    controls: torch.Tensor
    latents: torch.Tensor
    loss_history: tuple[float, ...]
    trace_evaluations: tuple[int, ...] = ()
    trace_losses: tuple[float, ...] = ()
    trace_controls: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class _TapeStep:
    x: torch.Tensor
    u: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    rlx: torch.Tensor
    rlu: torch.Tensor
    rlxx: torch.Tensor
    rluu: torch.Tensor
    rlux: torch.Tensor


class TutorialILQRVAE(torch.nn.Module):
    """The MCMaze tutorial model in PyTorch.

    This ports the Student input prior, ``Mini_GRU_IO`` dynamics, Poisson
    likelihood, and a structured iLQR posterior-control solver matching the
    tutorial recognition path. ``infer_controls`` also keeps Adam and LBFGS
    solvers for objective checks.
    """

    def __init__(
        self,
        params: TutorialParams,
        *,
        dt: float = 5e-3,
        trainable: bool = False,
        control_hessian_mode: ControlHessianMode = "true",
    ) -> None:
        super().__init__()
        self.dt = dt
        self.trainable = bool(trainable)
        self.control_hessian_mode = control_hessian_mode
        self.n_latent = params.wh.shape[0]
        self.n_input = params.b.shape[0]
        if self.n_latent % self.n_input != 0:
            raise ValueError("latent dimension must be divisible by input dimension")
        self.n_beg = self.n_latent // self.n_input

        self._register_model_tensor("spatial_stds", _tensor(params.spatial_stds.reshape(-1)))
        self._register_model_tensor("nu", torch.tensor(float(params.nu), dtype=torch.float64))
        self._register_model_tensor("first_step", _tensor(params.first_step.reshape(-1)))
        self._register_model_tensor("uf", _tensor(params.uf))
        self._register_model_tensor("wh", _tensor(params.wh))
        self._register_model_tensor("uh", _tensor(params.uh))
        self._register_model_tensor("bh", _tensor(params.bh))
        self._register_model_tensor("b", _tensor(params.b))
        self._register_model_tensor("c", _tensor(params.c))
        self._register_model_tensor("bias", _tensor(params.bias))
        self._register_model_tensor("gain", _tensor(params.gain))
        self._register_model_tensor("space_cov_d", _tensor(params.space_cov_d.reshape(-1)))
        self._register_model_tensor("space_cov_t", _tensor(params.space_cov_t))
        self._register_model_tensor("time_cov_d", _tensor(params.time_cov_d.reshape(-1)))
        self._register_model_tensor("time_cov_t", _tensor(params.time_cov_t))
        self.register_buffer("beg_bs", self._make_initial_condition_maps())

    @property
    def n_neurons(self) -> int:
        return self.c.shape[0]

    def _register_model_tensor(self, name: str, value: torch.Tensor) -> None:
        if self.trainable:
            self.register_parameter(name, torch.nn.Parameter(value.clone()))
        else:
            self.register_buffer(name, value)

    @staticmethod
    def _positive(value: torch.Tensor, *, lower: float = 1e-6) -> torch.Tensor:
        return value.clamp_min(lower)

    def _cov_chol(self, d: torch.Tensor, t: torch.Tensor, size: int) -> torch.Tensor:
        if d.shape[0] < size or t.shape[0] < size or t.shape[1] < size:
            raise ValueError(
                f"covariance parameters are too short for size {size}: "
                f"d={tuple(d.shape)}, t={tuple(t.shape)}"
            )
        diag = self._positive(d[:size])
        triangle = torch.triu(t[:size, :size], diagonal=1)
        return triangle + torch.diag(diag)

    def project_parameters(self) -> None:
        """Project bounded original parameters back into their valid domain."""

        with torch.no_grad():
            for name in (
                "spatial_stds",
                "nu",
                "first_step",
                "gain",
                "space_cov_d",
                "time_cov_d",
            ):
                value = getattr(self, name, None)
                if isinstance(value, torch.nn.Parameter):
                    lower = 2.0 + 1e-6 if name == "nu" else 1e-6
                    value.clamp_(min=lower)

    def infer_controls(
        self,
        spikes: np.ndarray | torch.Tensor,
        *,
        held_in_neurons: int | None = None,
        solver: Solver = "lbfgs",
        max_iter: int = 200,
        lr: float | None = None,
        differentiable: bool = False,
        include_constants: bool = False,
        trace_every: int | None = None,
    ) -> InferenceResult:
        """Infer posterior-mean controls for one trial.

        ``spikes`` must have shape ``time x neurons``. ``held_in_neurons`` can
        truncate the likelihood readout for flexible co-smoothing.
        """

        obs = torch.as_tensor(spikes, dtype=torch.float64, device=self.c.device)
        if obs.ndim != 2:
            raise ValueError(f"expected spikes with shape time x neurons, got {tuple(obs.shape)}")
        if held_in_neurons is None:
            held_in_neurons = obs.shape[1]
        if not 0 < held_in_neurons <= min(obs.shape[1], self.n_neurons):
            raise ValueError("held_in_neurons must select available observation/readout channels")
        obs = obs[:, :held_in_neurons]
        if max_iter < 0:
            raise ValueError("max_iter must be nonnegative")
        if differentiable and solver == "lbfgs":
            raise ValueError(
                "Differentiable controls require solver='ilqr' or 'adam'; "
                "torch.optim.LBFGS does not preserve the inference graph."
            )

        n_controls = obs.shape[0] + self.n_beg - 1
        controls = torch.zeros(
            n_controls,
            self.n_input,
            dtype=torch.float64,
            device=obs.device,
            requires_grad=True,
        )
        history: list[float] = []
        trace_evaluations: list[int] = []
        trace_losses: list[float] = []
        trace_controls: list[torch.Tensor] = []

        final_controls: torch.Tensor | None = None
        if solver == "lbfgs":
            with torch.enable_grad():
                self._infer_lbfgs(
                    controls,
                    obs,
                    held_in_neurons=held_in_neurons,
                    history=history,
                    max_iter=max_iter,
                    lr=1.0 if lr is None else lr,
                    include_constants=include_constants,
                    trace_every=trace_every,
                    trace_evaluations=trace_evaluations,
                    trace_losses=trace_losses,
                    trace_controls=trace_controls,
                )
        elif solver == "adam":
            with torch.enable_grad():
                final_controls = self._infer_adam(
                    controls,
                    obs,
                    held_in_neurons=held_in_neurons,
                    history=history,
                    max_iter=max_iter,
                    lr=0.03 if lr is None else lr,
                    include_constants=include_constants,
                    trace_every=trace_every,
                    trace_evaluations=trace_evaluations,
                    trace_losses=trace_losses,
                    trace_controls=trace_controls,
                    differentiable=differentiable,
                )
        elif solver == "ilqr":
            if differentiable:
                with torch.enable_grad():
                    controls.requires_grad_(False)
                    final_controls = self._infer_ilqr_unrolled(
                        controls,
                        obs,
                        held_in_neurons=held_in_neurons,
                        history=history,
                        max_iter=max_iter,
                        include_constants=include_constants,
                        trace_every=trace_every,
                        trace_evaluations=trace_evaluations,
                        trace_losses=trace_losses,
                        trace_controls=trace_controls,
                    )
            else:
                with torch.no_grad():
                    controls.requires_grad_(False)
                    self._infer_ilqr(
                        controls,
                        obs,
                        held_in_neurons=held_in_neurons,
                        history=history,
                        max_iter=max_iter,
                        include_constants=include_constants,
                        trace_every=trace_every,
                        trace_evaluations=trace_evaluations,
                        trace_losses=trace_losses,
                        trace_controls=trace_controls,
                    )
        else:
            raise ValueError(f"unknown solver {solver!r}")

        if final_controls is None:
            final_controls = controls.detach()
        if differentiable:
            latents = self.integrate(final_controls)
            with torch.no_grad():
                final_loss = float(
                    self.ilqr_objective(
                        final_controls,
                        obs,
                        held_in_neurons=held_in_neurons,
                        include_constants=include_constants,
                    )
                    .cpu()
                )
        else:
            with torch.no_grad():
                final_controls = final_controls.detach()
                latents = self.integrate(final_controls)
                if solver == "ilqr":
                    final_objective = self.ilqr_objective
                else:
                    final_objective = self.posterior_objective
                final_loss = float(
                    final_objective(
                        final_controls,
                        obs,
                        held_in_neurons=held_in_neurons,
                        include_constants=include_constants,
                    )
                    .detach()
                    .cpu()
                )
        if not trace_controls or not torch.equal(trace_controls[-1].to(final_controls.device), final_controls):
            trace_evaluations.append(len(history))
            trace_losses.append(final_loss)
            trace_controls.append(final_controls.detach().cpu())
        return InferenceResult(
            final_controls,
            latents,
            tuple(history),
            tuple(trace_evaluations),
            tuple(trace_losses),
            tuple(trace_controls),
        )

    def posterior_objective(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        include_constants: bool = False,
    ) -> torch.Tensor:
        latents = self.integrate(controls)
        observed_latents = self.observation_latents(latents, n_observed_steps=spikes.shape[0])
        return self.student_prior_nll(controls, include_constants=include_constants) + self.poisson_nll(
            observed_latents,
            spikes,
            held_in_neurons=held_in_neurons,
            include_constants=include_constants,
        )

    def ilqr_objective(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        include_constants: bool = False,
    ) -> torch.Tensor:
        """Control objective with observation costs before each transition.

        The last observed state follows the final transition, so its likelihood
        is a terminal cost. All solvers therefore use the same full likelihood.
        """

        return self.posterior_objective(
            controls,
            spikes,
            held_in_neurons=held_in_neurons,
            include_constants=include_constants,
        )

    def integrate(self, controls: torch.Tensor) -> torch.Tensor:
        """Propagate controls through Mini_GRU_IO dynamics."""

        if controls.ndim != 2 or controls.shape[1] != self.n_input:
            raise ValueError(
                f"expected controls with shape time x {self.n_input}, got {tuple(controls.shape)}"
            )

        x = torch.zeros(1, self.n_latent, dtype=controls.dtype, device=controls.device)
        latents = []
        for k in range(controls.shape[0]):
            u = controls[k : k + 1]
            x = self._dynamics_step(k, x, u)
            latents.append(x)
        return torch.cat(latents, dim=0)

    def integrate_samples(self, controls: torch.Tensor) -> torch.Tensor:
        """Propagate sampled controls with shape samples x time x inputs."""

        if controls.ndim != 3 or controls.shape[2] != self.n_input:
            raise ValueError(
                "expected sampled controls with shape "
                f"samples x time x {self.n_input}, got {tuple(controls.shape)}"
            )

        x = torch.zeros(
            controls.shape[0],
            self.n_latent,
            dtype=controls.dtype,
            device=controls.device,
        )
        latents = []
        for k in range(controls.shape[1]):
            u = controls[:, k, :]
            x = self._dynamics_step(k, x, u)
            latents.append(x.unsqueeze(1))
        return torch.cat(latents, dim=1)

    def observation_latents(self, latents: torch.Tensor, *, n_observed_steps: int | None = None) -> torch.Tensor:
        observed = latents[self.n_beg - 1 :]
        if n_observed_steps is not None:
            observed = observed[:n_observed_steps]
        return observed

    def observation_latents_samples(
        self,
        latents: torch.Tensor,
        *,
        n_observed_steps: int | None = None,
    ) -> torch.Tensor:
        observed = latents[:, self.n_beg - 1 :]
        if n_observed_steps is not None:
            observed = observed[:, :n_observed_steps]
        return observed

    def firing_rates(self, latents: torch.Tensor, *, mode: RateMode = "likelihood") -> torch.Tensor:
        """Return firing rates in Hz for all neurons."""

        linear = latents @ self.c.T + self.bias
        if mode == "pre_sample":
            return _safe_exp(linear)
        if mode == "likelihood":
            return self._positive(self.gain) * (1e-3 + _safe_exp(linear))
        raise ValueError(f"unknown rate mode {mode!r}")

    def poisson_nll(
        self,
        latents: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        include_constants: bool = False,
    ) -> torch.Tensor:
        rates_hz = self.firing_rates(latents, mode="likelihood")[:, :held_in_neurons]
        lambdas = (self.dt * rates_hz).clamp_min(1e-12)
        observed = spikes[:, :held_in_neurons]
        nll = torch.sum(lambdas - observed * torch.log(lambdas))
        if include_constants:
            nll = nll + torch.sum(torch.lgamma(observed + 1.0))
        return nll

    def posterior_cov_sample(
        self,
        *,
        n_controls: int,
        n_samples: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Sample the shared Kronecker posterior covariance."""

        dtype = dtype or self.c.dtype
        device = device or self.c.device
        chol_space = self._cov_chol(self.space_cov_d, self.space_cov_t, self.n_input).to(
            dtype=dtype,
            device=device,
        )
        chol_time = self._cov_chol(self.time_cov_d, self.time_cov_t, n_controls).to(
            dtype=dtype,
            device=device,
        )
        xi = torch.randn(
            n_samples * n_controls,
            self.n_input,
            dtype=dtype,
            device=device,
        )
        v = xi @ chol_space
        v = v.reshape(n_samples, n_controls, self.n_input)
        v = v.transpose(0, 1).reshape(n_controls, n_samples * self.n_input)
        v = chol_time.T @ v
        return v.reshape(n_controls, n_samples, self.n_input).transpose(0, 1)

    def posterior_entropy(self, *, n_controls: int) -> torch.Tensor:
        """Entropy of the shared Gaussian posterior covariance."""

        d_space = self._positive(self.space_cov_d[: self.n_input])
        d_time = self._positive(self.time_cov_d[:n_controls])
        dim = float(self.n_input * n_controls)
        log_det = 2.0 * (
            float(self.n_input) * torch.sum(torch.log(d_time))
            + float(n_controls) * torch.sum(torch.log(d_space))
        )
        return 0.5 * (log_det + dim * (1.0 + math.log(2.0 * math.pi)))

    def student_prior_log_prob_samples(
        self,
        controls: torch.Tensor,
        *,
        include_constants: bool = True,
    ) -> torch.Tensor:
        """Student input prior log probability for samples x time x inputs."""

        if controls.ndim != 3 or controls.shape[-1] != self.n_input:
            raise ValueError(f"expected controls samples x time x {self.n_input}")
        n_samples = int(controls.shape[0])
        u0 = controls[:, : self.n_beg, :].reshape(-1, self.n_input)
        u_rest = controls[:, self.n_beg :, :].reshape(-1, self.n_input)

        sigma0 = self._positive(self.first_step).reshape(1, -1)
        nll = 0.5 * torch.sum((u0 / sigma0) ** 2)

        nu = self._positive(self.nu, lower=2.0 + 1e-6)
        sigma = torch.sqrt((nu - 2.0) / nu) * self._positive(self.spatial_stds).reshape(1, -1)
        if u_rest.numel() > 0:
            tau = 1.0 + torch.sum((u_rest / sigma) ** 2, dim=1) / nu
            nll = nll + 0.5 * (nu + self.n_input) * torch.sum(torch.log(tau))

        if include_constants:
            nll = nll + n_samples * self.n_beg * 0.5 * (
                self.n_input * math.log(2.0 * math.pi) + 2.0 * torch.sum(torch.log(sigma0))
            )
            n_rest = u_rest.shape[0]
            if n_rest > 0:
                student_const = (
                    torch.lgamma(0.5 * nu)
                    - torch.lgamma(0.5 * (nu + self.n_input))
                    + 0.5 * self.n_input * torch.log(math.pi * nu)
                    + torch.sum(torch.log(sigma))
                )
                nll = nll + n_rest * student_const
        return -nll

    def poisson_log_likelihood_samples(
        self,
        latents: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        include_constants: bool = True,
    ) -> torch.Tensor:
        """Poisson observation log-likelihood for sampled latent trajectories."""

        if latents.ndim != 3:
            raise ValueError("expected latents with shape samples x time x latent_dim")
        if spikes.ndim != 2:
            raise ValueError("expected spikes with shape time x neurons")
        c = self.c[:held_in_neurons]
        bias = self.bias[:, :held_in_neurons]
        gain = self._positive(self.gain[:, :held_in_neurons])
        linear = latents @ c.T + bias
        rates = (self.dt * gain * (1e-3 + _safe_exp(linear))).clamp_min(1e-12)
        obs = spikes[: latents.shape[1], :held_in_neurons].unsqueeze(0).to(rates.dtype)
        logp = torch.sum(obs * torch.log(rates) - rates)
        if include_constants:
            logp = logp - float(latents.shape[0]) * torch.sum(torch.lgamma(obs + 1.0))
        return logp

    def elbo_from_controls(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        n_posterior_samples: int = 1,
        include_constants: bool = True,
    ) -> torch.Tensor:
        """Sampled iLQR-VAE ELBO, preserving the supplied control gradient."""

        if controls.ndim != 2:
            raise ValueError("expected posterior mean controls with shape time x input_dim")
        cov = self.posterior_cov_sample(
            n_controls=int(controls.shape[0]),
            n_samples=int(n_posterior_samples),
            dtype=controls.dtype,
            device=controls.device,
        )
        samples = controls.unsqueeze(0) + cov
        latents = self.integrate_samples(samples)
        observed_latents = self.observation_latents_samples(
            latents,
            n_observed_steps=int(spikes.shape[0]),
        )
        log_prior = self.student_prior_log_prob_samples(samples, include_constants=include_constants)
        log_likelihood = self.poisson_log_likelihood_samples(
            observed_latents,
            spikes,
            held_in_neurons=held_in_neurons,
            include_constants=include_constants,
        )
        norm_const = 1.0 / float(n_posterior_samples)
        return self.posterior_entropy(n_controls=int(controls.shape[0])) + norm_const * (
            log_prior + log_likelihood
        )

    def student_prior_nll(self, controls: torch.Tensor, *, include_constants: bool = False) -> torch.Tensor:
        u0 = controls[: self.n_beg]
        u_rest = controls[self.n_beg :]

        sigma0 = self._positive(self.first_step).reshape(1, -1)
        nll = 0.5 * torch.sum((u0 / sigma0) ** 2)

        nu = self._positive(self.nu, lower=2.0 + 1e-6)
        sigma = torch.sqrt((nu - 2.0) / nu) * self._positive(self.spatial_stds).reshape(1, -1)
        tau = 1.0 + torch.sum((u_rest / sigma) ** 2, dim=1) / nu
        nll = nll + 0.5 * (nu + self.n_input) * torch.sum(torch.log(tau))

        if include_constants:
            nll = nll + self.n_beg * 0.5 * (
                self.n_input * math.log(2.0 * math.pi) + 2.0 * torch.sum(torch.log(sigma0))
            )
            n_rest = u_rest.shape[0]
            student_const = (
                torch.lgamma(0.5 * nu)
                - torch.lgamma(0.5 * (nu + self.n_input))
                + 0.5 * self.n_input * torch.log(math.pi * nu)
                + torch.sum(torch.log(sigma))
            )
            nll = nll + n_rest * student_const
        return nll

    def _default_dynamics_step(self, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        x_eff = control @ self.b
        gate = torch.sigmoid(state @ self.uf)
        candidate = _requad(self.bh + (state * gate) @ self.uh) - 1.0 + x_eff @ self.wh
        return (1.0 - gate) * state + gate * candidate

    def _dynamics_step(self, k: int, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if self.n_beg != 1 and k < self.n_beg:
            return state + control @ self.beg_bs[k]
        return self._default_dynamics_step(state, control)

    def _dynamics_x(self, k: int, state: torch.Tensor, control: torch.Tensor) -> torch.Tensor:
        if self.n_beg != 1 and k < self.n_beg:
            return torch.eye(self.n_latent, dtype=state.dtype, device=state.device)

        x_eff = control @ self.b
        f_pre = state @ self.uf
        gate = torch.sigmoid(f_pre)
        d_gate = gate * (1.0 - gate)
        h_hat_pre = self.bh + (state * gate) @ self.uh
        h_hat = _requad(h_hat_pre) - 1.0 + x_eff @ self.wh
        d_phi = _d_requad(h_hat_pre)

        term0 = torch.diag((1.0 - gate).reshape(-1))
        term1 = self.uf * ((state - h_hat) * d_gate)
        term2_left = gate.T * self.uh
        term2_right = self.uf @ ((state * d_gate).T * self.uh)
        term2 = (term2_left + term2_right) * (gate * d_phi)
        return term0 - term1 + term2

    def _dynamics_u(self, k: int, state: torch.Tensor) -> torch.Tensor:
        if self.n_beg != 1 and k < self.n_beg:
            return self.beg_bs[k]
        gate = torch.sigmoid(state @ self.uf)
        return self.b @ (self.wh * gate)

    def _prior_nll_t(self, k: int, control: torch.Tensor, *, include_constants: bool = False) -> torch.Tensor:
        if k < self.n_beg:
            sigma0 = self._positive(self.first_step).reshape(1, -1)
            nll = 0.5 * torch.sum((control / sigma0) ** 2)
            if include_constants:
                nll = nll + 0.5 * (
                    self.n_input * math.log(2.0 * math.pi) + 2.0 * torch.sum(torch.log(sigma0))
                )
            return nll

        nu = self._positive(self.nu, lower=2.0 + 1e-6)
        sigma = torch.sqrt((nu - 2.0) / nu) * self._positive(self.spatial_stds).reshape(1, -1)
        tau = 1.0 + torch.sum((control / sigma) ** 2) / nu
        nll = 0.5 * (nu + self.n_input) * torch.log(tau)
        if include_constants:
            nll = nll + (
                torch.lgamma(0.5 * nu)
                - torch.lgamma(0.5 * (nu + self.n_input))
                + 0.5 * self.n_input * torch.log(math.pi * nu)
                + torch.sum(torch.log(sigma))
            )
        return nll

    def _prior_grad_hess_t(self, k: int, control: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k < self.n_beg:
            var0 = self._positive(self.first_step).reshape(-1) ** 2
            return control / var0, torch.diag(1.0 / var0)

        nu = self._positive(self.nu, lower=2.0 + 1e-6)
        sigma = torch.sqrt((nu - 2.0) / nu) * self._positive(self.spatial_stds).reshape(1, -1)
        sigma2 = sigma.reshape(-1) ** 2
        u = control.reshape(-1)
        u_over_s = u / sigma.reshape(-1)
        tau = 1.0 + torch.sum(u_over_s**2) / nu
        grad = (0.5 * (nu + self.n_input)) * (2.0 * u / sigma2 / nu) / tau
        cst = (nu + self.n_input) / nu / (tau**2)
        term1 = torch.diag(tau / sigma2)
        term2 = 2.0 * torch.outer(u / sigma2, u / sigma2) / nu
        hess = cst * (term1 - term2)
        return grad.reshape(1, -1), hess

    def _poisson_nll_t(
        self,
        state: torch.Tensor,
        spikes_t: torch.Tensor,
        *,
        held_in_neurons: int,
        include_constants: bool = False,
    ) -> torch.Tensor:
        c = self.c[:held_in_neurons]
        bias = self.bias[:, :held_in_neurons]
        gain = self._positive(self.gain[:, :held_in_neurons])
        linear = state @ c.T + bias
        rates = (self.dt * gain * (1e-3 + _safe_exp(linear))).clamp_min(1e-12)
        observed = spikes_t[:, :held_in_neurons]
        nll = torch.sum(rates - observed * torch.log(rates))
        if include_constants:
            nll = nll + torch.sum(torch.lgamma(observed + 1.0))
        return nll

    def _poisson_grad_hess_t(
        self,
        state: torch.Tensor,
        spikes_t: torch.Tensor,
        *,
        held_in_neurons: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.c[:held_in_neurons]
        bias = self.bias[:, :held_in_neurons]
        gain = self._positive(self.gain[:, :held_in_neurons])
        linear = state @ c.T + bias
        active = (linear <= _MAX_LOG_RATE).to(linear.dtype)
        exp_linear = _safe_exp(linear)
        exp_deriv = exp_linear * active
        link = 1e-3 + exp_linear
        tmp1 = self.dt * gain * exp_deriv
        tmp2 = spikes_t * exp_deriv / link
        grad = (tmp1 - tmp2) @ c

        d2_log_link = active * exp_linear * 1e-3 / (link**2)
        if self.control_hessian_mode == "fisher":
            weights = tmp1.reshape(-1)
        else:
            weights = (tmp1 - spikes_t * d2_log_link).reshape(-1)
            if self.control_hessian_mode == "clamped":
                weights = weights.clamp_min(1.0e-12)
            elif self.control_hessian_mode != "true":
                raise ValueError(f"unknown control_hessian_mode {self.control_hessian_mode!r}")
        hess = (c.T * weights) @ c
        return grad, hess

    def _infer_lbfgs(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        history: list[float],
        max_iter: int,
        lr: float,
        include_constants: bool,
        trace_every: int | None,
        trace_evaluations: list[int],
        trace_losses: list[float],
        trace_controls: list[torch.Tensor],
    ) -> None:
        optimizer = torch.optim.LBFGS(
            [controls],
            lr=lr,
            max_iter=max_iter,
            max_eval=max_iter * 5,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            loss = self.posterior_objective(
                controls,
                spikes,
                held_in_neurons=held_in_neurons,
                include_constants=include_constants,
            )
            loss.backward(inputs=[controls])
            loss_value = float(loss.detach().cpu())
            history.append(loss_value)
            _maybe_record_trace(
                controls,
                loss_value,
                evaluation=len(history),
                trace_every=trace_every,
                trace_evaluations=trace_evaluations,
                trace_losses=trace_losses,
                trace_controls=trace_controls,
            )
            return loss

        optimizer.step(closure)

    def _infer_adam(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        history: list[float],
        max_iter: int,
        lr: float,
        include_constants: bool,
        trace_every: int | None,
        trace_evaluations: list[int],
        trace_losses: list[float],
        trace_controls: list[torch.Tensor],
        differentiable: bool = False,
    ) -> torch.Tensor:
        # Functional updates retain the posterior-mean derivative for training.
        current = controls
        first_moment = torch.zeros_like(controls)
        second_moment = torch.zeros_like(controls)
        best_loss = float("inf")
        best_controls = current
        for iteration in range(max_iter + 1):
            loss = self.posterior_objective(
                current,
                spikes,
                held_in_neurons=held_in_neurons,
                include_constants=include_constants,
            )
            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_controls = current
            history.append(loss_value)
            _maybe_record_trace(
                current,
                loss_value,
                evaluation=len(history),
                trace_every=trace_every,
                trace_evaluations=trace_evaluations,
                trace_losses=trace_losses,
                trace_controls=trace_controls,
            )
            if iteration == max_iter:
                break
            gradient = torch.autograd.grad(loss, current, create_graph=differentiable)[0]
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient.square()
            step = iteration + 1
            mean = first_moment / (1.0 - 0.9**step)
            variance = second_moment / (1.0 - 0.999**step)
            # Smooth at zero so inactive controls have finite second derivatives.
            current = current - lr * mean / torch.sqrt(variance + 1.0e-16)
            if not differentiable:
                current = current.detach().requires_grad_(True)
        if not math.isfinite(best_loss):
            raise RuntimeError("Adam posterior-control inference produced no finite objective")
        return best_controls if differentiable else best_controls.detach()

    def _infer_ilqr(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        history: list[float],
        max_iter: int,
        include_constants: bool,
        trace_every: int | None,
        trace_evaluations: list[int],
        trace_losses: list[float],
        trace_controls: list[torch.Tensor],
    ) -> None:
        prev_loss = 1e9
        for iteration in range(max_iter + 1):
            loss = float(
                self.ilqr_objective(
                    controls,
                    spikes,
                    held_in_neurons=held_in_neurons,
                    include_constants=include_constants,
                )
                .detach()
                .cpu()
            )
            history.append(loss)
            _maybe_record_trace(
                controls,
                loss,
                evaluation=len(history),
                trace_every=trace_every,
                trace_evaluations=trace_evaluations,
                trace_losses=trace_losses,
                trace_controls=trace_controls,
            )
            pct_change = abs((loss - prev_loss) / prev_loss)
            if pct_change < 1e-6:
                break
            prev_loss = loss
            if iteration == max_iter:
                break

            tape = self._ilqr_tape(controls, spikes, held_in_neurons=held_in_neurons)
            gains, df1, df2 = self._ilqr_backward(
                tape, spikes[-1:], held_in_neurons=held_in_neurons
            )
            next_controls = self._ilqr_linesearch(
                controls,
                spikes,
                tape,
                gains,
                f0=loss,
                df1=df1,
                df2=df2,
                held_in_neurons=held_in_neurons,
                include_constants=include_constants,
            )
            controls.copy_(next_controls)

    def _infer_ilqr_unrolled(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
        history: list[float],
        max_iter: int,
        include_constants: bool,
        trace_every: int | None,
        trace_evaluations: list[int],
        trace_losses: list[float],
        trace_controls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Run the analytic iLQR updates while preserving a gradient graph.

        The published implementation differentiates through the iLQR posterior
        mean with an implicit control adjoint. For short training solves
        (`max_iter=2` in the tutorial), unrolling the analytic updates gives
        NLB2 a practical gradient path through the recognition mean while
        preserving the existing detached iLQR path for evaluation.
        """

        current = controls
        prev_loss = 1e9
        for iteration in range(max_iter + 1):
            with torch.no_grad():
                loss = float(self.ilqr_objective(
                    current,
                    spikes,
                    held_in_neurons=held_in_neurons,
                    include_constants=include_constants,
                ).cpu())
            history.append(loss)
            _maybe_record_trace(
                current,
                loss,
                evaluation=len(history),
                trace_every=trace_every,
                trace_evaluations=trace_evaluations,
                trace_losses=trace_losses,
                trace_controls=trace_controls,
            )
            pct_change = abs((loss - prev_loss) / prev_loss)
            if pct_change < 1e-6:
                break
            prev_loss = loss
            if iteration == max_iter:
                break

            tape = self._ilqr_tape(current, spikes, held_in_neurons=held_in_neurons)
            gains, df1, df2 = self._ilqr_backward(
                tape, spikes[-1:], held_in_neurons=held_in_neurons
            )
            current = self._ilqr_linesearch(
                current,
                spikes,
                tape,
                gains,
                f0=loss,
                df1=df1,
                df2=df2,
                held_in_neurons=held_in_neurons,
                include_constants=include_constants,
            )
        return current

    def _ilqr_tape(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        *,
        held_in_neurons: int,
    ) -> list[_TapeStep]:
        # Once the recurrent rollout is known, all local derivatives are
        # independent over time. Unbind once to avoid a scatter per timestep
        # when the sequential Riccati recursion backpropagates into this tape.
        states = torch.cat([controls.new_zeros(1, self.n_latent), self.integrate(controls)[:-1]])
        state_rows, control_rows = states.unsqueeze(1), controls.unsqueeze(1)
        start = self.n_beg if self.n_beg != 1 else 0
        if len(controls) > start:
            a = torch.vmap(lambda x, u: self._dynamics_x(self.n_beg, x, u))(
                state_rows[start:], control_rows[start:]
            )
            b = torch.vmap(lambda x: self._dynamics_u(self.n_beg, x))(state_rows[start:])
        else:
            a = controls.new_zeros(0, self.n_latent, self.n_latent)
            b = controls.new_zeros(0, self.n_input, self.n_latent)
        if self.n_beg != 1:
            identity = torch.eye(self.n_latent, dtype=controls.dtype, device=controls.device)
            a = torch.cat([identity.expand(self.n_beg, -1, -1), a])
            b = torch.cat([self.beg_bs, b])
        initial_grad, initial_hess = torch.vmap(
            lambda u: self._prior_grad_hess_t(0, u)
        )(control_rows[:self.n_beg])
        if len(controls) > self.n_beg:
            rest_grad, rest_hess = torch.vmap(
                lambda u: self._prior_grad_hess_t(self.n_beg, u)
            )(control_rows[self.n_beg:])
            likelihood_grad, likelihood_hess = torch.vmap(
                lambda x, obs: self._poisson_grad_hess_t(x, obs, held_in_neurons=held_in_neurons)
            )(state_rows[self.n_beg:], spikes[:-1].unsqueeze(1))
        else:
            rest_grad = controls.new_zeros(0, 1, self.n_input)
            rest_hess = controls.new_zeros(0, self.n_input, self.n_input)
            likelihood_grad = controls.new_zeros(0, 1, self.n_latent)
            likelihood_hess = controls.new_zeros(0, self.n_latent, self.n_latent)
        rlu = torch.cat([initial_grad, rest_grad])
        rluu = torch.cat([initial_hess, rest_hess])
        rlx = torch.cat([controls.new_zeros(self.n_beg, 1, self.n_latent), likelihood_grad])
        rlxx = torch.cat([
            controls.new_zeros(self.n_beg, self.n_latent, self.n_latent), likelihood_hess
        ])
        rlux = controls.new_zeros(self.n_input, self.n_latent)
        fields = (state_rows, control_rows, a, b, rlx, rlu, rlxx, rluu)
        return [
            _TapeStep(x=x, u=u, a=at, b=bt, rlx=lx, rlu=lu, rlxx=lxx, rluu=luu, rlux=rlux)
            for x, u, at, bt, lx, lu, lxx, luu in zip(*(value.unbind(0) for value in fields))
        ]

    def _ilqr_backward(
        self,
        tape: list[_TapeStep],
        terminal_spikes: torch.Tensor,
        *,
        held_in_neurons: int,
    ) -> tuple[list[tuple[_TapeStep, torch.Tensor, torch.Tensor]], float, float]:
        terminal_state = self._dynamics_step(len(tape) - 1, tape[-1].x, tape[-1].u)
        flx, flxx = self._poisson_grad_hess_t(
            terminal_state, terminal_spikes, held_in_neurons=held_in_neurons
        )
        eye_u = torch.eye(self.n_input, dtype=self.c.dtype, device=self.c.device)

        delta = 1.0
        mu = 0.0
        regularization_attempts = 0
        max_regularization_attempts = 64
        while True:
            vxx = flxx
            vx = flx
            acc_reversed: list[tuple[_TapeStep, torch.Tensor, torch.Tensor]] = []
            df1 = torch.zeros((), dtype=self.c.dtype, device=self.c.device)
            df2 = torch.zeros((), dtype=self.c.dtype, device=self.c.device)
            restart = False

            for step in reversed(tape):
                at = step.a.T
                bt = step.b.T
                qx = step.rlx + vx @ at
                qu = step.rlu + vx @ bt
                qxx = step.rlxx + step.a @ vxx @ at
                quu = step.rluu + step.b @ vxx @ bt
                quu = 0.5 * (quu + quu.T)
                qtuu = quu + mu * eye_u
                try:
                    cholesky, info = torch.linalg.cholesky_ex(qtuu)
                    is_pos_def = bool((info == 0).all().detach().cpu())
                except RuntimeError:
                    is_pos_def = False
                if not is_pos_def:
                    regularization_attempts += 1
                    if regularization_attempts > max_regularization_attempts:
                        raise RuntimeError("iLQR backward pass did not find a positive definite Q_uu.")
                    delta, mu = _increase_regularization(delta, mu)
                    restart = True
                    break

                qux = step.rlux + step.b @ vxx @ at
                try:
                    feedback = -torch.cholesky_solve(qux, cholesky).T
                    feedforward = -torch.cholesky_solve(qu.T, cholesky).T
                except RuntimeError:
                    regularization_attempts += 1
                    if regularization_attempts > max_regularization_attempts:
                        raise RuntimeError("iLQR backward pass failed to solve regularized Q_uu.")
                    delta, mu = _increase_regularization(delta, mu)
                    restart = True
                    break
                vxx = qxx + (feedback @ qux).T
                vxx = 0.5 * (vxx + vxx.T)
                vx = qx + qu @ feedback.T
                acc_reversed.append((step, feedback, feedforward))
                df1 = df1 + torch.sum(feedforward * qu)
                df2 = df2 + torch.sum(feedforward @ quu @ feedforward.T)

            if not restart:
                acc = list(reversed(acc_reversed))
                return acc, float(df1.detach().cpu()), float(df2.detach().cpu())

    def _ilqr_linesearch(
        self,
        controls: torch.Tensor,
        spikes: torch.Tensor,
        tape: list[_TapeStep],
        gains: list[tuple[_TapeStep, torch.Tensor, torch.Tensor]],
        *,
        f0: float,
        df1: float,
        df2: float,
        held_in_neurons: int,
        include_constants: bool,
        alpha_min: float = 1e-8,
        tau: float = 0.5,
        beta: float = 0.1,
    ) -> torch.Tensor:
        del tape
        alpha = tau
        while alpha >= alpha_min:
            candidate = self._ilqr_forward_update(gains, alpha)
            with torch.no_grad():
                candidate_loss = float(
                    self.ilqr_objective(
                        candidate,
                        spikes,
                        held_in_neurons=held_in_neurons,
                        include_constants=include_constants,
                    )
                    .cpu()
                )
            if not math.isfinite(candidate_loss):
                alpha *= tau
                continue
            predicted_change = alpha * df1 + 0.5 * alpha * alpha * df2
            if candidate_loss <= f0 + beta * min(predicted_change, 0.0):
                return candidate
            alpha *= tau
        raise RuntimeError("iLQR line search did not converge")

    def _ilqr_forward_update(
        self,
        gains: list[tuple[_TapeStep, torch.Tensor, torch.Tensor]],
        alpha: float,
    ) -> torch.Tensor:
        xhat = torch.zeros(1, self.n_latent, dtype=self.c.dtype, device=self.c.device)
        updated = []
        for k, (step, feedback, feedforward) in enumerate(gains):
            dx = xhat - step.x
            du = dx @ feedback + alpha * feedforward
            uhat = step.u + du
            updated.append(uhat)
            xhat = self._dynamics_step(k, xhat, uhat)
        return torch.cat(updated, dim=0)

    def _make_initial_condition_maps(self) -> torch.Tensor:
        maps = []
        for k in range(self.n_beg):
            matrix = torch.zeros(self.n_input, self.n_latent, dtype=torch.float64)
            rows = torch.arange(self.n_input)
            cols = torch.arange(k * self.n_input, (k + 1) * self.n_input)
            matrix[rows, cols] = 1.0
            maps.append(matrix)
        return torch.stack(maps, dim=0)


def poisson_log_likelihood(spikes: np.ndarray, rates_hz: np.ndarray, *, dt: float = 5e-3) -> float:
    lambdas = np.clip(dt * rates_hz, 1e-12, None)
    return float(np.sum(spikes * np.log(lambdas) - lambdas - gammaln(spikes + 1.0)))


def co_bps(spikes: np.ndarray, model_rates_hz: np.ndarray, baseline_rates_hz: np.ndarray, *, dt: float = 5e-3) -> float:
    spike_count = float(np.sum(spikes))
    if spike_count <= 0:
        raise ValueError("cannot compute bits/spike with zero held-out spikes")
    model_ll = poisson_log_likelihood(spikes, model_rates_hz, dt=dt)
    baseline_ll = poisson_log_likelihood(spikes, baseline_rates_hz, dt=dt)
    return (model_ll - baseline_ll) / (math.log(2.0) * spike_count)


def nlb_bits_per_spike(rates: np.ndarray, spikes: np.ndarray) -> float:
    """Neural Latents Benchmark bits/spike.

    ``rates`` and ``spikes`` are expected spike counts per bin with identical
    shapes, matching ``nlb_tools.evaluation.bits_per_spike``.
    """

    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")
    nll_model = poisson_negative_log_likelihood_counts(rates, spikes)
    null_rates = np.tile(
        np.nanmean(spikes, axis=tuple(range(spikes.ndim - 1)), keepdims=True),
        spikes.shape[:-1] + (1,),
    )
    nll_null = poisson_negative_log_likelihood_counts(null_rates, spikes)
    spike_count = np.nansum(spikes)
    if spike_count <= 0:
        raise ValueError("cannot compute bits/spike with zero spikes")
    return float((nll_null - nll_model) / spike_count / np.log(2.0))


def poisson_negative_log_likelihood_counts(
    rates: np.ndarray,
    spikes: np.ndarray,
    *,
    zero_floor: float = 1e-9,
) -> float:
    if rates.shape != spikes.shape:
        raise ValueError(f"rates and spikes shapes differ: {rates.shape} != {spikes.shape}")
    rates = np.array(rates, dtype=np.float64, copy=True)
    spikes = np.asarray(spikes, dtype=np.float64)
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


def _requad(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + torch.sqrt(4.0 + x * x))


def _d_requad(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + x / torch.sqrt(4.0 + x * x))


def _tensor(value: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float64)


def _increase_regularization(delta: float, mu: float) -> tuple[float, float]:
    delta = max(2.0, 2.0 * delta)
    mu = max(1e-6, mu * delta)
    return delta, mu


def _safe_exp(linear: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(linear, max=_MAX_LOG_RATE))


def _maybe_record_trace(
    controls: torch.Tensor,
    loss_value: float,
    *,
    evaluation: int,
    trace_every: int | None,
    trace_evaluations: list[int],
    trace_losses: list[float],
    trace_controls: list[torch.Tensor],
) -> None:
    if trace_every is None or trace_every <= 0:
        return
    if evaluation == 1 or evaluation % trace_every == 0:
        trace_evaluations.append(evaluation)
        trace_losses.append(loss_value)
        trace_controls.append(controls.detach().cpu().clone())
