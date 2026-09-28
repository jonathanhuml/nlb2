"""iLQR-VAE adapter for posterior-control inference and ELBO training."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator
from torch import Tensor

from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.models.ilqr_vae_core import ILQRVAE as TutorialILQRVAE
from nlb2.models.ilqr_vae_core import load_tutorial_params
from nlb2.models.ilqr_vae_core import make_random_params
from nlb2.models.ilqr_vae_core.params import TutorialParams
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
