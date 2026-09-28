"""PyTorch translation of the tutorial iLQR-VAE generative model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.special import gammaln
import torch

from .params import TutorialParams


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


class ILQRVAE(torch.nn.Module):
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
