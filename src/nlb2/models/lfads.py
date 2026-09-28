"""LFADS adapter for raw-count neural population modeling."""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
from pydantic import Field
from torch import Tensor, nn
import torch.nn.functional as F

from nlb2.metrics import (
    EvaluationAdapter,
    EvaluationResult,
    SyntheticEvaluationAdapter,
    compute_available_metrics,
)
from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


def _kl_diag_gaussian(
    posterior_mean: Tensor,
    posterior_logvar: Tensor,
    prior_mean: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    """KL divergence from posterior diagonal Gaussian to prior diagonal Gaussian."""

    return 0.5 * (
        prior_logvar
        - posterior_logvar
        + torch.exp(posterior_logvar - prior_logvar)
        + (posterior_mean - prior_mean).pow(2) / torch.exp(prior_logvar)
        - 1.0
    ).sum(dim=-1)


def _init_variance_scaling_(weight: Tensor, scale_dim: int) -> None:
    nn.init.normal_(weight, std=float(scale_dim) ** -0.5)


def _init_linear_(linear: nn.Linear) -> None:
    _init_variance_scaling_(linear.weight, int(linear.in_features))
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_gru_cell_(cell: nn.GRUCell, scale_dim: Optional[int] = None) -> None:
    ih_scale = int(scale_dim or cell.input_size)
    hh_scale = int(scale_dim or cell.hidden_size)
    _init_variance_scaling_(cell.weight_ih, ih_scale)
    _init_variance_scaling_(cell.weight_hh, hh_scale)
    nn.init.ones_(cell.bias_ih)
    cell.bias_ih.data[-cell.hidden_size :] = 0.0
    nn.init.zeros_(cell.bias_hh)


class _ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        clip_value: float,
        is_encoder: bool = False,
    ) -> None:
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = float(clip_value)
        scale_dim = input_size + hidden_size if is_encoder else None
        self._lfads_scale_dim = scale_dim
        _init_gru_cell_(self, scale_dim=scale_dim)

    def forward(self, input: Tensor, hidden: Tensor) -> Tensor:
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        weight_hh_zr, weight_hh_n = torch.split(
            self.weight_hh,
            [2 * self.hidden_size, self.hidden_size],
            dim=0,
        )
        bias_hh_zr, bias_hh_n = torch.split(
            self.bias_hh,
            [2 * self.hidden_size, self.hidden_size],
            dim=0,
        )
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1.0 - z) * n
        return torch.clamp(hidden, min=-self.clip_value, max=self.clip_value)


class _ClippedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, clip_value: float) -> None:
        super().__init__()
        self.cell = _ClippedGRUCell(
            input_size,
            hidden_size,
            clip_value=clip_value,
            is_encoder=True,
        )

    def forward(self, input: Tensor, h_0: Tensor) -> tuple[Tensor, Tensor]:
        hidden = h_0
        output = []
        for input_step in input.transpose(0, 1):
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        return torch.stack(output, dim=1), hidden


class _BidirectionalClippedGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, *, clip_value: float) -> None:
        super().__init__()
        self.fwd_gru = _ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = _ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: Tensor, h_0: Tensor) -> tuple[Tensor, Tensor]:
        h0_fwd, h0_bwd = h_0
        output_fwd, hn_fwd = self.fwd_gru(input, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(torch.flip(input, [1]), h0_bwd)
        output_bwd = torch.flip(output_bwd, [1])
        output = torch.cat([output_fwd, output_bwd], dim=-1)
        h_n = torch.stack([hn_fwd, hn_bwd], dim=0)
        return output, h_n


@BaseModelConfig.register
class LFADSConfig(BaseModelConfig):
    """Config for LFADS on raw spike-count observations."""

    name: Literal["lfads"] = "lfads"
    objective: str = "lfads_elbo"
    generator_dim: int = 64
    initial_condition_dim: Optional[int] = None
    inferred_input_dim: int = 2
    factor_dim: int = 20
    g0_encoder_dim: int = 64
    controller_encoder_dim: int = 64
    controller_dim: int = 64
    g0_prior_kappa: float = 0.1
    inferred_input_prior_kappa: float = 0.1
    keep_prob: float = 0.95
    clip_val: float = 5.0
    dt: float = 1.0
    log_rate_min: float = -8.0
    log_rate_max: float = 8.0
    posterior_logvar_min: float = math.log(1e-4)
    posterior_logvar_max: float = 5.0
    controller_posterior_logvar_min: Optional[float] = None
    controller_posterior_logvar_max: Optional[float] = None
    reconstruction_time_steps: Optional[int] = None
    controller_lag: int = 0
    readout_neurons: Optional[int] = None
    output_neuron_start: Optional[int] = None
    output_neurons: Optional[int] = None
    use_log1p_encoder_inputs: bool = True
    initialize_log_rate_bias: bool = True
    prediction_samples: int = 1
    loss_scale: float = 1.0
    reconstruction_reduce_mean: bool = False
    coordinated_dropout_rate: float = 0.0
    coordinated_dropout_pass_rate: float = 0.0
    coordinated_dropout_ic_enc_seq_len: int = 0
    inferred_input_prior: Literal["independent", "autoregressive"] = "independent"
    inferred_input_prior_tau: float = 10.0
    kl_weight_schedule_start: int = 0
    kl_weight_schedule_dur: int = 2_000
    kl_g0_scale: float = 1.0
    kl_u_scale: float = 1.0
    l2_weight_schedule_start: int = 0
    l2_weight_schedule_dur: int = 2_000
    l2_gen_scale: float = 0.0
    l2_con_scale: float = 0.0
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=1e-3,
            weight_decay=0.0,
            gradient_clip=200.0,
        )
    )

    def build(self, n_neurons: int, n_time: int) -> "LFADS":
        return LFADS(
            input_neurons=n_neurons,
            n_time=n_time,
            readout_neurons=self.readout_neurons or n_neurons,
            generator_dim=self.generator_dim,
            initial_condition_dim=self.initial_condition_dim,
            inferred_input_dim=self.inferred_input_dim,
            factor_dim=self.factor_dim,
            g0_encoder_dim=self.g0_encoder_dim,
            controller_encoder_dim=self.controller_encoder_dim,
            controller_dim=self.controller_dim,
            g0_prior_kappa=self.g0_prior_kappa,
            inferred_input_prior_kappa=self.inferred_input_prior_kappa,
            keep_prob=self.keep_prob,
            clip_val=self.clip_val,
            dt=self.dt,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            posterior_logvar_min=self.posterior_logvar_min,
            posterior_logvar_max=self.posterior_logvar_max,
            controller_posterior_logvar_min=self.controller_posterior_logvar_min,
            controller_posterior_logvar_max=self.controller_posterior_logvar_max,
            reconstruction_time_steps=self.reconstruction_time_steps,
            controller_lag=self.controller_lag,
            output_neuron_start=self.output_neuron_start,
            output_neurons=self.output_neurons,
            use_log1p_encoder_inputs=self.use_log1p_encoder_inputs,
            initialize_log_rate_bias=self.initialize_log_rate_bias,
            prediction_samples=self.prediction_samples,
            loss_scale=self.loss_scale,
            reconstruction_reduce_mean=self.reconstruction_reduce_mean,
            coordinated_dropout_rate=self.coordinated_dropout_rate,
            coordinated_dropout_pass_rate=self.coordinated_dropout_pass_rate,
            coordinated_dropout_ic_enc_seq_len=self.coordinated_dropout_ic_enc_seq_len,
            inferred_input_prior=self.inferred_input_prior,
            inferred_input_prior_tau=self.inferred_input_prior_tau,
            kl_weight_schedule_start=self.kl_weight_schedule_start,
            kl_weight_schedule_dur=self.kl_weight_schedule_dur,
            kl_g0_scale=self.kl_g0_scale,
            kl_u_scale=self.kl_u_scale,
            l2_weight_schedule_start=self.l2_weight_schedule_start,
            l2_weight_schedule_dur=self.l2_weight_schedule_dur,
            l2_gen_scale=self.l2_gen_scale,
            l2_con_scale=self.l2_con_scale,
            objective=self.objective,
        )

    def build_from_data(self, data: Any) -> "LFADS":
        model = self.build(n_neurons=data.n_neurons, n_time=data.n_time)
        if self.initialize_log_rate_bias:
            full_spikes = _full_spikes_from_dataset(getattr(data, "train_dataset", None))
            if (
                full_spikes is not None
                and int(full_spikes.shape[-1]) == model.readout_neurons
                and int(full_spikes.shape[1]) == model.reconstruction_time_steps
            ):
                model.initialize_log_rate_bias_from_counts(full_spikes)
        return model


class LFADS(BaseDynamicsModel):
    """Latent Factor Analysis via Dynamical Systems for binned spike counts.

    ## When to use

    Use LFADS as a nonlinear variational sequence model for neural population
    spike counts. This implementation adapts the LFADS demo architecture into
    the NLB2 model contract: bidirectional encoders infer a generator initial
    condition and controller context, a generator GRU produces latent factors,
    and an exponential readout returns Poisson firing rates.

    ## Assumptions

    LFADS expects raw nonnegative spike counts. Dataset-level smoothing should
    be disabled for this model. The optional `log1p` encoder transform only
    changes the recognition network input; the reconstruction loss still uses
    raw counts and the returned `rates` remain in the generated Lorenz rate
    space. The default readout bias initialization uses observed count means,
    not generated ground-truth rates.

    ## Outputs

    `forward` returns nonnegative rate predictions, latent factor trajectories,
    and variational diagnostics in `extras`. `loss` computes a Poisson
    reconstruction objective plus scheduled KL and optional recurrent L2 terms.
    """

    def __init__(
        self,
        input_neurons: int,
        n_time: int,
        readout_neurons: Optional[int] = None,
        generator_dim: int = 64,
        initial_condition_dim: Optional[int] = None,
        inferred_input_dim: int = 2,
        factor_dim: int = 20,
        g0_encoder_dim: int = 64,
        controller_encoder_dim: int = 64,
        controller_dim: int = 64,
        g0_prior_kappa: float = 0.1,
        inferred_input_prior_kappa: float = 0.1,
        keep_prob: float = 0.95,
        clip_val: float = 5.0,
        dt: float = 1.0,
        log_rate_min: float = -8.0,
        log_rate_max: float = 8.0,
        posterior_logvar_min: float = math.log(1e-4),
        posterior_logvar_max: float = 5.0,
        controller_posterior_logvar_min: Optional[float] = None,
        controller_posterior_logvar_max: Optional[float] = None,
        reconstruction_time_steps: Optional[int] = None,
        controller_lag: int = 0,
        output_neuron_start: Optional[int] = None,
        output_neurons: Optional[int] = None,
        use_log1p_encoder_inputs: bool = True,
        initialize_log_rate_bias: bool = True,
        prediction_samples: int = 1,
        loss_scale: float = 1.0,
        reconstruction_reduce_mean: bool = False,
        coordinated_dropout_rate: float = 0.0,
        coordinated_dropout_pass_rate: float = 0.0,
        coordinated_dropout_ic_enc_seq_len: int = 0,
        inferred_input_prior: Literal["independent", "autoregressive"] = "independent",
        inferred_input_prior_tau: float = 10.0,
        kl_weight_schedule_start: int = 0,
        kl_weight_schedule_dur: int = 2_000,
        kl_g0_scale: float = 1.0,
        kl_u_scale: float = 1.0,
        l2_weight_schedule_start: int = 0,
        l2_weight_schedule_dur: int = 2_000,
        l2_gen_scale: float = 0.0,
        l2_con_scale: float = 0.0,
        objective: str = "lfads_elbo",
    ) -> None:
        super().__init__()
        self.input_neurons = int(input_neurons)
        self.readout_neurons = int(readout_neurons or input_neurons)
        self.n_neurons = self.input_neurons
        self.n_time = int(n_time)
        self.generator_dim = int(generator_dim)
        self.initial_condition_dim = int(initial_condition_dim or generator_dim)
        self.inferred_input_dim = int(inferred_input_dim)
        self.factor_dim = int(factor_dim)
        self.g0_encoder_dim = int(g0_encoder_dim)
        self.controller_encoder_dim = int(controller_encoder_dim)
        self.controller_dim = int(controller_dim)
        self.keep_prob = float(keep_prob)
        self.clip_val = float(clip_val)
        self.dt = float(dt)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.posterior_logvar_min = float(posterior_logvar_min)
        self.posterior_logvar_max = float(posterior_logvar_max)
        self.controller_posterior_logvar_min = (
            None
            if controller_posterior_logvar_min is None
            else float(controller_posterior_logvar_min)
        )
        self.controller_posterior_logvar_max = (
            None
            if controller_posterior_logvar_max is None
            else float(controller_posterior_logvar_max)
        )
        self.reconstruction_time_steps = int(reconstruction_time_steps or n_time)
        self.controller_lag = int(controller_lag)
        self.output_neuron_start = output_neuron_start
        self.output_neurons = output_neurons
        self.use_log1p_encoder_inputs = bool(use_log1p_encoder_inputs)
        self.initialize_log_rate_bias = bool(initialize_log_rate_bias)
        self.prediction_samples = int(prediction_samples)
        self.loss_scale = float(loss_scale)
        self.reconstruction_reduce_mean = bool(reconstruction_reduce_mean)
        self.coordinated_dropout_rate = float(coordinated_dropout_rate)
        self.coordinated_dropout_pass_rate = float(coordinated_dropout_pass_rate)
        self.coordinated_dropout_ic_enc_seq_len = int(coordinated_dropout_ic_enc_seq_len)
        self.inferred_input_prior = str(inferred_input_prior)
        self.inferred_input_prior_tau = float(inferred_input_prior_tau)
        self.kl_weight_schedule_start = int(kl_weight_schedule_start)
        self.kl_weight_schedule_dur = int(kl_weight_schedule_dur)
        self.kl_g0_scale = float(kl_g0_scale)
        self.kl_u_scale = float(kl_u_scale)
        self.l2_weight_schedule_start = int(l2_weight_schedule_start)
        self.l2_weight_schedule_dur = int(l2_weight_schedule_dur)
        self.l2_gen_scale = float(l2_gen_scale)
        self.l2_con_scale = float(l2_con_scale)
        self.objective = objective

        if not 0.0 < self.keep_prob <= 1.0:
            raise ValueError("keep_prob must be in (0, 1].")
        if self.inferred_input_dim < 1:
            raise ValueError("inferred_input_dim must be positive.")
        if self.initial_condition_dim < 1:
            raise ValueError("initial_condition_dim must be positive.")
        if self.readout_neurons < 1:
            raise ValueError("readout_neurons must be positive.")
        if self.output_neuron_start is not None and self.output_neuron_start < 0:
            raise ValueError("output_neuron_start must be nonnegative.")
        if self.output_neurons is not None and self.output_neurons < 1:
            raise ValueError("output_neurons must be positive.")
        if not 0.0 <= self.coordinated_dropout_rate < 1.0:
            raise ValueError("coordinated_dropout_rate must be in [0, 1).")
        if not 0.0 <= self.coordinated_dropout_pass_rate <= 1.0:
            raise ValueError("coordinated_dropout_pass_rate must be in [0, 1].")
        if self.coordinated_dropout_ic_enc_seq_len < 0:
            raise ValueError("coordinated_dropout_ic_enc_seq_len must be nonnegative.")
        if self.inferred_input_prior not in {"independent", "autoregressive"}:
            raise ValueError("inferred_input_prior must be 'independent' or 'autoregressive'.")
        if self.inferred_input_prior_tau <= 0.0:
            raise ValueError("inferred_input_prior_tau must be positive.")
        if self.reconstruction_time_steps < self.n_time:
            raise ValueError("reconstruction_time_steps must be >= n_time.")
        if self.controller_lag < 0:
            raise ValueError("controller_lag must be nonnegative.")
        if (
            self.controller_posterior_logvar_min is not None
            and self.controller_posterior_logvar_max is not None
            and self.controller_posterior_logvar_min > self.controller_posterior_logvar_max
        ):
            raise ValueError(
                "controller_posterior_logvar_min must be <= "
                "controller_posterior_logvar_max."
            )

        self.g0_encoder = _BidirectionalClippedGRU(
            input_size=self.input_neurons,
            hidden_size=self.g0_encoder_dim,
            clip_value=self.clip_val,
        )
        self.controller_encoder = _BidirectionalClippedGRU(
            input_size=self.input_neurons,
            hidden_size=self.controller_encoder_dim,
            clip_value=self.clip_val,
        )
        self.g0_encoder_h0 = nn.Parameter(torch.zeros(2, 1, self.g0_encoder_dim))
        self.controller_encoder_h0 = nn.Parameter(
            torch.zeros(2, 1, self.controller_encoder_dim)
        )
        self.controller = _ClippedGRUCell(
            input_size=2 * self.controller_encoder_dim + self.factor_dim,
            hidden_size=self.controller_dim,
            clip_value=self.clip_val,
        )
        self.generator = _ClippedGRUCell(
            input_size=self.inferred_input_dim,
            hidden_size=self.generator_dim,
            clip_value=self.clip_val,
        )

        self.fc_g0_mean = nn.Linear(2 * self.g0_encoder_dim, self.initial_condition_dim)
        self.fc_g0_logvar = nn.Linear(2 * self.g0_encoder_dim, self.initial_condition_dim)
        self.fc_ic_to_generator = nn.Linear(self.initial_condition_dim, self.generator_dim)
        self.fc_u_mean = nn.Linear(self.controller_dim, self.inferred_input_dim)
        self.fc_u_logvar = nn.Linear(self.controller_dim, self.inferred_input_dim)
        self.fc_factors = nn.Linear(self.generator_dim, self.factor_dim, bias=False)
        self.fc_log_rates = nn.Linear(self.factor_dim, self.readout_neurons)
        self.dropout = nn.Dropout(1.0 - self.keep_prob)

        self.g0_prior_mean = nn.Parameter(torch.tensor(0.0))
        self.u_prior_mean = nn.Parameter(
            torch.tensor(0.0),
            requires_grad=self.inferred_input_prior == "independent",
        )
        self.g0_prior_logvar = nn.Parameter(torch.tensor(math.log(g0_prior_kappa)))
        self.u_prior_logvar = nn.Parameter(torch.tensor(math.log(inferred_input_prior_kappa)))
        self.u_prior_logtau = nn.Parameter(
            torch.tensor(math.log(self.inferred_input_prior_tau)),
            requires_grad=self.inferred_input_prior == "autoregressive",
        )
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))
        self.register_buffer("_rate_bias_initialized", torch.tensor(False))
        self._last_cd_grad_mask: Tensor | None = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, _ClippedGRUCell):
                _init_gru_cell_(module, scale_dim=module._lfads_scale_dim)
                module.bias_hh.requires_grad = False
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        parameter.data.normal_(std=parameter.shape[1] ** -0.5)
                    elif "weight_hh" in name:
                        parameter.data.normal_(std=parameter.shape[1] ** -0.5)
                    elif "bias" in name:
                        parameter.data.zero_()
            elif isinstance(module, nn.GRUCell):
                module.weight_ih.data.normal_(std=module.weight_ih.shape[1] ** -0.5)
                module.weight_hh.data.normal_(std=module.weight_hh.shape[1] ** -0.5)
                module.bias_ih.data.zero_()
                module.bias_hh.data.zero_()
            elif isinstance(module, nn.Linear):
                _init_linear_(module)
        self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data, dim=1)

    def forward(self, x: Tensor) -> ModelOutput:
        return self._forward(x, sample=self.training)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        if output.rates is None:
            raise RuntimeError("LFADS.forward did not return rates.")
        target = self._reconstruction_target(batch, output.rates).to(
            device=self.device,
            dtype=output.rates.dtype,
        )
        rates = self._rates_for_reconstruction_target(output.rates, target)
        dt = self._batch_dt(batch, target).to(device=self.device, dtype=output.rates.dtype)
        spike_means = (rates * dt).clamp_min(1e-8)
        finite_target = torch.isfinite(target)
        if not finite_target.any():
            raise ValueError("LFADS loss requires at least one finite reconstruction target.")
        safe_target = torch.where(finite_target, target, torch.zeros_like(target))
        recon_terms = spike_means - safe_target * torch.log(spike_means)
        recon_terms = self._apply_coordinated_dropout_grad_mask(recon_terms)
        recon_terms = recon_terms[finite_target]
        if self.reconstruction_reduce_mean:
            recon_nll = recon_terms.mean()
        else:
            recon_nll = recon_terms.sum() / target.shape[0]

        kl_g0 = self.kl_g0_scale * output.extras["kl_g0"]
        kl_u = self.kl_u_scale * output.extras["kl_u"]
        kl = kl_g0 + kl_u
        kl_weight = self._scheduled_weight(
            int(self._train_step.item()),
            self.kl_weight_schedule_start,
            self.kl_weight_schedule_dur,
        )
        l2 = self._l2_penalty()
        l2_weight = self._scheduled_weight(
            int(self._train_step.item()),
            self.l2_weight_schedule_start,
            self.l2_weight_schedule_dur,
        )
        total = self.loss_scale * (recon_nll + kl_weight * kl + l2_weight * l2)

        if self.training:
            self._train_step.add_(1)

        return LossOutput(
            total=total,
            named_terms={
                "reconstruction_nll": recon_nll,
                "kl": kl,
                "kl_g0": kl_g0,
                "kl_u": kl_u,
                "kl_weight": kl_weight,
                "l2": l2,
                "l2_weight": l2_weight,
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        if self.prediction_samples <= 1:
            return self._forward(x, sample=False).rates

        rates = []
        was_training = self.training
        self.eval()
        try:
            for _ in range(self.prediction_samples):
                rates.append(self._forward(x, sample=True).rates)
        finally:
            self.train(was_training)
        return torch.stack(rates, dim=0).mean(dim=0)

    def _forward(self, x: Tensor, sample: bool) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("LFADS expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.input_neurons:
            raise ValueError(f"Expected {self.input_neurons} input neurons, got {x.shape[-1]}.")
        if not torch.isfinite(x).all():
            raise ValueError("LFADS expects finite spike-count observations.")
        if torch.any(x < 0):
            raise ValueError("LFADS expects nonnegative spike-count observations.")

        x = x.to(device=self.device, dtype=self.fc_log_rates.weight.dtype)
        self._maybe_initialize_rate_bias(x)
        encoder_x = torch.log1p(x) if self.use_log1p_encoder_inputs else x
        encoder_x = self._apply_coordinated_dropout_input(encoder_x)
        if self.keep_prob < 1.0:
            encoder_x = self.dropout(encoder_x)

        g0_h0 = self.g0_encoder_h0.expand(-1, x.shape[0], -1).contiguous()
        _, g0_hidden = self.g0_encoder(encoder_x, g0_h0)
        g0_context = torch.cat((g0_hidden[0], g0_hidden[1]), dim=-1)
        if self.keep_prob < 1.0:
            g0_context = self.dropout(g0_context)

        g0_mean = self.fc_g0_mean(g0_context)
        g0_logvar = self._ic_posterior_logvar(self.fc_g0_logvar(g0_context))
        initial_condition = self._reparameterize(g0_mean, g0_logvar, sample=sample)
        generator_state = self.fc_ic_to_generator(initial_condition)

        controller_h0 = self.controller_encoder_h0.expand(-1, x.shape[0], -1).contiguous()
        controller_context, _ = self.controller_encoder(encoder_x, controller_h0)
        controller_context = self._controller_context_for_reconstruction(controller_context)
        controller_state = torch.zeros(
            x.shape[0],
            self.controller_dim,
            device=x.device,
            dtype=x.dtype,
        )
        factor_state = self.dropout(generator_state) if self.keep_prob < 1.0 else generator_state
        factors_t = self._factors_from_generator(factor_state)

        rates = []
        factors = []
        u_means = []
        u_logvars = []
        u_samples = []
        kl_u_terms = []

        prior_u_mean = self.u_prior_mean.expand(x.shape[0], self.inferred_input_dim)
        prior_u_logvar = self.u_prior_logvar.expand(x.shape[0], self.inferred_input_dim)

        for t in range(self.reconstruction_time_steps):
            controller_input = torch.cat((controller_context[:, t], factors_t), dim=-1)
            if self.keep_prob < 1.0:
                controller_input = self.dropout(controller_input)

            controller_state = self.controller(controller_input, controller_state)

            u_mean = self.fc_u_mean(controller_state)
            u_logvar = self._controller_posterior_logvar(self.fc_u_logvar(controller_state))
            inferred_input = self._reparameterize(u_mean, u_logvar, sample=sample)
            u_samples.append(inferred_input)
            if self.inferred_input_prior == "independent":
                kl_u_terms.append(
                    _kl_diag_gaussian(
                        u_mean,
                        u_logvar,
                        prior_u_mean,
                        prior_u_logvar,
                    ).mean()
                )

            generator_state = self.generator(inferred_input, generator_state)
            factor_state = self.dropout(generator_state) if self.keep_prob < 1.0 else generator_state
            factors_t = self._factors_from_generator(factor_state)
            log_rates_t = self.fc_log_rates(factors_t).clamp(
                min=self.log_rate_min,
                max=self.log_rate_max,
            )
            rates.append(torch.exp(log_rates_t))
            factors.append(factors_t)
            u_means.append(u_mean)
            u_logvars.append(u_logvar)

        prior_g0_mean = self.g0_prior_mean.expand_as(g0_mean)
        prior_g0_logvar = self.g0_prior_logvar.expand_as(g0_logvar)
        kl_g0 = _kl_diag_gaussian(g0_mean, g0_logvar, prior_g0_mean, prior_g0_logvar).mean()
        if self.inferred_input_prior == "autoregressive":
            kl_u = self._kl_autoregressive_u(
                torch.stack(u_means, dim=1),
                torch.stack(u_logvars, dim=1),
                torch.stack(u_samples, dim=1),
            )
        else:
            kl_u = torch.stack(kl_u_terms).sum()
        rates_tensor = torch.stack(rates, dim=1)
        factors_tensor = torch.stack(factors, dim=1)

        return ModelOutput(
            rates=rates_tensor,
            rates_unit="hz",
            latents=factors_tensor,
            reconstruction=rates_tensor,
            extras={
                "kl_g0": kl_g0,
                "kl_u": kl_u,
                "g0_mean": g0_mean,
                "g0_logvar": g0_logvar,
                "u_mean": torch.stack(u_means, dim=1),
                "u_logvar": torch.stack(u_logvars, dim=1),
            },
        )

    def _factors_from_generator(self, generator_state: Tensor) -> Tensor:
        weight = F.normalize(self.fc_factors.weight, dim=1)
        return F.linear(generator_state, weight, self.fc_factors.bias)

    def _ic_posterior_logvar(self, raw_logvar: Tensor) -> Tensor:
        logvar = torch.logaddexp(
            raw_logvar,
            raw_logvar.new_tensor(self.posterior_logvar_min),
        )
        return logvar.clamp(max=self.posterior_logvar_max)

    def _controller_posterior_logvar(self, raw_logvar: Tensor) -> Tensor:
        if (
            self.controller_posterior_logvar_min is None
            and self.controller_posterior_logvar_max is None
        ):
            return raw_logvar
        return raw_logvar.clamp(
            min=self.controller_posterior_logvar_min,
            max=self.controller_posterior_logvar_max,
        )

    def _controller_context_for_reconstruction(self, controller_context: Tensor) -> Tensor:
        if self.controller_lag > 0:
            fwd, bwd = torch.split(
                controller_context,
                self.controller_encoder_dim,
                dim=-1,
            )
            lag = self.controller_lag
            fwd = F.pad(fwd, (0, 0, lag, 0, 0, 0))[:, : self.n_time]
            bwd = F.pad(bwd, (0, 0, 0, lag, 0, 0))[:, -self.n_time :]
            controller_context = torch.cat([fwd, bwd], dim=-1)

        forward_steps = self.reconstruction_time_steps - int(controller_context.shape[1])
        if forward_steps <= 0:
            return controller_context[:, : self.reconstruction_time_steps]
        return F.pad(controller_context, (0, 0, 0, forward_steps, 0, 0))

    @staticmethod
    def _reparameterize(mean: Tensor, logvar: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        noise = torch.randn_like(mean)
        return mean + noise * torch.exp(0.5 * logvar)

    def _apply_coordinated_dropout_input(self, encoder_x: Tensor) -> Tensor:
        self._last_cd_grad_mask = None
        if not self.training or self.coordinated_dropout_rate <= 0.0:
            return encoder_x

        start = min(self.coordinated_dropout_ic_enc_seq_len, int(encoder_x.shape[1]))
        unmaskable = encoder_x[:, :start]
        maskable = encoder_x[:, start:]
        if maskable.numel() == 0:
            self._last_cd_grad_mask = torch.ones_like(encoder_x)
            return encoder_x

        keep_prob = 1.0 - self.coordinated_dropout_rate
        keep_mask = torch.bernoulli(torch.full_like(maskable, keep_prob))
        if self.coordinated_dropout_pass_rate > 0.0:
            pass_mask = torch.bernoulli(
                torch.full_like(maskable, self.coordinated_dropout_pass_rate)
            )
        else:
            pass_mask = torch.zeros_like(maskable)
        maskable_grad = torch.logical_or(keep_mask <= 0.0, pass_mask > 0.0).to(maskable.dtype)
        if start > 0:
            self._last_cd_grad_mask = torch.cat([torch.ones_like(unmaskable), maskable_grad], dim=1)
        else:
            self._last_cd_grad_mask = maskable_grad
        masked = maskable * keep_mask / keep_prob
        return torch.cat([unmaskable, masked], dim=1) if start > 0 else masked

    def _apply_coordinated_dropout_grad_mask(self, recon_terms: Tensor) -> Tensor:
        if not self.training or self._last_cd_grad_mask is None:
            return recon_terms
        mask = self._last_cd_grad_mask.to(device=recon_terms.device, dtype=recon_terms.dtype)
        if mask.shape[0] != recon_terms.shape[0]:
            raise ValueError(
                f"Coordinated-dropout mask batch {mask.shape[0]} does not match "
                f"reconstruction batch {recon_terms.shape[0]}."
            )
        if mask.shape[1] > recon_terms.shape[1] or mask.shape[2] > recon_terms.shape[2]:
            mask = mask[:, : recon_terms.shape[1], : recon_terms.shape[2]]
        pad_time = int(recon_terms.shape[1] - mask.shape[1])
        pad_neurons = int(recon_terms.shape[2] - mask.shape[2])
        if pad_time > 0 or pad_neurons > 0:
            mask = F.pad(mask, (0, pad_neurons, 0, pad_time), value=1.0)
        grad_terms = recon_terms * mask
        nograd_terms = (recon_terms * (1.0 - mask)).detach()
        return grad_terms + nograd_terms

    def _kl_autoregressive_u(
        self,
        posterior_mean: Tensor,
        posterior_logvar: Tensor,
        sample: Tensor,
    ) -> Tensor:
        log_two_pi = math.log(2.0 * math.pi)
        posterior_log_prob = -0.5 * (
            log_two_pi
            + posterior_logvar
            + (sample - posterior_mean).pow(2) / torch.exp(posterior_logvar)
        ).sum(dim=(1, 2))

        tau = torch.exp(self.u_prior_logtau).clamp_min(1.0e-6)
        alpha = torch.exp(-1.0 / tau)
        log_nvar = self.u_prior_logvar
        log_process_var0 = log_nvar - torch.log1p(-alpha.pow(2)).clamp_min(math.log(1.0e-8))
        prior_mean = alpha * torch.roll(sample, shifts=1, dims=1)
        prior_mean = torch.cat([torch.zeros_like(prior_mean[:, :1]), prior_mean[:, 1:]], dim=1)
        prior_logvar = torch.ones_like(sample) * log_nvar
        prior_logvar = torch.cat(
            [torch.ones_like(prior_logvar[:, :1]) * log_process_var0, prior_logvar[:, 1:]],
            dim=1,
        )
        prior_log_prob = -0.5 * (
            log_two_pi
            + prior_logvar
            + (sample - prior_mean).pow(2) / torch.exp(prior_logvar)
        ).sum(dim=(1, 2))
        return (posterior_log_prob - prior_log_prob).mean()

    def _l2_penalty(self) -> Tensor:
        gen = self.generator.weight_hh.norm(2) / self.generator.weight_hh.numel()
        con = self.controller.weight_hh.norm(2) / self.controller.weight_hh.numel()
        return self.l2_gen_scale * gen + self.l2_con_scale * con

    def _maybe_initialize_rate_bias(self, x: Tensor) -> None:
        if not self.initialize_log_rate_bias or bool(self._rate_bias_initialized.item()):
            return
        with torch.no_grad():
            mean_rate = x.mean(dim=(0, 1)) / max(self.dt, 1e-8)
            if mean_rate.shape[0] < self.readout_neurons:
                fill = mean_rate.mean().expand(self.readout_neurons - mean_rate.shape[0])
                mean_rate = torch.cat([mean_rate, fill], dim=0)
            mean_rate = mean_rate.clamp(
                min=math.exp(self.log_rate_min),
                max=math.exp(self.log_rate_max),
            )
            self.fc_log_rates.bias.copy_(torch.log(mean_rate))
            self._rate_bias_initialized.copy_(
                torch.tensor(True, device=self._rate_bias_initialized.device)
            )

    def initialize_log_rate_bias_from_counts(self, spikes: Tensor) -> None:
        if spikes.ndim != 3:
            raise ValueError(
                f"expected spikes with shape trials x time x neurons, got {tuple(spikes.shape)}"
            )
        if int(spikes.shape[-1]) != self.readout_neurons:
            raise ValueError(
                f"rate-bias initialization expected {self.readout_neurons} neurons, "
                f"got {int(spikes.shape[-1])}."
            )
        with torch.no_grad():
            mean_rate = spikes.to(
                dtype=self.fc_log_rates.bias.dtype,
                device=self.fc_log_rates.bias.device,
            ).mean(dim=(0, 1)) / max(self.dt, 1e-8)
            mean_rate = mean_rate.clamp(
                min=math.exp(self.log_rate_min),
                max=math.exp(self.log_rate_max),
            )
            self.fc_log_rates.bias.copy_(torch.log(mean_rate))
            self._rate_bias_initialized.copy_(
                torch.tensor(True, device=self._rate_bias_initialized.device)
            )

    def evaluation_adapter(self, task: str) -> EvaluationAdapter | None:
        if task == "synthetic" and self.prediction_samples > 1:
            return SyntheticEvaluationAdapter(use_predict_rates=True)
        if (
            task == "nlb"
            and self.output_neuron_start is not None
            and self.output_neurons is not None
        ):
            return _LFADSNLBDirectAdapter(
                output_neuron_start=int(self.output_neuron_start),
                output_neurons=int(self.output_neurons),
                dt=self.dt,
            )
        return None

    def _reconstruction_target(
        self,
        batch: Tensor | dict[str, Tensor],
        rates: Tensor,
    ) -> Tensor:
        if isinstance(batch, dict) and "reconstruction_spikes" in batch:
            target = batch["reconstruction_spikes"].to(device=rates.device, dtype=rates.dtype)
            if target.shape == rates.shape:
                return target
            if (
                target.ndim == rates.ndim
                and target.shape[0] == rates.shape[0]
                and target.shape[1] <= rates.shape[1]
                and target.shape[-1] == rates.shape[-1]
            ):
                return target
        target = observations_from_batch(batch)
        if isinstance(batch, dict) and "heldout_spikes" not in batch:
            raw = batch.get("raw_spikes")
            if raw is not None and raw.shape == target.shape:
                target = raw
        target = target.to(device=rates.device, dtype=rates.dtype)
        if target.shape == rates.shape:
            return target
        if (
            target.ndim == rates.ndim
            and target.shape[0] == rates.shape[0]
            and target.shape[1] <= rates.shape[1]
            and target.shape[-1] == rates.shape[-1]
        ):
            return target

        if target.shape[:-1] != rates.shape[:-1]:
            raise ValueError(
                f"LFADS target shape {tuple(target.shape)} is incompatible with "
                f"rates {tuple(rates.shape)}."
            )

        if isinstance(batch, dict) and target.shape[-1] < rates.shape[-1]:
            heldout = batch.get("heldout_spikes")
            if heldout is None:
                heldout = batch.get("raw_spikes")
            if heldout is not None and heldout.shape[:-1] == target.shape[:-1]:
                full_target = torch.cat(
                    [target, heldout.to(device=rates.device, dtype=rates.dtype)],
                    dim=-1,
                )
                if full_target.shape == rates.shape:
                    return full_target

        if target.shape[-1] > rates.shape[-1]:
            return target[..., : rates.shape[-1]]

        raise ValueError(
            f"LFADS cannot build a reconstruction target with shape {tuple(rates.shape)} "
            f"from batch target {tuple(target.shape)}."
        )

    @staticmethod
    def _rates_for_reconstruction_target(rates: Tensor, target: Tensor) -> Tensor:
        if rates.shape == target.shape:
            return rates
        if (
            target.ndim == rates.ndim
            and target.shape[0] == rates.shape[0]
            and target.shape[1] <= rates.shape[1]
            and target.shape[-1] == rates.shape[-1]
        ):
            return rates[:, : target.shape[1]]
        raise ValueError(
            f"LFADS target shape {tuple(target.shape)} is incompatible with "
            f"rates {tuple(rates.shape)}."
        )

    @staticmethod
    def _scheduled_weight(step: int, start: int, duration: int) -> float:
        if duration <= 0:
            return 1.0 if step >= start else 0.0
        return float(min(max(step - start, 0) / duration, 1.0))

    def _batch_dt(self, batch: Tensor | dict[str, Tensor], target: Tensor) -> Tensor:
        if isinstance(batch, dict) and "dt" in batch:
            dt = batch["dt"]
            while dt.ndim < target.ndim:
                dt = dt.unsqueeze(-1)
            return dt
        return torch.as_tensor(self.dt, device=target.device, dtype=target.dtype)


class _LFADSNLBDirectAdapter(EvaluationAdapter):
    task = "nlb"

    def __init__(
        self,
        *,
        output_neuron_start: int,
        output_neurons: int,
        dt: float,
        prediction_floor: float = 1e-9,
    ) -> None:
        self.output_neuron_start = int(output_neuron_start)
        self.output_neurons = int(output_neurons)
        self.dt = float(dt)
        self.prediction_floor = float(prediction_floor)

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader,
        device: torch.device,
    ) -> EvaluationResult:
        predictions: list[Tensor] = []
        targets: list[Tensor] = []
        latents: list[Tensor] = []
        start = self.output_neuron_start
        stop = start + self.output_neurons

        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                output = model(x)
                if output.rates is None:
                    raise RuntimeError("LFADS direct NLB evaluation requires rate outputs.")
                if output.rates.shape[-1] < stop:
                    raise ValueError(
                        f"LFADS readout has {output.rates.shape[-1]} neurons, "
                        f"but slice {start}:{stop} was requested."
                    )
                target = _nlb_target_from_batch(batch).to(
                    device=output.rates.device,
                    dtype=output.rates.dtype,
                )
                prediction = (output.rates[..., start:stop] * self.dt).clamp_min(
                    self.prediction_floor
                )
                if prediction.shape[1] > target.shape[1]:
                    prediction = prediction[:, : target.shape[1]]
                if prediction.shape != target.shape:
                    raise ValueError(
                        f"LFADS direct NLB prediction shape {tuple(prediction.shape)} "
                        f"differs from target {tuple(target.shape)}."
                    )
                predictions.append(prediction.detach().cpu())
                targets.append(target.detach().cpu())
                if output.latents is not None:
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


def _nlb_target_from_batch(batch: Tensor | dict[str, Tensor]) -> Tensor:
    if not isinstance(batch, dict):
        raise TypeError("NLB evaluation requires dict batches with heldout_spikes.")
    if "heldout_spikes" in batch:
        return batch["heldout_spikes"]
    if "raw_spikes" in batch:
        return batch["raw_spikes"]
    raise KeyError("NLB batch is missing heldout_spikes/raw_spikes.")


def _full_spikes_from_dataset(dataset: Any) -> Tensor | None:
    if dataset is None:
        return None
    base = getattr(dataset, "dataset", dataset)
    reconstruction = getattr(base, "reconstruction_spikes", None)
    if reconstruction is not None:
        return reconstruction
    heldin = getattr(base, "spikes", getattr(dataset, "spikes", None))
    heldout = getattr(base, "raw_spikes", None)
    if heldin is None:
        return None
    if heldout is None:
        return heldin
    if heldin.shape[:-1] != heldout.shape[:-1]:
        return heldin
    return torch.cat([heldin, heldout.to(device=heldin.device, dtype=heldin.dtype)], dim=-1)
