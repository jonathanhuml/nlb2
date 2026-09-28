"""LangevinFlow adapter for neural spike-count sequence modeling.

The encoder defaults to the current-bin alignment in Algorithm 1 of
arXiv:2507.11531v2. The reference code's lagged alignment is available explicitly.
The transition KL uses the actual Gaussian mean and log variance, correcting
the reference code's sampled-mean/variance inputs.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator
from torch import Tensor, nn
import torch.nn.functional as F

from nlb2.metrics import (
    EvaluationAdapter,
    EvaluationResult,
    NLBCoSmoothingAdapter,
    SyntheticEvaluationAdapter,
    compute_available_metrics,
)
from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.preprocessing import PreprocessedDataset
from nlb2.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


class FixupTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """One-layer Transformer decoder block with the upstream T-Fixup scaling."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fixup_initialization()

    def fixup_initialization(self) -> None:
        scale = 0.67 * (3.0 ** (-0.25))
        with torch.no_grad():
            self.linear1.weight.mul_(scale)
            self.linear2.weight.mul_(scale)
            self.self_attn.out_proj.weight.mul_(scale)


class CoupledOscillatorPotential(nn.Module):
    """Locally coupled oscillator potential over grouped latent coordinates."""

    def __init__(self, hidden_size: int, groups: int = 4, kernel_size: int = 3) -> None:
        super().__init__()
        if groups < 1:
            raise ValueError("groups must be positive.")
        if hidden_size % groups != 0:
            raise ValueError("hidden_size must be divisible by potential groups.")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("potential_kernel_size must be a positive odd integer.")

        self.hidden_size = int(hidden_size)
        self.groups = int(groups)
        self.kernel_size = int(kernel_size)
        self.channels_per_group = self.hidden_size // self.groups
        self.conv_z = nn.Parameter(torch.zeros(self.groups, self.groups, self.kernel_size))

    def forward(self, z: Tensor) -> Tensor:
        grouped = z.reshape(z.shape[0], self.groups, self.channels_per_group)
        weight = F.normalize(self.conv_z, p=2, dim=2, eps=1e-8)
        coupled = F.conv1d(
            grouped,
            weight,
            bias=None,
            stride=1,
            padding=self.kernel_size // 2,
        )
        interaction = coupled.bmm(grouped.transpose(1, 2))
        return interaction.sum(dim=(1, 2))


@BaseModelConfig.register
class LangevinFlowConfig(BaseModelConfig):
    """Config for the LangevinFlow sequential VAE."""

    name: Literal["langevin_flow"] = "langevin_flow"
    objective: str = "langevin_flow_elbo"
    hidden_size: int = 64
    initialization: Literal["nlb2", "upstream"] = "nlb2"
    encoder_input_alignment: Literal["current", "upstream_lagged"] = "current"
    output_neurons: Optional[int] = None
    output_mode: Literal["auto", "heldin", "heldin_heldout"] = "auto"
    fwd_steps: int = 0
    dropout: float = 0.05
    gamma: float = 0.55
    langevin_step: float = 0.01
    potential_groups: int = 4
    potential_kernel_size: int = 3
    transformer_heads: int = 2
    transformer_feedforward: int = 512
    coordinated_dropout_rate: float = 0.5
    kl_weight: float = 0.1
    kl_warmup_epochs: int = 500
    weight_decay_warmup_epochs: int = 500
    velocity_prior_var: float = 0.1
    log_rate_min: Optional[float] = -8.0
    log_rate_max: Optional[float] = 8.0
    posterior_logvar_min: Optional[float] = math.log(1e-4)
    posterior_logvar_max: Optional[float] = 5.0
    sample_train: bool = True
    sample_eval: bool = False
    prediction_samples: int = 1
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=3.0e-3,
            weight_decay=2.0e-5,
            gradient_clip=200.0,
        )
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "LangevinFlowConfig":
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if self.potential_groups < 1:
            raise ValueError("potential_groups must be positive.")
        if self.hidden_size % self.potential_groups != 0:
            raise ValueError("hidden_size must be divisible by potential_groups.")
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be positive.")
        if (3 * self.hidden_size) % self.transformer_heads != 0:
            raise ValueError("3 * hidden_size must be divisible by transformer_heads.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1).")
        if self.langevin_step <= 0.0:
            raise ValueError("langevin_step must be positive.")
        if self.potential_kernel_size < 1 or self.potential_kernel_size % 2 == 0:
            raise ValueError("potential_kernel_size must be a positive odd integer.")
        if not 0.0 < self.coordinated_dropout_rate <= 1.0:
            raise ValueError("coordinated_dropout_rate must be in (0, 1].")
        if self.kl_weight < 0.0:
            raise ValueError("kl_weight must be nonnegative.")
        if self.kl_warmup_epochs < 0:
            raise ValueError("kl_warmup_epochs must be nonnegative.")
        if self.weight_decay_warmup_epochs < 0:
            raise ValueError("weight_decay_warmup_epochs must be nonnegative.")
        if self.velocity_prior_var <= 0.0:
            raise ValueError("velocity_prior_var must be positive.")
        if self.prediction_samples < 1:
            raise ValueError("prediction_samples must be positive.")
        if self.output_neurons is not None and self.output_neurons < 1:
            raise ValueError("output_neurons must be positive when provided.")
        if self.fwd_steps < 0:
            raise ValueError("fwd_steps must be nonnegative.")
        return self

    def build(self, n_neurons: int, n_time: int) -> "LangevinFlow":
        output_neurons = self.output_neurons or n_neurons
        return self._build(n_neurons=n_neurons, n_time=n_time, output_neurons=output_neurons)

    def build_from_data(self, data: Any) -> "LangevinFlow":
        n_neurons = int(data.n_neurons)
        n_time = int(data.n_time)
        output_neurons = self.output_neurons
        fwd_steps = self.fwd_steps
        if output_neurons is None:
            output_neurons = n_neurons
            if self.output_mode != "heldin":
                train_dataset = data.train_dataset
                if train_dataset is None:
                    raise RuntimeError("DataModule.setup() must run before build_from_data().")
                while isinstance(train_dataset, PreprocessedDataset):
                    train_dataset = train_dataset.dataset
                heldout = getattr(train_dataset, "raw_spikes", None)
                heldin = getattr(train_dataset, "heldin_spikes", None)
                if heldout is not None and heldin is not None:
                    if heldin is not None and n_neurons >= int(heldin.shape[-1]) + int(
                        heldout.shape[-1]
                    ):
                        output_neurons = n_neurons
                    else:
                        output_neurons = n_neurons + int(heldout.shape[-1])
                    dataset_config = getattr(train_dataset, "config", None)
                    include_forward = bool(getattr(dataset_config, "include_forward", False))
                    heldin_forward = getattr(train_dataset, "heldin_forward_spikes", None)
                    heldout_forward = getattr(train_dataset, "heldout_forward_spikes", None)
                    if include_forward and (
                        heldin_forward is None or heldout_forward is None
                    ):
                        valid_dataset = getattr(data, "valid_dataset", None)
                        while isinstance(valid_dataset, PreprocessedDataset):
                            valid_dataset = valid_dataset.dataset
                        if valid_dataset is not None:
                            heldin_forward = getattr(
                                valid_dataset,
                                "heldin_forward_spikes",
                                heldin_forward,
                            )
                            heldout_forward = getattr(
                                valid_dataset,
                                "heldout_forward_spikes",
                                heldout_forward,
                            )
                    if include_forward and heldin_forward is not None and heldout_forward is not None:
                        fwd_steps = fwd_steps or int(heldin_forward.shape[1])
                elif self.output_mode == "heldin_heldout":
                    raise ValueError(
                        "output_mode='heldin_heldout' requires NLB held-in and held-out spikes."
                    )
        return self._build(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            fwd_steps=fwd_steps,
        )

    def _build(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        fwd_steps: int | None = None,
    ) -> "LangevinFlow":
        fwd_steps = self.fwd_steps if fwd_steps is None else int(fwd_steps)
        return LangevinFlow(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            hidden_size=self.hidden_size,
            initialization=self.initialization,
            encoder_input_alignment=self.encoder_input_alignment,
            fwd_steps=fwd_steps,
            dropout=self.dropout,
            gamma=self.gamma,
            langevin_step=self.langevin_step,
            potential_groups=self.potential_groups,
            potential_kernel_size=self.potential_kernel_size,
            transformer_heads=self.transformer_heads,
            transformer_feedforward=self.transformer_feedforward,
            coordinated_dropout_rate=self.coordinated_dropout_rate,
            kl_weight=self.kl_weight,
            kl_warmup_epochs=self.kl_warmup_epochs,
            weight_decay_warmup_epochs=self.weight_decay_warmup_epochs,
            velocity_prior_var=self.velocity_prior_var,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            posterior_logvar_min=self.posterior_logvar_min,
            posterior_logvar_max=self.posterior_logvar_max,
            sample_train=self.sample_train,
            sample_eval=self.sample_eval,
            prediction_samples=self.prediction_samples,
            objective=self.objective,
        )


class LangevinFlow(BaseDynamicsModel):
    """LangevinFlow sequential VAE for binned neural spike counts.

    ## When to use

    Use LangevinFlow as a nonlinear latent dynamics model for raw spike-count
    sequences. A GRU encoder updates short-range hidden state, latent position
    and velocity variables evolve through an underdamped Langevin step with a
    locally coupled oscillator potential, and a one-layer Transformer decoder
    reads the whole latent sequence into Poisson firing rates.

    ## Training Budget

    LangevinFlow needs longer training runs than the quick smoke-test settings
    used for development. For reproduction-style runs, prefer the YAML
    experiment budgets: the NLB configs use the released-scale epoch counts,
    and the Lorenz config is set to a longer default. Short runs such as 20
    epochs are useful only for checking that the loss and co-bps move in the
    right direction.

    ## Assumptions

    LangevinFlow expects nonnegative spike counts. On synthetic datasets the
    readout reconstructs the observed neurons. When built by `Experiment` on an
    NLB dataset, `output_mode: auto` sizes the readout to reconstruct held-in
    plus held-out training neurons and evaluates the held-out output slice.

    ## Reference Differences

    `encoder_input_alignment: current` follows Algorithm 1, line 9 in the
    [paper](https://arxiv.org/html/2507.11531v2), consuming each observed bin once.
    `upstream_lagged` reproduces the encoder indexing in the
    [released code](https://github.com/KingJamesSong/LangevinFlow_CCN/blob/main/nlb_lightning/models.py):
    bin 0 initializes the GRU and is consumed again for bin 1; the final
    observed bin enters only forward prediction steps. Paper and code disagree
    here, so alignment is explicit rather than a claim of exact reproduction.
    Both modes compute transition KL from the Gaussian mean and log variance,
    correcting the released code's sampled-mean/variance arguments to match
    the transition distribution in the paper's Equation 17.

    ## Outputs

    `forward` returns natural-space firing rates, concatenated
    `[position, velocity, hidden]` latent trajectories, and ELBO diagnostics in
    `extras`. `loss` computes Poisson reconstruction with a scheduled
    Langevin KL penalty and coordinated-dropout gradient masking.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        hidden_size: int = 64,
        initialization: Literal["nlb2", "upstream"] = "nlb2",
        fwd_steps: int = 0,
        dropout: float = 0.05,
        gamma: float = 0.55,
        langevin_step: float = 0.01,
        potential_groups: int = 4,
        potential_kernel_size: int = 3,
        transformer_heads: int = 2,
        transformer_feedforward: int = 512,
        coordinated_dropout_rate: float = 0.5,
        kl_weight: float = 0.1,
        kl_warmup_epochs: int = 500,
        weight_decay_warmup_epochs: int = 500,
        velocity_prior_var: float = 0.1,
        log_rate_min: float | None = -8.0,
        log_rate_max: float | None = 8.0,
        posterior_logvar_min: float | None = math.log(1e-4),
        posterior_logvar_max: float | None = 5.0,
        sample_train: bool = True,
        sample_eval: bool = False,
        prediction_samples: int = 1,
        objective: str = "langevin_flow_elbo",
        encoder_input_alignment: Literal["current", "upstream_lagged"] = "current",
    ) -> None:
        super().__init__()
        if encoder_input_alignment not in ("current", "upstream_lagged"):
            raise ValueError("encoder_input_alignment must be 'current' or 'upstream_lagged'.")
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.output_neurons = int(output_neurons)
        self.hidden_size = int(hidden_size)
        self.initialization = initialization
        self.encoder_input_alignment = encoder_input_alignment
        self.fwd_steps = int(fwd_steps)
        self.dropout_rate = float(dropout)
        self.gamma = float(gamma)
        self.langevin_step = float(langevin_step)
        self.coordinated_dropout_rate = float(coordinated_dropout_rate)
        self.kl_weight = float(kl_weight)
        self.kl_warmup_epochs = int(kl_warmup_epochs)
        self.weight_decay_warmup_epochs = int(weight_decay_warmup_epochs)
        self.velocity_prior_var = float(velocity_prior_var)
        self.log_rate_min = None if log_rate_min is None else float(log_rate_min)
        self.log_rate_max = None if log_rate_max is None else float(log_rate_max)
        self.posterior_logvar_min = (
            None if posterior_logvar_min is None else float(posterior_logvar_min)
        )
        self.posterior_logvar_max = (
            None if posterior_logvar_max is None else float(posterior_logvar_max)
        )
        self.sample_train = bool(sample_train)
        self.sample_eval = bool(sample_eval)
        self.prediction_samples = int(prediction_samples)
        self.objective = objective

        self.encoder = nn.GRUCell(input_size=self.n_neurons, hidden_size=self.hidden_size)
        self.linear_z_means = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_z_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_means = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_logvar = nn.Linear(self.hidden_size, self.hidden_size)
        self.decoder = FixupTransformerEncoderLayer(
            d_model=3 * self.hidden_size,
            nhead=int(transformer_heads),
            dim_feedforward=int(transformer_feedforward),
            dropout=self.dropout_rate,
        )
        self.potential = CoupledOscillatorPotential(
            hidden_size=self.hidden_size,
            groups=int(potential_groups),
            kernel_size=int(potential_kernel_size),
        )
        self.readout = nn.Linear(3 * self.hidden_size, self.output_neurons)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.register_buffer("_train_step", torch.zeros((), dtype=torch.long))

        if self.initialization == "nlb2":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.GRUCell):
                module.weight_ih.data.normal_(std=module.weight_ih.shape[1] ** -0.5)
                module.weight_hh.data.normal_(std=module.weight_hh.shape[1] ** -0.5)
                module.bias_ih.data.zero_()
                module.bias_hh.data.zero_()
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(std=module.in_features ** -0.5)
                module.bias.data.zero_()
        self.decoder.fixup_initialization()
        self.potential.conv_z.data.zero_()

    def forward(self, x: Tensor) -> ModelOutput:
        sample = self.sample_train if self.training else self.sample_eval
        return self._forward(x, sample=sample)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        if output.rates is None:
            raise RuntimeError("LangevinFlow forward output is missing rates.")
        log_rates = output.extras["log_rates"]
        target = self._reconstruction_target(batch, log_rates)
        target = target.to(device=self.device, dtype=log_rates.dtype)
        log_rates = log_rates[:, : target.shape[1], : target.shape[2]]
        finite = torch.isfinite(target)
        if not bool(finite.any()):
            raise ValueError("LangevinFlow loss requires at least one finite target count.")
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        recon = F.poisson_nll_loss(log_rates, safe_target, log_input=True, reduction="none")
        cd_mask = output.extras.get("coordinated_dropout_mask")
        if self.training and cd_mask is not None:
            cd_mask = self._pad_cd_mask(cd_mask, recon)
            recon = recon * cd_mask + (recon * (1.0 - cd_mask)).detach()
        recon_nll = recon[finite].mean()

        kl = output.extras["kl"]
        kl_weight = self._kl_weight(epoch) if self.training else 0.0
        total = recon_nll + kl_weight * kl
        if self.training:
            self._train_step.add_(1)

        return LossOutput(
            total=total,
            named_terms={
                "reconstruction_nll": recon_nll,
                "kl": kl,
                "kl_weight": kl_weight,
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            if self.prediction_samples <= 1:
                return self._forward(x, sample=self.sample_eval).rates
            log_rates = [
                self._forward(x, sample=True).extras["log_rates"]
                for _ in range(self.prediction_samples)
            ]
        finally:
            self.train(was_training)
        return torch.stack(log_rates, dim=0).mean(dim=0).exp()

    def evaluation_adapter(self, task: str) -> EvaluationAdapter | None:
        if task == "synthetic" and self.prediction_samples > 1:
            return SyntheticEvaluationAdapter(use_predict_rates=True)
        if task != "nlb":
            return None
        if self.output_neurons > self.n_neurons:
            return LangevinFlowNLBAdapter()
        return NLBCoSmoothingAdapter(feature_source="latents")

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> None:
        if self.weight_decay_warmup_epochs <= 0:
            return
        ramp = min(max(float(epoch) / float(self.weight_decay_warmup_epochs), 0.0), 1.0)
        for group in optimizer.param_groups:
            base = group.setdefault(
                "_langevin_flow_base_weight_decay",
                float(group.get("weight_decay", 0.0)),
            )
            group["weight_decay"] = float(base) * ramp

    def _forward(self, x: Tensor, sample: bool) -> ModelOutput:
        if x.ndim != 3:
            raise ValueError("LangevinFlow expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")
        if torch.any(x < 0):
            raise ValueError("LangevinFlow expects nonnegative spike-count observations.")

        x = x.to(device=self.device, dtype=next(self.parameters()).dtype)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        observ, cd_mask = self._coordinated_dropout(x)
        total_steps = self.n_time + self.fwd_steps
        hidden = self.dropout(self.encoder(observ[:, 0]))
        z_mu = self.linear_z_means(hidden)
        z_logvar = self._clamp_logvar(self.linear_z_logvar(hidden))
        v_mu = self.linear_v_means(hidden)
        v_logvar = self._clamp_logvar(self.linear_v_logvar(hidden))
        z = self._reparameterize(z_mu, z_logvar, sample=sample)
        v = self._reparameterize(v_mu, v_logvar, sample=sample)
        kl = self._kl_diag_gaussian(z_mu, z_logvar, prior_var=1.0)
        kl = kl + self._kl_diag_gaussian(v_mu, v_logvar, prior_var=1.0)

        latent_steps = [torch.cat([z, v, hidden], dim=1)]
        noise_var = max(2.0 * self.gamma, 1e-8)
        noise_std = math.sqrt(noise_var)
        for t in range(1, total_steps):
            if t < self.n_time:
                input_index = t if self.encoder_input_alignment == "current" else t - 1
                hidden_input = observ[:, input_index]
            else:
                hidden_input = observ[:, -1]
            hidden = self.dropout(self.encoder(hidden_input, hidden))
            z, v, step_kl = self._langevin_step(z, v, noise_std, sample)
            kl = kl + step_kl
            latent_steps.append(torch.cat([z, v, hidden], dim=1))

        latents = torch.stack(latent_steps, dim=1)
        decoded = self.decoder(latents)
        log_rates = self.readout(decoded)
        if self.log_rate_min is not None or self.log_rate_max is not None:
            log_rates = log_rates.clamp(min=self.log_rate_min, max=self.log_rate_max)
        rates = torch.exp(log_rates)
        return ModelOutput(
            rates=rates,
            latents=latents,
            extras={
                "log_rates": log_rates,
                "kl": kl,
                "coordinated_dropout_mask": cd_mask,
            },
        )

    def _langevin_step(
        self,
        z: Tensor,
        v: Tensor,
        noise_std: float,
        sample: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            z_for_grad = z.clone().requires_grad_(True)
            energy = self.potential(z_for_grad)
            force = torch.autograd.grad(
                energy.sum(),
                z_for_grad,
                create_graph=self.training,
                retain_graph=self.training,
            )[0]

        z_next = z_for_grad + self.langevin_step * v
        v_half = v - self.langevin_step * force
        v_mean = (1.0 - self.gamma) * v_half
        if sample:
            v_next = v_mean + torch.randn_like(v_mean) * noise_std
        else:
            v_next = v_mean
        step_kl = self._kl_diag_gaussian(
            v_mean,
            torch.full_like(v_mean, math.log(noise_std ** 2)),
            prior_var=self.velocity_prior_var,
        )
        return z_next, v_next, step_kl

    def _coordinated_dropout(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        if not self.training or self.coordinated_dropout_rate >= 1.0:
            return x, None
        keep_prob = self.coordinated_dropout_rate
        keep = torch.bernoulli(torch.full_like(x, keep_prob))
        pass_mask = torch.bernoulli(torch.full_like(x, 1.0 - keep_prob))
        grad_mask = torch.logical_or(keep == 0.0, pass_mask == 1.0).to(dtype=x.dtype)
        return x * keep / keep_prob, grad_mask

    def _reconstruction_target(
        self,
        batch: Tensor | dict[str, Tensor],
        log_rates: Tensor,
    ) -> Tensor:
        observed = observations_from_batch(batch)
        if isinstance(batch, dict):
            reconstruction = batch.get("reconstruction_spikes")
            if reconstruction is not None and reconstruction.shape == log_rates.shape:
                return reconstruction
            raw = batch.get("raw_spikes")
            if "heldout_spikes" not in batch and raw is not None and raw.shape == observed.shape:
                observed = raw
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            heldout = batch["heldout_spikes"]
            heldin = batch.get("heldin_spikes")
            if heldin is not None:
                observed = heldin
            total_neurons = observed.shape[-1] + heldout.shape[-1]
            if log_rates.shape[-1] >= total_neurons:
                target = torch.cat([observed, heldout], dim=-1)
                if (
                    "heldin_forward_spikes" in batch
                    and "heldout_forward_spikes" in batch
                    and log_rates.shape[1] > target.shape[1]
                ):
                    forward = torch.cat(
                        [batch["heldin_forward_spikes"], batch["heldout_forward_spikes"]],
                        dim=-1,
                    )
                    target = torch.cat([target, forward], dim=1)
                return target
        return observed

    @staticmethod
    def _pad_cd_mask(mask: Tensor, loss: Tensor) -> Tensor:
        time_pad = loss.shape[1] - mask.shape[1]
        neuron_pad = loss.shape[2] - mask.shape[2]
        if time_pad < 0 or neuron_pad < 0:
            return mask[:, : loss.shape[1], : loss.shape[2]]
        return F.pad(mask, (0, neuron_pad, 0, time_pad), value=1.0)

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_warmup_epochs <= 0:
            return self.kl_weight
        progress = max(float(epoch) / float(self.kl_warmup_epochs), 0.0)
        return self.kl_weight * min(progress, 1.0)

    def _clamp_logvar(self, logvar: Tensor) -> Tensor:
        if self.posterior_logvar_min is None and self.posterior_logvar_max is None:
            return logvar
        return logvar.clamp(min=self.posterior_logvar_min, max=self.posterior_logvar_max)

    @staticmethod
    def _reparameterize(mean: Tensor, logvar: Tensor, sample: bool) -> Tensor:
        if not sample:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    @staticmethod
    def _kl_diag_gaussian(mean: Tensor, logvar: Tensor, prior_var: float) -> Tensor:
        prior_logvar = math.log(prior_var)
        var = torch.exp(logvar)
        kl = 0.5 * (prior_logvar - logvar - 1.0 + (mean.pow(2) + var) / prior_var)
        return kl.sum() / mean.shape[0]


class LangevinFlowNLBAdapter(EvaluationAdapter):
    """Direct held-out NLB scorer for LangevinFlow full readouts."""

    task = "nlb"

    def evaluate(
        self,
        model: BaseDynamicsModel,
        loader: Any,
        device: torch.device,
    ) -> EvaluationResult:
        predictions: list[Tensor] = []
        targets: list[Tensor] = []

        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                x = observations_from_batch(batch)
                if not isinstance(batch, dict) or "heldout_spikes" not in batch:
                    raise TypeError("NLB evaluation requires heldout_spikes in dict batches.")
                rates = model.predict_rates(x)
                target = batch["heldout_spikes"]
                heldin = batch.get("heldin_spikes")
                n_heldin = x.shape[-1] if heldin is None else heldin.shape[-1]
                n_heldout = target.shape[-1]
                pred = rates[:, : target.shape[1], n_heldin : n_heldin + n_heldout]
                if pred.shape != target.shape:
                    raise ValueError(
                        "LangevinFlow direct NLB predictions have shape "
                        f"{tuple(pred.shape)}, expected {tuple(target.shape)}."
                    )
                predictions.append(pred.detach().cpu())
                targets.append(target.detach().cpu())

        pred_dict: dict[str, Tensor] = {"rates": torch.cat(predictions, dim=0)}
        target_dict: dict[str, Tensor] = {"spikes": torch.cat(targets, dim=0)}
        metrics = compute_available_metrics(pred_dict, target_dict)
        return EvaluationResult(
            metrics=metrics,
            predictions={key: value.numpy() for key, value in pred_dict.items()},
            targets={key: value.numpy() for key, value in target_dict.items()},
        )
