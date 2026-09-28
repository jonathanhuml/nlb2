"""STNDT adapter for spatiotemporal masked spike-count modeling."""

from __future__ import annotations

from copy import deepcopy
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
    compute_available_metrics,
)
from nlb2.models.base import (
    BaseDynamicsModel,
    BaseModelConfig,
    EnsembleDynamicsModel,
    OptimizationConfig,
)
from nlb2.types import LossOutput, ModelOutput, move_batch_to_device, observations_from_batch


UNMASKED_LABEL = -100.0


@BaseModelConfig.register
class STNDTConfig(BaseModelConfig):
    """Config for the Spatiotemporal Neural Data Transformer adapter."""

    name: Literal["stndt"] = "stndt"
    objective: str = "stndt_masked_poisson_nll"
    ensemble: bool = False
    ensemble_size: int = 2
    output_neurons: Optional[int] = None
    output_mode: Literal["auto", "heldin", "heldin_heldout"] = "auto"
    fwd_steps: int = 0
    context_forward: int = 4
    context_backward: int = 8
    context_wrap_initial: bool = False
    full_context: bool = False
    hidden_size: int = 128
    dropout: float = 0.1
    dropout_rates: float = 0.2
    dropout_embedding: float = 0.2
    num_heads: int = 2
    num_layers: int = 6
    activation: Literal["relu", "gelu"] = "relu"
    linear_embedder: bool = False
    embed_dim: Literal[0, 1] = 1
    learnable_position: bool = False
    max_spike_count: int = 20
    lograte: bool = True
    log_rate_min: float = -8.0
    log_rate_max: float = 8.0
    spike_log_init: bool = False
    fixup_init: bool = True
    pre_norm: bool = True
    scale_norm: bool = False
    decoder_layers: int = 1
    position_offset: bool = False
    mask_ratio: float = 0.25
    mask_mode: Literal["full", "timestep", "neuron", "timestep_only"] = "full"
    mask_token_ratio: float = 1.0
    mask_random_ratio: float = 0.5
    mask_max_span: int = 1
    mask_span_expand_prob: float = 0.0
    mask_span_ramp_start: int = 0
    mask_span_ramp_end: int = 0
    use_zero_mask: bool = True
    topk_loss_fraction: float = 1.0
    do_contrast: bool = True
    contrast_mask_ratio: float = 0.05
    contrast_mask_mode: Literal["full", "timestep", "neuron", "timestep_only"] = "full"
    contrast_mask_token_ratio: float = 0.5
    contrast_mask_random_ratio: float = 0.5
    contrast_mask_max_span: int = 1
    contrast_mask_span_expand_prob: float = 0.0
    contrast_mask_span_ramp_start: int = 0
    contrast_mask_span_ramp_end: int = 0
    temperature: float = 0.07
    contrast_lambda: float = 0.1
    use_contrast_projector: bool = False
    linear_projector: bool = True
    contrast_layer: Literal["embedder", "decoder"] = "embedder"
    nlb_decoder: Literal["direct", "latents"] = "direct"
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="gradient",
            optimizer="Adam",
            lr=1e-3,
            weight_decay=5e-5,
            gradient_clip=200.0,
        )
    )

    @model_validator(mode="after")
    def validate_dimensions(self) -> "STNDTConfig":
        if self.output_neurons is not None and self.output_neurons < 1:
            raise ValueError("output_neurons must be positive when provided.")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if self.ensemble and self.ensemble_size < 2:
            raise ValueError("ensemble_size must be at least 2 when ensemble=true.")
        if self.fwd_steps < 0:
            raise ValueError("fwd_steps must be nonnegative.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.decoder_layers < 1:
            raise ValueError("decoder_layers must be positive.")
        if self.max_spike_count < 1:
            raise ValueError("max_spike_count must be positive.")
        for name in ("dropout", "dropout_rates", "dropout_embedding"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        for name in (
            "mask_ratio",
            "mask_token_ratio",
            "mask_random_ratio",
            "contrast_mask_ratio",
            "contrast_mask_token_ratio",
            "contrast_mask_random_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.mask_max_span < 1:
            raise ValueError("mask_max_span must be positive.")
        if self.contrast_mask_max_span < 1:
            raise ValueError("contrast_mask_max_span must be positive.")
        if not 0.0 <= self.mask_span_expand_prob <= 1.0:
            raise ValueError("mask_span_expand_prob must be in [0, 1].")
        if not 0.0 <= self.contrast_mask_span_expand_prob <= 1.0:
            raise ValueError("contrast_mask_span_expand_prob must be in [0, 1].")
        if self.mask_span_ramp_start < 0 or self.mask_span_ramp_end < 0:
            raise ValueError("mask_span_ramp_start/end must be nonnegative.")
        if self.contrast_mask_span_ramp_start < 0 or self.contrast_mask_span_ramp_end < 0:
            raise ValueError("contrast_mask_span_ramp_start/end must be nonnegative.")
        if not 0.0 < self.topk_loss_fraction <= 1.0:
            raise ValueError("topk_loss_fraction must be in (0, 1].")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.contrast_lambda < 0.0:
            raise ValueError("contrast_lambda must be nonnegative.")
        return self

    def build(self, n_neurons: int, n_time: int) -> BaseDynamicsModel:
        output_neurons = self.output_neurons or n_neurons
        return self._build_configured(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            fwd_steps=self.fwd_steps,
        )

    def build_from_data(self, data: Any) -> BaseDynamicsModel:
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
                dataset = getattr(train_dataset, "dataset", train_dataset)
                heldin = getattr(dataset, "heldin_spikes", None)
                heldout = getattr(dataset, "raw_spikes", None)
                if heldin is not None and heldout is not None:
                    dataset_config = getattr(dataset, "config", None)
                    input_mode = getattr(dataset_config, "input_mode", None)
                    if input_mode != "full_observed":
                        output_neurons = n_neurons + int(heldout.shape[-1])
                    heldin_forward = getattr(dataset, "heldin_forward_spikes", None)
                    heldout_forward = getattr(dataset, "heldout_forward_spikes", None)
                    if heldin_forward is not None and heldout_forward is not None:
                        fwd_steps = fwd_steps or int(heldin_forward.shape[1])
                elif self.output_mode == "heldin_heldout":
                    raise ValueError(
                        "output_mode='heldin_heldout' requires held-in and held-out dataset channels."
                    )
        return self._build_configured(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            fwd_steps=fwd_steps,
        )

    def _build_configured(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        fwd_steps: int,
    ) -> BaseDynamicsModel:
        if not self.ensemble:
            return self._build(
                n_neurons=n_neurons,
                n_time=n_time,
                output_neurons=output_neurons,
                fwd_steps=fwd_steps,
            )
        members = [
            self._build(
                n_neurons=n_neurons,
                n_time=n_time,
                output_neurons=output_neurons,
                fwd_steps=fwd_steps,
            )
            for _ in range(self.ensemble_size)
        ]
        return EnsembleDynamicsModel(
            members=members,
            objective=f"ensemble_{self.objective}",
        )

    def _build(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        fwd_steps: int,
    ) -> "STNDT":
        return STNDT(
            n_neurons=n_neurons,
            n_time=n_time,
            output_neurons=output_neurons,
            fwd_steps=fwd_steps,
            context_forward=self.context_forward,
            context_backward=self.context_backward,
            context_wrap_initial=self.context_wrap_initial,
            full_context=self.full_context,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            dropout_rates=self.dropout_rates,
            dropout_embedding=self.dropout_embedding,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            activation=self.activation,
            linear_embedder=self.linear_embedder,
            embed_dim=self.embed_dim,
            learnable_position=self.learnable_position,
            max_spike_count=self.max_spike_count,
            lograte=self.lograte,
            log_rate_min=self.log_rate_min,
            log_rate_max=self.log_rate_max,
            spike_log_init=self.spike_log_init,
            fixup_init=self.fixup_init,
            pre_norm=self.pre_norm,
            scale_norm=self.scale_norm,
            decoder_layers=self.decoder_layers,
            position_offset=self.position_offset,
            mask_ratio=self.mask_ratio,
            mask_mode=self.mask_mode,
            mask_token_ratio=self.mask_token_ratio,
            mask_random_ratio=self.mask_random_ratio,
            mask_max_span=self.mask_max_span,
            mask_span_expand_prob=self.mask_span_expand_prob,
            mask_span_ramp_start=self.mask_span_ramp_start,
            mask_span_ramp_end=self.mask_span_ramp_end,
            use_zero_mask=self.use_zero_mask,
            topk_loss_fraction=self.topk_loss_fraction,
            do_contrast=self.do_contrast,
            contrast_mask_ratio=self.contrast_mask_ratio,
            contrast_mask_mode=self.contrast_mask_mode,
            contrast_mask_token_ratio=self.contrast_mask_token_ratio,
            contrast_mask_random_ratio=self.contrast_mask_random_ratio,
            contrast_mask_max_span=self.contrast_mask_max_span,
            contrast_mask_span_expand_prob=self.contrast_mask_span_expand_prob,
            contrast_mask_span_ramp_start=self.contrast_mask_span_ramp_start,
            contrast_mask_span_ramp_end=self.contrast_mask_span_ramp_end,
            temperature=self.temperature,
            contrast_lambda=self.contrast_lambda,
            use_contrast_projector=self.use_contrast_projector,
            linear_projector=self.linear_projector,
            contrast_layer=self.contrast_layer,
            nlb_decoder=self.nlb_decoder,
            objective=self.objective,
        )


class STNDT(BaseDynamicsModel):
    """Spatiotemporal Neural Data Transformer for binned spike counts.

    ## When to use

    Use STNDT when NDT-style masked count modeling should also learn spatial
    dependencies between neurons. The adapter ports the PyTorch STNDT design:
    temporal self-attention over time, spatial self-attention over neurons, a
    spatially mixed temporal stream, and optional SimCLR-style contrastive
    consistency between two independently masked views.

    ## Assumptions

    STNDT expects raw nonnegative spike counts. On synthetic datasets it
    reconstructs the observed neurons. When built by `Experiment` on an NLB
    dataset with `output_mode: auto`, it expands the readout to held-in plus
    held-out training neurons, feeds zeros for unavailable held-out inputs, and
    scores the held-out output slice with co-smoothing bits/spike.

    ## Outputs

    `forward` returns natural-space rates, spatiotemporal transformer factors,
    and masking/contrast diagnostics in `extras`. Training uses masked held-in
    entries plus any available NLB held-out/forward targets; evaluation uses all
    available target entries.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        output_neurons: int,
        fwd_steps: int = 0,
        context_forward: int = 4,
        context_backward: int = 8,
        context_wrap_initial: bool = False,
        full_context: bool = False,
        hidden_size: int = 128,
        dropout: float = 0.1,
        dropout_rates: float = 0.2,
        dropout_embedding: float = 0.2,
        num_heads: int = 2,
        num_layers: int = 6,
        activation: str = "relu",
        linear_embedder: bool = False,
        embed_dim: int = 1,
        learnable_position: bool = False,
        max_spike_count: int = 20,
        lograte: bool = True,
        log_rate_min: float = -8.0,
        log_rate_max: float = 8.0,
        spike_log_init: bool = False,
        fixup_init: bool = True,
        pre_norm: bool = True,
        scale_norm: bool = False,
        decoder_layers: int = 1,
        position_offset: bool = False,
        mask_ratio: float = 0.25,
        mask_mode: str = "full",
        mask_token_ratio: float = 1.0,
        mask_random_ratio: float = 0.5,
        mask_max_span: int = 1,
        mask_span_expand_prob: float = 0.0,
        mask_span_ramp_start: int = 0,
        mask_span_ramp_end: int = 0,
        use_zero_mask: bool = True,
        topk_loss_fraction: float = 1.0,
        do_contrast: bool = True,
        contrast_mask_ratio: float = 0.05,
        contrast_mask_mode: str = "full",
        contrast_mask_token_ratio: float = 0.5,
        contrast_mask_random_ratio: float = 0.5,
        contrast_mask_max_span: int = 1,
        contrast_mask_span_expand_prob: float = 0.0,
        contrast_mask_span_ramp_start: int = 0,
        contrast_mask_span_ramp_end: int = 0,
        temperature: float = 0.07,
        contrast_lambda: float = 0.1,
        use_contrast_projector: bool = False,
        linear_projector: bool = True,
        contrast_layer: str = "embedder",
        nlb_decoder: str = "direct",
        objective: str = "stndt_masked_poisson_nll",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        self.output_neurons = int(output_neurons)
        self.fwd_steps = int(fwd_steps)
        self.total_time = self.n_time + self.fwd_steps
        self.context_forward = int(context_forward)
        self.context_backward = int(context_backward)
        self.context_wrap_initial = bool(context_wrap_initial)
        self.full_context = bool(full_context)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        self.dropout_rates = float(dropout_rates)
        self.dropout_embedding = float(dropout_embedding)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.activation_name = str(activation)
        self.linear_embedder = bool(linear_embedder)
        self.embed_dim = int(embed_dim)
        self.learnable_position = bool(learnable_position)
        self.max_spike_count = int(max_spike_count)
        self.lograte = bool(lograte)
        self.log_rate_min = float(log_rate_min)
        self.log_rate_max = float(log_rate_max)
        self.spike_log_init = bool(spike_log_init)
        self.fixup_init = bool(fixup_init)
        self.pre_norm = bool(pre_norm)
        self.scale_norm = bool(scale_norm)
        self.decoder_layers = int(decoder_layers)
        self.position_offset = bool(position_offset)
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = str(mask_mode)
        self.mask_token_ratio = float(mask_token_ratio)
        self.mask_random_ratio = float(mask_random_ratio)
        self.mask_max_span = int(mask_max_span)
        self.mask_span_expand_prob = float(mask_span_expand_prob)
        self.mask_span_ramp_start = int(mask_span_ramp_start)
        self.mask_span_ramp_end = int(mask_span_ramp_end)
        self.use_zero_mask = bool(use_zero_mask)
        self.topk_loss_fraction = float(topk_loss_fraction)
        self.do_contrast = bool(do_contrast)
        self.contrast_mask_ratio = float(contrast_mask_ratio)
        self.contrast_mask_mode = str(contrast_mask_mode)
        self.contrast_mask_token_ratio = float(contrast_mask_token_ratio)
        self.contrast_mask_random_ratio = float(contrast_mask_random_ratio)
        self.contrast_mask_max_span = int(contrast_mask_max_span)
        self.contrast_mask_span_expand_prob = float(contrast_mask_span_expand_prob)
        self.contrast_mask_span_ramp_start = int(contrast_mask_span_ramp_start)
        self.contrast_mask_span_ramp_end = int(contrast_mask_span_ramp_end)
        self.temperature = float(temperature)
        self.contrast_lambda = float(contrast_lambda)
        self.use_contrast_projector = bool(use_contrast_projector)
        self.linear_projector = bool(linear_projector)
        self.contrast_layer = str(contrast_layer)
        self.nlb_decoder = str(nlb_decoder)
        self.objective = objective
        self.training_epoch = 0

        if self.n_neurons < 1 or self.output_neurons < 1:
            raise ValueError("n_neurons and output_neurons must be positive.")
        if self.n_neurons > self.output_neurons:
            raise ValueError("output_neurons must be at least n_neurons.")
        if self.n_time < 1 or self.total_time < 1:
            raise ValueError("n_time and total_time must be positive.")
        if self.embed_dim not in (0, 1):
            raise ValueError("STNDT embed_dim must be 0 or 1.")
        if self.output_neurons % self.num_heads != 0:
            raise ValueError(
                "STNDT temporal model dimension must be divisible by num_heads; got "
                f"output_neurons={self.output_neurons}, num_heads={self.num_heads}."
            )
        if self.total_time % self.num_heads != 0:
            raise ValueError(
                "STNDT spatial model dimension must be divisible by num_heads; got "
                f"total_time={self.total_time}, num_heads={self.num_heads}."
            )

        if self.linear_embedder:
            self.embedder: nn.Module = nn.Linear(self.output_neurons, self.output_neurons)
            self.spatial_embedder: nn.Module = nn.Linear(self.total_time, self.total_time)
        elif self.embed_dim == 0:
            self.embedder = nn.Identity()
            self.spatial_embedder = nn.Identity()
        else:
            self.embedder = SpikeEmbedding(self.max_spike_count + 2)
            self.spatial_embedder = SpikeEmbedding(self.max_spike_count + 2)

        self.temporal_scale = math.sqrt(self.output_neurons)
        self.spatial_scale = math.sqrt(self.total_time)
        self.temporal_position = PositionalEncoding(
            sequence_length=self.total_time,
            model_dim=self.output_neurons,
            dropout=self.dropout_embedding,
            learnable=self.learnable_position,
            offset=self.position_offset,
        )
        self.spatial_position = PositionalEncoding(
            sequence_length=self.output_neurons,
            model_dim=self.total_time,
            dropout=self.dropout_embedding,
            learnable=self.learnable_position,
            offset=self.position_offset,
        )
        self.encoder = SpatiotemporalTransformerEncoder(
            model_dim=self.output_neurons,
            spatial_dim=self.total_time,
            num_heads=self.num_heads,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            activation=self.activation_name,
            num_layers=self.num_layers,
            pre_norm=self.pre_norm,
            scale_norm=self.scale_norm,
        )
        self.rate_dropout = nn.Dropout(self.dropout_rates)
        self.decoder = self._build_decoder()
        self.projector = self._build_projector(self.output_neurons)
        self.spatial_projector = self._build_projector(self.total_time)
        self._attn_masks: dict[str, Tensor | None] = {}

        self.init_weights()
        if self.fixup_init:
            self.fixup_initialization()

    def set_training_epoch(self, epoch: int) -> None:
        self.training_epoch = int(epoch)

    def forward(self, x: Tensor) -> ModelOutput:
        return self._forward(x, should_mask=self.training)

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        del epoch
        target = self._reconstruction_target(batch, output.extras["log_rates"])
        target = target.to(device=self.device, dtype=output.extras["log_rates"].dtype)
        log_rates = output.extras["log_rates"][:, : target.shape[1], : target.shape[2]]
        rates = output.rates[:, : target.shape[1], : target.shape[2]]
        finite = torch.isfinite(target)
        if not finite.any():
            raise ValueError("STNDT loss requires at least one finite reconstruction target.")
        safe_target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        loss_mask = self._loss_mask_for_target(batch, output, target).to(target.device)
        loss_mask = loss_mask & finite

        if self.lograte:
            per_entry = F.poisson_nll_loss(
                log_rates,
                safe_target,
                log_input=True,
                full=False,
                reduction="none",
            )
        else:
            rates = rates.clamp_min(1e-8)
            per_entry = rates - safe_target * torch.log(rates)

        masked = per_entry[loss_mask]
        if masked.numel() == 0:
            masked = per_entry[finite].reshape(-1)
        if self.topk_loss_fraction < 1.0 and masked.numel() > 1:
            k = max(1, int(masked.numel() * self.topk_loss_fraction))
            masked = torch.topk(masked, k=k).values

        poisson_loss = masked.mean()
        contrast_loss = output.extras.get("contrast_loss", poisson_loss.new_zeros(()))
        total = poisson_loss + contrast_loss
        return LossOutput(
            total=total,
            named_terms={
                "masked_poisson_nll": poisson_loss,
                "contrast_loss": contrast_loss,
                "mask_fraction": loss_mask.float().mean(),
            },
            objective=self.objective,
        )

    @torch.no_grad()
    def predict_rates(self, x: Tensor) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            return self._forward(x, should_mask=False).rates
        finally:
            self.train(was_training)

    def evaluation_adapter(self, task: str) -> EvaluationAdapter | None:
        if task != "nlb":
            return None
        if self.nlb_decoder == "direct":
            return STNDTNLBAdapter()
        return NLBCoSmoothingAdapter(feature_source="latents")

    def _forward(self, x: Tensor, should_mask: bool) -> ModelOutput:
        self._validate_input(x)
        x = x.to(device=self.device, dtype=next(self.parameters()).dtype)
        masked_observed, labels, observed_loss_mask = self._mask_observations(
            x,
            should_mask=should_mask,
            contrast=False,
        )
        full_input = self._pad_observed(masked_observed)
        encoded, temporal_embedded, spatial_embedded, layer_weights = self._encode(
            full_input,
            return_weights=False,
        )
        decoded = self.decoder(self.rate_dropout(encoded))
        log_rates = decoded.permute(1, 0, 2)
        if self.lograte:
            log_rates = log_rates.clamp(self.log_rate_min, self.log_rate_max)
            rates = torch.exp(log_rates)
        else:
            rates = log_rates.clamp_min(0.0)
            log_rates = torch.log(rates.clamp_min(1e-8))

        contrast_loss = log_rates.new_zeros(())
        if self.training and self.do_contrast and self.contrast_lambda > 0.0:
            contrast_loss = self._contrast_loss(x)

        return ModelOutput(
            rates=rates,
            latents=encoded.permute(1, 0, 2),
            extras={
                "log_rates": log_rates,
                "mask_labels": labels,
                "observed_loss_mask": observed_loss_mask,
                "contrast_loss": contrast_loss,
                "temporal_embedding": temporal_embedded.permute(1, 0, 2),
                "spatial_embedding": spatial_embedded.permute(1, 0, 2),
                "layer_weights": layer_weights,
            },
        )

    def _encode(
        self,
        x: Tensor,
        return_outputs: bool = False,
        return_weights: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, list[tuple[Tensor, Tensor]] | None]:
        temporal = x.permute(1, 0, 2)
        temporal = self.embedder(temporal) * self.temporal_scale
        temporal = self.temporal_position(temporal)
        spatial = x.permute(2, 0, 1)
        spatial = self.spatial_embedder(spatial) * self.spatial_scale
        spatial = self.spatial_position(spatial)
        temporal_mask = self._get_or_generate_context_mask(
            size=temporal.shape[0],
            device=temporal.device,
            dtype=temporal.dtype,
        )
        encoded, _, weights = self.encoder(
            temporal,
            spatial,
            temporal_mask=temporal_mask,
            spatial_mask=None,
            return_outputs=return_outputs,
            return_weights=return_weights,
        )
        return encoded, temporal, spatial, weights

    def _contrast_loss(self, x: Tensor) -> Tensor:
        view1, _, _ = self._mask_observations(x, should_mask=True, contrast=True)
        view2, _, _ = self._mask_observations(x, should_mask=True, contrast=True)
        full1 = self._pad_observed(view1)
        full2 = self._pad_observed(view2)
        enc1, temporal1, spatial1, _ = self._encode(full1)
        enc2, temporal2, spatial2, _ = self._encode(full2)
        dec1 = self.decoder(self.rate_dropout(enc1))
        dec2 = self.decoder(self.rate_dropout(enc2))

        if self.contrast_layer == "decoder":
            out1 = dec1
            out2 = dec2
            projector = self.projector
            losses = [self._info_nce_from_sequence(out1, out2, projector)]
        else:
            losses = [
                self._info_nce_from_sequence(temporal1, temporal2, self.projector),
                self._info_nce_from_sequence(spatial1, spatial2, self.spatial_projector),
            ]
        return torch.stack(losses).sum() * self.contrast_lambda

    def _info_nce_from_sequence(self, out1: Tensor, out2: Tensor, projector: nn.Module) -> Tensor:
        out1 = projector(out1).permute(1, 0, 2).flatten(start_dim=1)
        out2 = projector(out2).permute(1, 0, 2).flatten(start_dim=1)
        features = torch.cat([out1, out2], dim=0)
        batch_size = out1.shape[0]
        labels = torch.cat(
            [torch.arange(batch_size, device=features.device) for _ in range(2)],
            dim=0,
        )
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        features = F.normalize(features, dim=1)
        similarity = features @ features.T
        identity = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
        labels = labels[~identity].view(labels.shape[0], -1)
        similarity = similarity[~identity].view(similarity.shape[0], -1)
        positives = similarity[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity[~labels.bool()].view(similarity.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        targets = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
        return F.cross_entropy(logits, targets)

    def _build_decoder(self) -> nn.Module:
        if self.decoder_layers == 1:
            layers: list[nn.Module] = [nn.Linear(self.output_neurons, self.output_neurons)]
        else:
            layers = [nn.Linear(self.output_neurons, 16), nn.ReLU()]
            for _ in range(self.decoder_layers - 2):
                layers.extend([nn.Linear(16, 16), nn.ReLU()])
            layers.append(nn.Linear(16, self.output_neurons))
        if not self.lograte:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def _build_projector(self, dim: int) -> nn.Module:
        if not self.use_contrast_projector:
            return nn.Identity()
        if self.linear_projector:
            return nn.Linear(dim, dim)
        return nn.Sequential(
            nn.Linear(dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim),
        )

    def _mask_observations(
        self,
        x: Tensor,
        should_mask: bool,
        contrast: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        labels = x.clone()
        if not should_mask:
            return x, labels, torch.ones_like(x, dtype=torch.bool)

        ratio = self.contrast_mask_ratio if contrast else self.mask_ratio
        if ratio <= 0.0:
            return x, labels, torch.ones_like(x, dtype=torch.bool)

        mask = self._sample_loss_mask(x, contrast=contrast)
        labels[~mask] = UNMASKED_LABEL
        masked_x = x.clone()
        token_ratio = self.contrast_mask_token_ratio if contrast else self.mask_token_ratio
        random_ratio = self.contrast_mask_random_ratio if contrast else self.mask_random_ratio

        replace_mask = (torch.rand_like(x, dtype=torch.float32) < token_ratio) & mask
        if self.use_zero_mask:
            masked_x[replace_mask] = 0.0
        else:
            masked_x[replace_mask] = float(self.max_spike_count + 1)

        random_mask = (
            (torch.rand_like(x, dtype=torch.float32) < random_ratio)
            & mask
            & ~replace_mask
        )
        if random_mask.any():
            max_count = max(int(torch.ceil(masked_x.max()).item()), 1)
            random_spikes = torch.randint(
                high=max_count,
                size=x.shape,
                device=x.device,
                dtype=torch.long,
            ).to(dtype=x.dtype)
            masked_x[random_mask] = random_spikes[random_mask]

        return masked_x, labels, mask

    def _sample_loss_mask(self, x: Tensor, contrast: bool) -> Tensor:
        batch, time, neurons = x.shape
        mode = self.contrast_mask_mode if contrast else self.mask_mode
        ratio = self.contrast_mask_ratio if contrast else self.mask_ratio
        max_span = self.contrast_mask_max_span if contrast else self.mask_max_span
        expand_prob = self._current_mask_span_expand_prob(contrast=contrast)
        width = 1
        should_expand = (
            max_span > 1
            and expand_prob > 0.0
            and torch.rand((), device=x.device).item() < expand_prob
        )
        if should_expand:
            width = int(torch.randint(1, max_span + 1, (), device=x.device).item())
            ratio = ratio / max(width, 1)

        if mode == "full":
            mask = torch.rand((batch, time, neurons), device=x.device) < ratio
        elif mode == "timestep":
            mask_2d = torch.rand((batch, time), device=x.device) < ratio
            if width > 1:
                mask_2d = _expand_time_mask(mask_2d, width)
            mask = mask_2d.unsqueeze(-1).expand(batch, time, neurons)
        elif mode == "neuron":
            mask_2d = torch.rand((batch, neurons), device=x.device) < ratio
            mask = mask_2d.unsqueeze(1).expand(batch, time, neurons)
        elif mode == "timestep_only":
            mask_1d = torch.rand((time,), device=x.device) < ratio
            if width > 1:
                mask_1d = _expand_time_mask(mask_1d.unsqueeze(0), width).squeeze(0)
            mask = mask_1d.view(1, time, 1).expand(batch, time, neurons)
        else:
            raise KeyError(f"Unknown STNDT mask mode '{mode}'.")

        if not bool(mask.any()):
            flat_index = torch.randint(mask.numel(), (), device=x.device)
            mask = mask.reshape(-1)
            mask[flat_index] = True
            mask = mask.reshape(batch, time, neurons)
        return mask

    def _current_mask_span_expand_prob(self, *, contrast: bool) -> float:
        if contrast:
            start = self.contrast_mask_span_ramp_start
            end = self.contrast_mask_span_ramp_end
            fallback = self.contrast_mask_span_expand_prob
            if end <= start:
                start = self.mask_span_ramp_start
                end = self.mask_span_ramp_end
        else:
            start = self.mask_span_ramp_start
            end = self.mask_span_ramp_end
            fallback = self.mask_span_expand_prob
        if end > start:
            progress = (float(self.training_epoch) - float(start)) / float(end - start)
            return max(0.0, min(1.0, progress))
        return fallback

    def _pad_observed(self, x: Tensor) -> Tensor:
        if x.shape[-1] < self.output_neurons:
            x = F.pad(x, (0, self.output_neurons - x.shape[-1]), value=0.0)
        if self.fwd_steps > 0:
            x = F.pad(x, (0, 0, 0, self.fwd_steps), value=0.0)
        return x

    def _reconstruction_target(
        self,
        batch: Tensor | dict[str, Tensor],
        log_rates: Tensor,
    ) -> Tensor:
        observed = observations_from_batch(batch)
        if isinstance(batch, dict):
            reconstruction = batch.get("reconstruction_spikes")
            if reconstruction is not None:
                if (
                    reconstruction.ndim != log_rates.ndim
                    or reconstruction.shape[0] != log_rates.shape[0]
                    or reconstruction.shape[1] > log_rates.shape[1]
                    or reconstruction.shape[2] != log_rates.shape[2]
                ):
                    raise ValueError("STNDT reconstruction_spikes shape is incompatible with its readout.")
                return reconstruction
            raw = batch.get("raw_spikes")
            if "heldout_spikes" not in batch and raw is not None and raw.shape == observed.shape:
                observed = raw
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            heldout = batch["heldout_spikes"]
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

    def _loss_mask_for_target(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        target: Tensor,
    ) -> Tensor:
        observed_mask = output.extras["observed_loss_mask"].to(
            device=target.device,
            dtype=torch.bool,
        )
        if not self.training:
            return torch.ones_like(target, dtype=torch.bool)
        if target.shape == observed_mask.shape:
            return observed_mask

        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, : self.n_time, : self.n_neurons] = observed_mask
        if isinstance(batch, dict) and "heldout_spikes" in batch:
            heldout = batch["heldout_spikes"]
            n_heldout = int(heldout.shape[-1])
            mask[:, : self.n_time, self.n_neurons : self.n_neurons + n_heldout] = True
            if target.shape[1] > self.n_time:
                mask[:, self.n_time :, :] = True
        return mask

    def _get_or_generate_context_mask(
        self,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.full_context:
            return None
        key = f"{device}:{size}:{dtype}"
        if key in self._attn_masks:
            return self._attn_masks[key]

        context_forward = self.context_forward if self.context_forward >= 0 else size
        allowed = (
            torch.triu(torch.ones(size, size, device=device), diagonal=-context_forward) == 1
        ).transpose(0, 1)
        if self.context_backward > 0:
            back_allowed = (
                torch.triu(torch.ones(size, size, device=device), diagonal=-self.context_backward)
                == 1
            )
            allowed = allowed & back_allowed
        if self.context_wrap_initial and self.context_backward > 0:
            initial = min(self.context_backward, size)
            initial_mask = torch.triu(torch.ones(initial, initial, device=device)).bool()
            allowed[:initial, :initial] |= initial_mask

        attn_mask = torch.zeros(size, size, device=device, dtype=dtype)
        attn_mask = attn_mask.masked_fill(~allowed, float("-inf"))
        self._attn_masks[key] = attn_mask
        return attn_mask

    def init_weights(self) -> None:
        initrange = 0.1
        if self.linear_embedder and self.embed_dim != 0 and isinstance(self.embedder, nn.Linear):
            self.embedder.weight.data.uniform_(-initrange, initrange)
        for embedder in (self.embedder, self.spatial_embedder):
            if isinstance(embedder, SpikeEmbedding):
                if self.spike_log_init:
                    max_spikes = embedder.embedding.num_embeddings + 1
                    log_scale = torch.arange(
                        1,
                        max_spikes,
                        device=embedder.embedding.weight.device,
                        dtype=embedder.embedding.weight.dtype,
                    ).log()
                    log_scale = (log_scale - log_scale.mean()) / (log_scale[-1] - log_scale[0])
                    log_scale = log_scale[: embedder.embedding.num_embeddings] * initrange
                    embedder.embedding.weight.data.uniform_(-initrange / 10.0, initrange / 10.0)
                    embedder.embedding.weight.data += log_scale.unsqueeze(1).expand_as(
                        embedder.embedding.weight.data
                    )
                else:
                    embedder.embedding.weight.data.uniform_(-initrange, initrange)

        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-initrange, initrange)
                module.bias.data.zero_()

    def fixup_initialization(self) -> None:
        scale = 0.67 * (self.num_layers ** (-0.25))
        for layer in self.encoder.layers:
            for module in (layer.temporal_linear1, layer.temporal_linear2):
                module.weight.data.mul_(scale)
            layer.temporal_self_attn.out_proj.weight.data.mul_(scale)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("STNDT expects input shape (batch, time, neurons).")
        if x.shape[1] != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {x.shape[1]}.")
        if x.shape[-1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {x.shape[-1]}.")
        if not torch.isfinite(x).all():
            raise ValueError("STNDT expects finite spike-count observations.")
        if torch.any(x < 0):
            raise ValueError("STNDT expects nonnegative spike-count observations.")


class STNDTNLBAdapter(EvaluationAdapter):
    """Direct held-out NLB scorer for STNDT full readouts."""

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
                        "STNDT direct NLB predictions have shape "
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


class SpatiotemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        spatial_dim: int,
        num_heads: int,
        hidden_size: int,
        dropout: float,
        activation: str,
        num_layers: int,
        pre_norm: bool,
        scale_norm: bool,
    ) -> None:
        super().__init__()
        prototype = SpatiotemporalTransformerEncoderLayer(
            model_dim=model_dim,
            spatial_dim=spatial_dim,
            num_heads=num_heads,
            hidden_size=hidden_size,
            dropout=dropout,
            activation=activation,
            pre_norm=pre_norm,
        )
        self.layers = nn.ModuleList([deepcopy(prototype) for _ in range(num_layers)])
        self.norm: nn.Module
        if scale_norm:
            self.norm = ScaleNorm(model_dim**0.5)
        else:
            self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        temporal: Tensor,
        spatial: Tensor,
        temporal_mask: Tensor | None = None,
        spatial_mask: Tensor | None = None,
        return_outputs: bool = False,
        return_weights: bool = False,
    ) -> tuple[Tensor, Tensor | None, list[tuple[Tensor, Tensor]] | None]:
        outputs = []
        weights = []
        src = temporal
        spatial_src = spatial
        for index, layer in enumerate(self.layers):
            if index > 0:
                spatial_src = src.permute(2, 1, 0)
            src, layer_weights = layer(
                src,
                spatial_src,
                temporal_mask=temporal_mask,
                spatial_mask=spatial_mask,
            )
            if return_outputs:
                outputs.append(src.permute(1, 0, 2))
            if return_weights:
                weights.append(layer_weights)
        src = self.norm(src)
        stacked_outputs = torch.stack(outputs, dim=-1) if outputs else None
        return src, stacked_outputs, weights if return_weights else None


class SpatiotemporalTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        spatial_dim: int,
        num_heads: int,
        hidden_size: int,
        dropout: float,
        activation: str,
        pre_norm: bool,
    ) -> None:
        super().__init__()
        self.pre_norm = bool(pre_norm)
        self.temporal_self_attn = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.spatial_self_attn = nn.MultiheadAttention(
            embed_dim=spatial_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.temporal_norm1 = nn.LayerNorm(model_dim)
        self.temporal_norm2 = nn.LayerNorm(model_dim)
        self.spatial_norm1 = nn.LayerNorm(spatial_dim)
        self.ts_norm1 = nn.LayerNorm(model_dim)
        self.ts_norm2 = nn.LayerNorm(model_dim)
        self.temporal_linear1 = nn.Linear(model_dim, hidden_size)
        self.temporal_linear2 = nn.Linear(hidden_size, model_dim)
        self.ts_linear1 = nn.Linear(model_dim, hidden_size)
        self.ts_linear2 = nn.Linear(hidden_size, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.temporal_dropout1 = nn.Dropout(dropout)
        self.temporal_dropout2 = nn.Dropout(dropout)
        self.ts_dropout1 = nn.Dropout(dropout)
        self.ts_dropout2 = nn.Dropout(dropout)
        self.ts_dropout3 = nn.Dropout(dropout)
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError("activation must be 'relu' or 'gelu'.")

    def forward(
        self,
        temporal_src: Tensor,
        spatial_src: Tensor,
        temporal_mask: Tensor | None = None,
        spatial_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        residual = temporal_src
        src = self.temporal_norm1(temporal_src) if self.pre_norm else temporal_src
        temporal_out, temporal_weights = self.temporal_self_attn(
            src,
            src,
            src,
            attn_mask=temporal_mask,
            need_weights=True,
        )
        src = residual + self.temporal_dropout1(temporal_out)
        if not self.pre_norm:
            src = self.temporal_norm1(src)

        residual = src
        feed_src = self.temporal_norm2(src) if self.pre_norm else src
        feed_out = self.temporal_linear2(
            self.dropout(self.activation(self.temporal_linear1(feed_src)))
        )
        src = residual + self.temporal_dropout2(feed_out)
        if not self.pre_norm:
            src = self.temporal_norm2(src)

        spatial_input = self.spatial_norm1(spatial_src) if self.pre_norm else spatial_src
        spatial_out, spatial_weights = self.spatial_self_attn(
            spatial_input,
            spatial_input,
            spatial_input,
            attn_mask=spatial_mask,
            need_weights=True,
        )
        residual = src
        ts_src = self.ts_norm1(src) if self.pre_norm else src
        spatial_mix = torch.bmm(spatial_weights, ts_src.permute(1, 2, 0)).permute(2, 0, 1)
        ts_out = residual + self.ts_dropout1(spatial_mix)
        if not self.pre_norm:
            ts_out = self.ts_norm1(ts_out)

        residual = ts_out
        ts_feed = self.ts_norm2(ts_out) if self.pre_norm else ts_out
        ts_feed = self.ts_linear2(self.ts_dropout2(self.activation(self.ts_linear1(ts_feed))))
        ts_out = residual + self.ts_dropout3(ts_feed)
        if not self.pre_norm:
            ts_out = self.ts_norm2(ts_out)
        return ts_out, (spatial_weights, temporal_weights)


class SpikeEmbedding(nn.Module):
    def __init__(self, num_embeddings: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, 1)

    def forward(self, x: Tensor) -> Tensor:
        tokens = x.round().long().clamp(min=0, max=self.embedding.num_embeddings - 1)
        return self.embedding(tokens).flatten(start_dim=-2)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        sequence_length: int,
        model_dim: int,
        dropout: float,
        learnable: bool,
        offset: bool = True,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learnable = bool(learnable)
        if self.learnable:
            self.register_buffer("position_ids", torch.arange(sequence_length, dtype=torch.long))
            self.embedding = nn.Embedding(sequence_length, model_dim)
        else:
            position = torch.arange(0, sequence_length, dtype=torch.float32).unsqueeze(1)
            if offset:
                position = position + 1
            pe = torch.zeros(sequence_length, model_dim, dtype=torch.float32)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / model_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            if model_dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(1))

    def forward(self, x: Tensor) -> Tensor:
        if self.learnable:
            positions = self.position_ids[: x.shape[0]].to(x.device)
            x = x + self.embedding(positions).unsqueeze(1)
        else:
            x = x + self.pe[: x.shape[0]].to(device=x.device, dtype=x.dtype)
        return self.dropout(x)


class ScaleNorm(nn.Module):
    def __init__(self, scale: float, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


def _expand_time_mask(mask: Tensor, width: int) -> Tensor:
    kernel = torch.ones(width, device=mask.device, dtype=mask.dtype).view(1, 1, -1)
    expanded = F.conv1d(mask.float().unsqueeze(1), kernel.float(), padding=width // 2)
    if width % 2 == 0:
        expanded = expanded[..., :-1]
    return expanded.squeeze(1).clamp_(0, 1).bool()
