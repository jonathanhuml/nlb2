"""Bayesian GPFA adapter backed by mgplvm-pytorch."""

from __future__ import annotations

import importlib
import math
import numpy as np
from typing import Any, Literal, Optional

import torch
from pydantic import Field, model_validator
from torch import Tensor

from nlb2.metrics import NLBCoSmoothingAdapter
from nlb2.models.base import BaseDynamicsModel, BaseModelConfig, OptimizationConfig
from nlb2.types import LossOutput, ModelOutput, observations_from_batch


@BaseModelConfig.register
class BGPFAConfig(BaseModelConfig):
    """Config for variational Bayesian GPFA."""

    name: Literal["bgpfa"] = "bgpfa"
    objective: str = "negative_elbo"
    latent_dim: int = 3
    binsize: float = 25.0
    ell0: Optional[float] = None
    rho: float = 2.0
    n_mc_train: int = 3
    n_mc_eval: int = 5
    kl_burnin_epochs: int = 1
    latent_scale_init: float = 1.0
    likelihood: Literal["gaussian", "poisson"] = "gaussian"
    learn_scale: bool = False
    ard: bool = True
    dtype: Literal["float64", "float32"] = "float64"
    latent_init: Literal["gp_prior", "fa"] = "gp_prior"
    observation_init: Literal["mgplvm", "fa"] = "mgplvm"
    nlb_feature_source: Literal["latents", "rates", "reconstruction"] = "latents"
    nlb_decoder: Literal["ridge", "poisson"] = "poisson"
    nlb_ridge_alpha: float = 1.0e-2
    nlb_poisson_max_iter: int = 80
    nlb_latent_infer_steps: int = Field(default=300, ge=1)
    nlb_latent_infer_n_mc: int = Field(default=20, ge=1)
    nlb_latent_infer_lr: float = Field(default=1e-1, gt=0)
    nlb_latent_infer_burnin: int = Field(default=1, ge=1)
    optimization: OptimizationConfig = Field(
        default_factory=lambda: OptimizationConfig(
            name="mgplvm_full_batch_gradient",
            optimizer="Adam",
            lr=1e-1,
            steps_per_epoch=1,
            burnin=150,
            n_mc=3,
            weight_decay=0.0,
        )
    )

    @model_validator(mode="after")
    def reject_em_optimization(self) -> "BGPFAConfig":
        if self.optimization.name == "em":
            raise ValueError(
                "BGPFA uses a differentiable variational ELBO and does not support "
                "optimization.name='em'."
            )
        return self

    def build(self, n_neurons: int, n_time: int) -> "BGPFA":
        return BGPFA(
            n_neurons=n_neurons,
            n_time=n_time,
            latent_dim=self.latent_dim,
            binsize=self.binsize,
            ell0=self.ell0,
            rho=self.rho,
            n_mc_train=self.n_mc_train,
            n_mc_eval=self.n_mc_eval,
            kl_burnin_epochs=self.kl_burnin_epochs,
            latent_scale_init=self.latent_scale_init,
            likelihood=self.likelihood,
            learn_scale=self.learn_scale,
            ard=self.ard,
            dtype=self.dtype,
            latent_init=self.latent_init,
            observation_init=self.observation_init,
            nlb_feature_source=self.nlb_feature_source,
            nlb_decoder=self.nlb_decoder,
            nlb_ridge_alpha=self.nlb_ridge_alpha,
            nlb_poisson_max_iter=self.nlb_poisson_max_iter,
            nlb_latent_infer_steps=self.nlb_latent_infer_steps,
            nlb_latent_infer_n_mc=self.nlb_latent_infer_n_mc,
            nlb_latent_infer_lr=self.nlb_latent_infer_lr,
            nlb_latent_infer_burnin=self.nlb_latent_infer_burnin,
            objective=self.objective,
        )


class BGPFA(BaseDynamicsModel):
    """Variational Bayesian GPFA with ARD and differentiable ELBO training.

    ## When to use

    Use BGPFA when you want the Bayesian GPFA objective from
    `tachukao/mgplvm-pytorch` inside the NLB2 trainer contract. Unlike the
    classical GPFA EM baseline, this adapter optimizes a Monte Carlo variational
    negative ELBO with standard PyTorch backpropagation.

    ## Assumptions

    Observations are passed as `(batch, time, neurons)` tensors and internally
    transposed to mgplvm's `(trials, neurons, time)` convention. The latent
    posterior has per-trial variational parameters, so the default optimization
    strategy is `mgplvm_full_batch_gradient`. One NLB2 epoch can run multiple
    mgplvm optimizer updates via `optimization.steps_per_epoch`; this is useful
    when matching reference bGPFA scripts that report fixed optimizer-step
    budgets.

    ## Outputs

    `forward` returns predictive rates/reconstructions, variational latent
    means, and ELBO terms in `extras`. The core mgplvm implementation is
    vendored in `src/mgplvm`; this class only adapts it to the NLB2 model,
    loss, and trainer contracts. Evaluation infers a new posterior from each
    input batch with the learned observation model and GP prior held fixed.
    The `nlb_latent_infer_*` options control this inference for both NLB and
    synthetic evaluation. Predictions are expected counts per input bin.
    """

    def __init__(
        self,
        n_neurons: int,
        n_time: int,
        latent_dim: int = 3,
        binsize: float = 25.0,
        ell0: float | None = None,
        rho: float = 2.0,
        n_mc_train: int = 3,
        n_mc_eval: int = 5,
        kl_burnin_epochs: int = 1,
        latent_scale_init: float = 1.0,
        likelihood: Literal["gaussian", "poisson"] = "gaussian",
        learn_scale: bool = False,
        ard: bool = True,
        dtype: Literal["float64", "float32"] = "float64",
        latent_init: str = "gp_prior",
        observation_init: str = "mgplvm",
        nlb_feature_source: str = "latents",
        nlb_decoder: str = "poisson",
        nlb_ridge_alpha: float = 1.0e-2,
        nlb_poisson_max_iter: int = 80,
        nlb_latent_infer_steps: int = 300,
        nlb_latent_infer_n_mc: int = 20,
        nlb_latent_infer_lr: float = 1e-1,
        nlb_latent_infer_burnin: int = 1,
        objective: str = "negative_elbo",
    ) -> None:
        super().__init__()
        self.n_neurons = int(n_neurons)
        self.n_time = int(n_time)
        if self.n_time < 2:
            raise ValueError("BGPFA requires at least two time bins for its GP prior.")
        self.latent_dim = int(latent_dim)
        self.binsize = float(binsize)
        self.ell0 = float(ell0) if ell0 is not None else 200.0 / self.binsize
        self.rho = float(rho)
        self.n_mc_train = int(n_mc_train)
        self.n_mc_eval = int(n_mc_eval)
        self.kl_burnin_epochs = int(kl_burnin_epochs)
        self.latent_scale_init = float(latent_scale_init)
        self.likelihood = likelihood
        self.learn_scale = bool(learn_scale)
        self.ard = bool(ard)
        self.dtype = dtype
        self.latent_init = str(latent_init)
        self.observation_init = str(observation_init)
        self.nlb_feature_source = str(nlb_feature_source)
        self.nlb_decoder = str(nlb_decoder)
        self.nlb_ridge_alpha = float(nlb_ridge_alpha)
        self.nlb_poisson_max_iter = int(nlb_poisson_max_iter)
        self.nlb_latent_infer_steps = int(nlb_latent_infer_steps)
        self.nlb_latent_infer_n_mc = int(nlb_latent_infer_n_mc)
        self.nlb_latent_infer_lr = float(nlb_latent_infer_lr)
        self.nlb_latent_infer_burnin = int(nlb_latent_infer_burnin)
        self.objective = objective

        fit_ts = torch.arange(self.n_time, dtype=self._torch_dtype())[None, None, :]
        self.register_buffer("fit_ts", fit_ts)
        self.register_buffer("_train_observations", fit_ts.new_empty(0))
        self._train_n_trials: int | None = None
        self._train_mod: torch.nn.Module | None = None
        # Keep only the last input/posterior pair; trial count is not identity.
        self._eval_cache: tuple[Tensor, torch.nn.Module] | None = None

    def forward(self, x: Tensor) -> ModelOutput:
        x = self._coerce_observations(x)
        self._validate_input(x)
        mod = self._model_for(x)
        y = self._to_mgplvm_observations(x)

        n_mc = self.n_mc_train if self.training else max(1, self.n_mc_eval)
        svgp_elbo, latent_kl = mod(
            y,
            n_mc,
            analytic_kl="GP" in mod.lat_dist.name,
        )
        latent_kl = latent_kl.mean() if latent_kl.ndim > 0 else latent_kl
        reconstruction = self._predict_from_latent_mean(mod)
        latents = mod.lat_dist.lat_mu.to(x.device, x.dtype)

        return ModelOutput(
            rates=reconstruction.clamp_min(0.0),
            latents=latents,
            reconstruction=reconstruction,
            extras={
                "svgp_elbo": svgp_elbo,
                "latent_kl": latent_kl,
                "mgplvm_model": mod,
            },
        )

    def loss(
        self,
        batch: Tensor | dict[str, Tensor],
        output: ModelOutput,
        epoch: int = 0,
    ) -> LossOutput:
        x = observations_from_batch(batch)
        svgp_elbo = output.extras["svgp_elbo"]
        latent_kl = output.extras["latent_kl"]
        kl_weight = self._kl_weight(epoch)
        total = (-svgp_elbo + kl_weight * latent_kl) / x.numel()
        return LossOutput(
            total=total,
            named_terms={
                "negative_elbo": total,
                "svgp_elbo": svgp_elbo.detach(),
                "latent_kl": latent_kl.detach(),
                "kl_weight": kl_weight,
            },
            objective=self.objective,
        )

    def evaluation_adapter(self, task: str):
        if task == "nlb":
            return NLBCoSmoothingAdapter(
                feature_source=self.nlb_feature_source,
                decoder=self.nlb_decoder,
                ridge_alpha=self.nlb_ridge_alpha,
                poisson_max_iter=self.nlb_poisson_max_iter,
            )
        return None

    def _model_for(self, x: Tensor) -> torch.nn.Module:
        x = self._coerce_observations(x)
        if self.training:
            return self.mgplvm_training_model(x)

        if self._train_mod is None:
            raise RuntimeError("BGPFA must be trained before evaluation.")
        if torch.equal(x, self._train_observations):
            return self._train_mod
        if self._eval_cache is not None and torch.equal(x, self._eval_cache[0]):
            return self._eval_cache[1]
        return self.infer_latents(
            x,
            max_steps=self.nlb_latent_infer_steps,
            n_mc=self.nlb_latent_infer_n_mc,
            lrate=self.nlb_latent_infer_lr,
            burnin=self.nlb_latent_infer_burnin,
        )

    def mgplvm_training_model(self, x: Tensor) -> torch.nn.Module:
        x = self._coerce_observations(x)
        self._validate_input(x)
        if self._train_mod is None:
            self._train_n_trials = int(x.shape[0])
            self._train_mod = self._build_mgplvm_model(x)
        elif int(x.shape[0]) != self._train_n_trials:
            raise ValueError(
                "BGPFA training requires a stable full-batch trial count. "
                f"Expected {self._train_n_trials}, got {int(x.shape[0])}."
            )
        if self._train_observations.numel() == 0:
            self._train_observations = x.detach().clone()
        elif not torch.equal(x, self._train_observations):
            raise ValueError(
                "BGPFA training requires the same full batch in the same trial order."
            )
        self._eval_cache = None
        return self._train_mod

    def mgplvm_observations(self, x: Tensor) -> Tensor:
        return self._to_mgplvm_observations(self._coerce_observations(x))

    @torch.inference_mode(False)
    @torch.enable_grad()
    def infer_latents(
        self,
        x: Tensor,
        max_steps: int = 300,
        n_mc: int = 20,
        lrate: float = 1e-1,
        burnin: int = 1,
    ) -> torch.nn.Module:
        if self._train_mod is None:
            raise RuntimeError("BGPFA must be trained before held-out latent inference.")
        if max_steps < 1 or n_mc < 1 or lrate <= 0 or burnin < 1:
            raise ValueError("BGPFA latent inference requires positive steps, samples, lr and burnin.")

        mgp = _require_mgplvm()
        x = self._coerce_observations(x).detach()
        if x.is_inference():
            x = x.clone()
        self._validate_input(x)
        mod = self._build_mgplvm_model(x, initialize=False)
        self._copy_observation_state(target=mod, source=self._train_mod)
        with torch.no_grad():
            mod.lat_dist._ell.copy_(self._train_mod.lat_dist._ell)
        for param in mod.parameters():
            param.requires_grad = False
        for param in (mod.lat_dist._nu, mod.lat_dist._scale, mod.lat_dist._c):
            param.requires_grad = True

        params = mgp.crossval.training_params(
            max_steps=max_steps,
            n_mc=n_mc,
            lrate=lrate,
            print_every=np.nan,
            burnin=burnin,
            mask_Ts=lambda value: value * 1,
        )
        mgp.crossval.train_model(mod, self._to_mgplvm_observations(x), params)
        mod.requires_grad_(False)
        mod.eval()
        self._eval_cache = (x.detach().clone(), mod)
        return mod

    def _build_mgplvm_model(self, x: Tensor, *, initialize: bool = True) -> torch.nn.Module:
        mgp = _require_mgplvm()
        y_np = self._to_mgplvm_observations(x).detach().cpu().numpy() if initialize else None
        n_trials = int(x.shape[0])
        manif = mgp.manifolds.Euclid(self.n_time, self.latent_dim)
        lat_dist = mgp.rdist.GP_circ(
            manif,
            self.n_time,
            n_trials,
            self.fit_ts.to(x.device, x.dtype),
            _scale=self.latent_scale_init,
            ell=self.ell0,
        )
        lprior = mgp.lpriors.Null(manif)
        likelihood = self._build_likelihood(mgp, x, y_np)
        mod = mgp.models.Lvgplvm(
            self.n_neurons,
            self.n_time,
            self.latent_dim,
            n_trials,
            lat_dist,
            lprior,
            likelihood,
            Y=y_np,
            learn_scale=self.learn_scale,
            ard=self.ard,
            rel_scale=self.rho,
        )
        mod = mod.to(device=x.device, dtype=x.dtype)
        if self.latent_init not in {"gp_prior", "fa"}:
            raise ValueError(f"Unsupported bGPFA latent_init '{self.latent_init}'.")
        if self.observation_init not in {"mgplvm", "fa"}:
            raise ValueError(f"Unsupported bGPFA observation_init '{self.observation_init}'.")
        if initialize and (self.latent_init == "fa" or self.observation_init == "fa"):
            self._initialize_from_fa(mod, y_np, x.device, x.dtype)
        return mod

    def _initialize_from_fa(
        self,
        mod: torch.nn.Module,
        y_np: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        from sklearn import decomposition

        n_trials, n_neurons, n_time = y_np.shape
        flat = np.asarray(y_np, dtype=np.float64).transpose(0, 2, 1).reshape(
            n_trials * n_time,
            n_neurons,
        )
        components = min(self.latent_dim, n_neurons, flat.shape[0])
        if components < 1:
            return
        fa = decomposition.FactorAnalysis(n_components=components)
        scores = fa.fit_transform(flat)
        loadings = np.asarray(fa.components_.T, dtype=np.float64)
        scale = np.std(scores, axis=0, keepdims=True)
        score_scale = 0.5 / np.maximum(scale, 1.0e-6)
        scores = scores * score_scale
        loadings = loadings / score_scale
        if components < self.latent_dim:
            padded = np.zeros((scores.shape[0], self.latent_dim), dtype=np.float64)
            padded[:, :components] = scores
            scores = padded
            padded_loadings = np.zeros((n_neurons, self.latent_dim), dtype=np.float64)
            padded_loadings[:, :components] = loadings
            loadings = padded_loadings
        latents = scores.reshape(n_trials, n_time, self.latent_dim)
        nu = torch.as_tensor(
            latents.transpose(0, 2, 1),
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            if self.latent_init == "fa":
                mod.lat_dist._nu.copy_(nu)
            if self.observation_init == "fa" and hasattr(mod.obs, "_q_mu"):
                obs = mod.obs
                effective_scale = (
                    obs.neuron_scale.detach()
                    * obs.scale.detach().reshape(1, 1)
                    * obs.dim_scale.detach().reshape(1, -1)
                )
                q_mu = torch.as_tensor(loadings, dtype=dtype, device=device) / effective_scale.clamp_min(
                    1.0e-8
                )
                obs._q_mu.copy_(q_mu.unsqueeze(0))

    def _build_likelihood(self, mgp: Any, x: Tensor, y_np: Any) -> torch.nn.Module:
        if self.likelihood == "gaussian":
            sigma = 0.1 * torch.ones(self.n_neurons, device=x.device, dtype=x.dtype)
            return mgp.likelihoods.Gaussian(
                self.n_neurons,
                Y=y_np,
                sigma=sigma,
            )
        if self.likelihood == "poisson":
            return mgp.likelihoods.Poisson(
                self.n_neurons,
                binsize=self.binsize,
            )
        raise ValueError(f"Unsupported BGPFA likelihood '{self.likelihood}'.")

    @staticmethod
    def _copy_observation_state(
        target: torch.nn.Module,
        source: torch.nn.Module,
    ) -> None:
        target.obs.load_state_dict(source.obs.state_dict())

    def _predict_from_latent_mean(self, mod: torch.nn.Module) -> Tensor:
        with torch.no_grad():
            query = mod.lat_dist.lat_mu.detach().transpose(-1, -2)
            mean, variance = mod.svgp.predict(query[None], full_cov=False)
            if self.likelihood == "poisson":
                likelihood = mod.obs.likelihood
                c, d = likelihood.prms
                mean = likelihood.binsize * torch.exp(
                    c[..., None] * mean + d[..., None]
                    + 0.5 * c[..., None].square() * variance
                )
            return mean[0].permute(0, 2, 1)

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_burnin_epochs <= 0:
            return 1.0
        return float(1.0 - math.exp(-float(epoch + 1) / (3.0 * self.kl_burnin_epochs)))

    def _to_mgplvm_observations(self, x: Tensor) -> Tensor:
        return x.permute(0, 2, 1).contiguous()

    def _coerce_observations(self, x: Tensor) -> Tensor:
        return x.to(device=self.fit_ts.device, dtype=self.fit_ts.dtype)

    def train(self, mode: bool = True) -> "BGPFA":
        if mode:
            self._eval_cache = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self._eval_cache = None
        result = super()._apply(fn, recurse=recurse)
        if self._train_mod is not None:
            # mgplvm stores these timestamps as an ordinary tensor attribute.
            self._train_mod.lat_dist.ts = self.fit_ts
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        self._eval_cache = None
        observations_key = prefix + "_train_observations"
        observations = state_dict.get(observations_key)
        if observations is None:
            # Older checkpoints did not store trial identity. Infer fresh latents
            # during evaluation until an explicit training batch supplies it.
            observations = self.fit_ts.new_empty(0)
            state_dict[observations_key] = observations
        self._train_observations = torch.empty_like(observations, device=self.fit_ts.device)
        latent_key = prefix + "_train_mod.lat_dist._nu"
        latent = state_dict.get(latent_key)
        if latent is not None:
            self._train_n_trials = int(latent.shape[0])
            if self._train_mod is None or self._train_mod.n_samples != self._train_n_trials:
                placeholder = self.fit_ts.new_empty(1, self.n_time, self.n_neurons).expand(
                    self._train_n_trials, -1, -1
                )
                self._train_mod = self._build_mgplvm_model(placeholder, initialize=False)
            self._train_mod.train(self.training)
        else:
            self._train_mod = None
            self._train_n_trials = None
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _torch_dtype(self) -> torch.dtype:
        if self.dtype == "float64":
            return torch.float64
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported BGPFA dtype '{self.dtype}'.")

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("BGPFA expects observations shaped (batch, time, neurons).")
        if int(x.shape[1]) != self.n_time:
            raise ValueError(f"Expected {self.n_time} time bins, got {int(x.shape[1])}.")
        if int(x.shape[2]) != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {int(x.shape[2])}.")


def _require_mgplvm() -> Any:
    try:
        return importlib.import_module("mgplvm")
    except ImportError as exc:
        raise ImportError(
            "BGPFA requires the vendored `mgplvm` package and its runtime "
            "dependencies. Reinstall NLB2 after this change, and make sure "
            "`scikit-learn` is available in the active environment."
        ) from exc
