"""Self-contained Bayesian GPFA adapted from mgplvm-pytorch.

The NLB2 model and configuration are followed by the numerical implementation.
Upstream MIT license notices are retained with the corresponding helpers below.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, List, Literal, Optional, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from pydantic import Field, model_validator
from sklearn import decomposition
import torch
from torch import Tensor, nn
import torch.distributions as dists
from torch.distributions import MultivariateNormal, kl_divergence, transform_to, constraints
from torch.fft import fft, ifft, rfft, irfft
from torch.optim.lr_scheduler import LambdaLR

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
    means, and ELBO terms in `extras`. This file includes the required
    numerical routines adapted from mgplvm. Evaluation infers a new posterior
    from each input batch with the learned observation model and GP prior held
    fixed.
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

        fit_latents(
            mod,
            self._to_mgplvm_observations(x),
            max_steps=max_steps,
            n_mc=n_mc,
            lrate=lrate,
            burnin=burnin,
        )
        mod.requires_grad_(False)
        mod.eval()
        self._eval_cache = (x.detach().clone(), mod)
        return mod

    def _build_mgplvm_model(self, x: Tensor, *, initialize: bool = True) -> torch.nn.Module:

        y_np = self._to_mgplvm_observations(x).detach().cpu().numpy() if initialize else None
        n_trials = int(x.shape[0])
        lat_dist = GP_circ(
            self.latent_dim,
            self.n_time,
            n_trials,
            self.fit_ts.to(x.device, x.dtype),
            _scale=self.latent_scale_init,
            ell=self.ell0,
        )
        likelihood = self._build_likelihood(x, y_np)
        mod = Lvgplvm(
            self.n_neurons,
            self.n_time,
            self.latent_dim,
            n_trials,
            lat_dist,
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

    def _build_likelihood(self, x: Tensor, y_np: Any) -> torch.nn.Module:

        if self.likelihood == "gaussian":
            sigma = 0.1 * torch.ones(self.n_neurons, device=x.device, dtype=x.dtype)
            return Gaussian(
                self.n_neurons,
                Y=y_np,
                sigma=sigma,
            )
        if self.likelihood == "poisson":
            return Poisson(
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


# Adapted from https://github.com/tachukao/mgplvm-pytorch
#
# MIT License
#
# Copyright (c) 2020 Ta-Chu Kao and Kristopher T. Jensen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Toeplitz and broadcasting routines adapted from GPyTorch.
# MIT License
#
# Copyright (c) 2017 Jake Gardner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Numerical helpers

def softplus(x):
    return torch.log(1 + torch.exp(x))


def inv_softplus(x):
    return torch.log(torch.exp(x) - 1)


def _mul_broadcast_shape(*shapes, error_msg=None):
    """Compute dimension suggested by multiple tensor indices (supports broadcasting)"""

    # Pad each shape so they have the same number of dimensions
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple(
        [1] * (num_dims - len(shape)) + list(shape) for shape in shapes)

    # Make sure that each dimension agrees in size
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0]
                   for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError(
                        "Shapes are not broadcastable for mul operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes
        else:
            final_size.append(1)

    return torch.Size(final_size)


def _matmul_broadcast_shape(shape_a, shape_b, error_msg=None):
    """Compute dimension of matmul operation on shapes (supports broadcasting)"""
    m, n, p = shape_a[-2], shape_a[-1], shape_b[-1]

    if len(shape_b) == 1:
        if n != p:
            if error_msg is None:
                raise RuntimeError(
                    f"Incompatible dimensions for matmul: {shape_a} and {shape_b}"
                )
            else:
                raise RuntimeError(error_msg)
        return shape_a[:-1]

    if n != shape_b[-2]:
        if error_msg is None:
            raise RuntimeError(
                f"Incompatible dimensions for matmul: {shape_a} and {shape_b}")
        else:
            raise RuntimeError(error_msg)

    tail_shape = torch.Size([m, p])

    # Figure out batch shape
    batch_shape_a = shape_a[:-2]
    batch_shape_b = shape_b[:-2]
    if batch_shape_a == batch_shape_b:
        bc_shape = batch_shape_a
    else:
        bc_shape = _mul_broadcast_shape(batch_shape_a, batch_shape_b)
    return bc_shape + tail_shape


def toeplitz_matmul(toeplitz_column, toeplitz_row, tensor):
    """
    Performs multiplication T * M where the matrix T is Toeplitz.
    Args:
        - toeplitz_column (vector n or b x n) - First column of the Toeplitz matrix T.
        - toeplitz_row (vector n or b x n) - First row of the Toeplitz matrix T.
        - tensor (matrix n x p or b x n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor (n x p or b x n x p) - The result of the matrix multiply T * M.
    """
    if toeplitz_column.size() != toeplitz_row.size():
        raise RuntimeError(
            "c and r should have the same length (Toeplitz matrices are necessarily square)."
        )

    toeplitz_shape = torch.Size((*toeplitz_column.shape, toeplitz_row.size(-1)))
    output_shape = _matmul_broadcast_shape(toeplitz_shape,
                                                        tensor.shape)
    broadcasted_t_shape = output_shape[:-1] if tensor.dim(
    ) > 1 else output_shape

    if tensor.ndimension() == 1:
        tensor = tensor.unsqueeze(-1)
    toeplitz_column = toeplitz_column.expand(*broadcasted_t_shape)
    toeplitz_row = toeplitz_row.expand(*broadcasted_t_shape)
    tensor = tensor.expand(*output_shape)

    if not torch.equal(toeplitz_column[..., 0], toeplitz_row[..., 0]):
        raise RuntimeError(
            "The first column and first row of the Toeplitz matrix should have "
            "the same first element, otherwise the value of T[0,0] is ambiguous. "
            "Got: c[0]={} and r[0]={}".format(toeplitz_column[0],
                                              toeplitz_row[0]))

    if type(toeplitz_column) != type(toeplitz_row) or type(
            toeplitz_column) != type(tensor):
        raise RuntimeError("The types of all inputs to ToeplitzMV must match.")

    *batch_shape, orig_size, num_rhs = tensor.size()
    r_reverse = toeplitz_row[..., 1:].flip(dims=(-1,))

    c_r_rev = torch.zeros(*batch_shape,
                          orig_size + r_reverse.size(-1),
                          dtype=tensor.dtype,
                          device=tensor.device)
    c_r_rev[..., :orig_size] = toeplitz_column
    c_r_rev[..., orig_size:] = r_reverse

    temp_tensor = torch.zeros(*batch_shape,
                              2 * orig_size - 1,
                              num_rhs,
                              dtype=toeplitz_column.dtype,
                              device=toeplitz_column.device)
    temp_tensor[..., :orig_size, :] = tensor

    fft_M = fft(temp_tensor.transpose(-1, -2).contiguous())
    fft_c = fft(c_r_rev).unsqueeze(-2).expand_as(fft_M)
    fft_product = fft_M.mul_(fft_c)

    output = ifft(fft_product).real.transpose(-1, -2)
    output = output[..., :orig_size, :]
    return output


def sym_toeplitz_matmul(toeplitz_column, tensor):
    """
    Performs a matrix-matrix multiplication TM where the matrix T is symmetric Toeplitz.
    Args:
        - toeplitz_column (vector n) - First column of the symmetric Toeplitz matrix T.
        - matrix (matrix n x p) - Matrix or vector to multiply the Toeplitz matrix with.
    Returns:
        - tensor
    """
    return toeplitz_matmul(toeplitz_column, toeplitz_column, tensor)


# Gaussian and Poisson observation likelihoods

log2pi: float = np.log(2 * np.pi)
n_gh_locs: int = 20


def exp_link(x):
    '''exponential link function used for positive observations'''
    return torch.exp(x)


def FA_init(Y, d: Optional[int] = None):
    n_samples, n, m = Y.shape
    if d is None:
        d = int(np.round(n / 4))
    pca = decomposition.FactorAnalysis(n_components=d)
    Y = Y.transpose(0, 2, 1).reshape(n_samples * m, n)
    mudata = pca.fit_transform(Y)  #m*n_samples x d
    sigmas = 1.5 * np.sqrt(pca.noise_variance_)


    return torch.tensor(sigmas, dtype=torch.get_default_dtype())


class Likelihood(nn.Module):
    """Shared dimensions for BGPFA observation likelihoods."""

    def __init__(self, n: int, n_gh_locs: int = n_gh_locs):
        super().__init__()
        self.n = n
        self.n_gh_locs = n_gh_locs


class Gaussian(Likelihood):
    name = "Gaussian"

    def __init__(self,
                 n: int,
                 sigma: Optional[Tensor] = None,
                 n_gh_locs=n_gh_locs,
                 learn_sigma=True,
                 Y: Optional[np.ndarray] = None,
                 d: Optional[int] = None):
        super().__init__(n, n_gh_locs)

        if sigma is None:
            if Y is None:
                sigma = 1 * torch.ones(n,)
            else:
                sigma = FA_init(Y, d=d)
        self._sigma = nn.Parameter(data=sigma, requires_grad=learn_sigma)

    @property
    def prms(self) -> Tensor:
        variance = torch.square(self._sigma)
        return variance

    @property
    def sigma(self) -> Tensor:
        return (1e-20 + self.prms).sqrt()

    def log_prob(self, y):
        raise Exception("Gaussian likelihood not implemented")

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting Gaussian distributions
        """
        prms = self.prms
        dist = torch.distributions.Normal(fs,
                                          torch.sqrt(prms)[None, None, :, None])
        return dist

    def sample(self, f_samps: Tensor) -> Tensor:
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting Gaussian distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(f_samps)
        #sample from p(y|f)
        y_samps = dist.sample()
        return y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting Gaussian distributions (n_mc x n_samples x n x m)
            for a Gaussian, this is simply fs
        """
        return fs

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n_samples x n)
        """
        n_mc, m = fmu.shape[0], fmu.shape[-1]
        variance = self.prms  #(n)
        inv_variance = 1 / variance[..., None]
        ve1 = -0.5 * log2pi * m  #scalar
        ve2 = -0.5 * torch.log(variance) * m  #(n)
        ve3 = -0.5 * torch.square(
            y - fmu) * inv_variance  #(n_mc x n_samples x n x m )
        ve4 = -0.5 * fvar * inv_variance  #(n_mc x n_samples x n x m)

        #(n_mc x n_samples x n)
        return ve1 + ve2 + ve3.sum(-1) + ve4.sum(-1)

    @property
    def msg(self):
        sig = torch.mean(self.sigma).item()
        return (' lik_sig {:.3f} |').format(sig)


class Poisson(Likelihood):
    name = "Poisson"

    def __init__(
            self,
            n: int,
            inv_link=exp_link,  #torch.exp,
            binsize=1,
            c: Optional[Tensor] = None,
            d: Optional[Tensor] = None,
            fixed_c=True,
            fixed_d=False,
            n_gh_locs: Optional[int] = n_gh_locs):
        super().__init__(n, n_gh_locs)
        self.inv_link = inv_link
        self.binsize = binsize
        c = torch.ones(n,) if c is None else c
        d = torch.zeros(n,) if d is None else d
        self.c = nn.Parameter(data=c, requires_grad=not fixed_c)
        self.d = nn.Parameter(data=d, requires_grad=not fixed_d)
        self.n_gh_locs = n_gh_locs

    @property
    def prms(self):
        return self.c, self.d

    def log_prob(self, lamb, y):
        #lambd: (n_mc, n_samples x n, m, n_gh)
        #y: (n, n_samples x m)
        p = dists.Poisson(lamb)
        return p.log_prob(y[None, ..., None])

    def dist(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        dist : distribution
            resulting Poisson distributions
        """
        c, d = self.prms
        lambd = self.binsize * self.inv_link(c[..., None] * fs + d[..., None])
        dist = torch.distributions.Poisson(lambd)
        return dist

    def sample(self, f_samps: Tensor):
        """
        Parameters
        ----------
        f_samps : Tensor
            GP output samples (n_mc x n_samples x n x m)

        Returns
        -------
        y_samps : Tensor
            samples from the resulting Poisson distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(f_samps)
        y_samps = dist.sample()
        return y_samps

    def dist_mean(self, fs: Tensor):
        """
        Parameters
        ----------
        fs : Tensor
            GP mean function values (n_mc x n_samples x n x m)

        Returns
        -------
        mean : Tensor
            means of the resulting Poisson distributions (n_mc x n_samples x n x m)
        """
        dist = self.dist(fs)
        mean = dist.mean.detach()
        return mean

    def variational_expectation(self, y, fmu, fvar):
        """
        Parameters
        ----------
        y : Tensor
            number of MC samples (n_samples x n x m)
        fmu : Tensor
            GP mean (n_mc x n_samples x n x m)
        fvar : Tensor
            GP diagonal variance (n_mc x n_samples x n x m)

        Returns
        -------
        Log likelihood : Tensor
            SVGP likelihood term per MC, neuron, sample (n_mc x n)
        """
        c, d = self.prms
        fmu = c[..., None] * fmu + d[..., None]
        fvar = fvar * torch.square(c[..., None])
        if self.inv_link == exp_link:
            n_mc = fmu.shape[0]
            v1 = (y * fmu) - (self.binsize * torch.exp(fmu + 0.5 * fvar))
            v2 = (y * np.log(self.binsize) - torch.lgamma(y + 1))
            #v1: (n_b x n_samples x n x m)  v2: (n_samples x n x m) (per mc sample)
            lp = v1.sum(-1) + v2.sum(-1)
            return lp

        else:
            # use Gauss-Hermite quadrature to approximate integral
            locs, ws = hermgauss(self.n_gh_locs)
            ws = torch.tensor(ws, device=fmu.device)
            locs = torch.tensor(locs, device=fvar.device)
            fvar = fvar[..., None]  #add n_gh
            fmu = fmu[..., None]  #add n_gh
            locs = self.inv_link(torch.sqrt(2. * fvar) * locs +
                                 fmu) * self.binsize  #(n_mc, n, m, n_gh)
            lp = self.log_prob(locs, y)
            return 1 / np.sqrt(np.pi) * (lp * ws).sum(-1).sum(-1)
            #return torch.sum(1 / np.sqrt(np.pi) * lp * ws)

    @property
    def msg(self):
        return " "


# Variational Bayesian factor analysis

class Bvfa(nn.Module):
    name = "Bvfa"

    def __init__(self,
                 n: int,
                 d: int,
                 m: int,
                 n_samples: int,
                 likelihood: Likelihood,
                 q_mu: Optional[Tensor] = None,
                 q_sqrt: Optional[Tensor] = None,
                 tied_samples=True,
                 Y=None,
                 learn_neuron_scale=False,
                 ard=False,
                 learn_scale=None,
                 rel_scale=1,
                 scale=None,
                 dim_scale=None,
                 neuron_scale=None):
        """
        __init__ method for Base Variational Factor Analysis
        Parameters
        ----------
        n : int
            number of neurons
        d: int
            latent dimensionality
        m : int
            number of conditions
        n_samples : int
            number of samples
        likelihood : Likelihood
            likliehood module used for computing variational expectation
        q_mu : Optional Tensor
            optional Tensor for initialization
        q_sqrt : Optional Tensor
            optional Tensor for initialization
        tied_samples : Optional bool
        """
        super().__init__()
        self.n = n
        self.d = d
        self.m = m
        self.tied_samples = tied_samples
        self.n_samples = n_samples

        #### initialize prior parameters ####

        _scale = torch.ones(1)
        _dim_scale = torch.ones(d)
        _neuron_scale = torch.ones(n)
        if learn_scale is None:
            learn_scale = not (ard or learn_neuron_scale)

        if Y is not None:  #initialize from FA
            n_samples_fa, n_fa, m_fa = Y.shape
            fa_rank = min(d, n_fa, n_samples_fa * m_fa)
            mod = decomposition.FactorAnalysis(n_components=fa_rank)
            Y_fa = Y.transpose(0, 2, 1).reshape(n_samples_fa * m_fa, n_fa)
            mudata = mod.fit_transform(Y_fa)  #m*n_samples x d
            C = torch.tensor(mod.components_.T)  # (n x d)
            if learn_scale:
                _scale = rel_scale * torch.square(
                    C).mean().sqrt()  #global scale
            if learn_neuron_scale:
                _neuron_scale = rel_scale * torch.square(C).mean(
                    1).sqrt()  #per neuron
            if ard:
                _dim_scale = rel_scale * torch.square(C).mean(
                    0).sqrt()  #per latent
                if _dim_scale.numel() < d or torch.any(_dim_scale <= 0):
                    # FA cannot identify more factors than the data rank. Keep
                    # the remaining ARD dimensions positive and learnable.
                    fallback = _dim_scale.square().mean().sqrt()
                    fallback = torch.where(fallback > 0, fallback,
                                           torch.ones_like(fallback))
                    padded_scale = fallback.expand(d).clone()
                    padded_scale[:_dim_scale.numel()] = torch.where(
                        _dim_scale > 0, _dim_scale, fallback)
                    _dim_scale = padded_scale

        ##optionally provide these as params##
        scale = _scale if scale is None else scale
        dim_scale = _dim_scale if dim_scale is None else dim_scale
        neuron_scale = _neuron_scale if neuron_scale is None else neuron_scale

        self._scale = nn.Parameter(inv_softplus(scale),
                                   requires_grad=learn_scale)
        self._neuron_scale = nn.Parameter(inv_softplus(neuron_scale),
                                          requires_grad=learn_neuron_scale)
        self._dim_scale = nn.Parameter(inv_softplus(dim_scale),
                                       requires_grad=ard)

        #### initialize variational distribution (should we initialize this to the Gaussian ground truth?)####
        if q_mu is None:
            if tied_samples:
                q_mu = torch.zeros(1, n, d)
            else:
                q_mu = torch.zeros(n_samples, n, d)

        if q_sqrt is None:
            if tied_samples:
                q_sqrt = torch.diag_embed(torch.ones(1, n, d))
            else:
                q_sqrt = torch.diag_embed(torch.ones(n_samples, n, d))
        else:
            q_sqrt = transform_to(constraints.lower_cholesky).inv(q_sqrt)

        assert (q_mu is not None)
        assert (q_sqrt is not None)
        if self.tied_samples:
            assert (q_mu.shape[0] == 1)
            assert (q_sqrt.shape[0] == 1)
        else:
            assert (q_mu.shape[0] == n_samples)
            assert (q_sqrt.shape[0] == n_samples)

        self._q_mu = nn.Parameter(q_mu, requires_grad=True)
        self._q_sqrt = nn.Parameter(q_sqrt, requires_grad=True)

        self.likelihood = likelihood

    @property
    def scale(self):
        return softplus(self._scale)

    @property
    def neuron_scale(self):
        return softplus(self._neuron_scale)[:, None]

    @property
    def dim_scale(self):
        return softplus(self._dim_scale)[:, None]

    @property
    def q_mu(self):
        return self._q_mu

    @property
    def q_sqrt(self):
        return transform_to(constraints.lower_cholesky)(self._q_sqrt)

    def prior_kl(self, sample_idxs=None):
        """
        KL(p(f) || q(f))
        """
        q_mu, q_sqrt = self.prms
        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if not self.tied_samples and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]
        q = MultivariateNormal(q_mu, scale_tril=q_sqrt)
        e = torch.eye(self.d).to(q_mu.device)
        p_mu = torch.zeros(self.n, self.d).to(q_mu.device)
        prior = MultivariateNormal(p_mu, scale_tril=e)
        return kl_divergence(q, prior)  ##consider implementing this directly

    def elbo(self,
             y: Tensor,
             x: Tensor,
             sample_idxs: Optional[List[int]] = None,
             m: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        y : Tensor
            data tensor with dimensions (n_samples x n x m)
        x : Tensor (single kernel) or Tensor list (product kernels)
            input tensor(s) with dimensions (n_mc x n_samples x d x m)
        m : Optional int
            used to scale the svgp likelihood.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        lik, prior_kl : Tuple[torch.Tensor, torch.Tensor]
            lik has dimensions (n_mc x n)
            prior_kl has dimensions (n)
        """

        assert (x.shape[-3] == y.shape[-3])
        assert (x.shape[-1] == y.shape[-1])
        batch_size = x.shape[-1]
        sample_size = x.shape[-3]

        # prior KL(q(u) || p(u)) (1 x n) if tied_samples otherwise (n_samples x n)
        prior_kl = self.prior_kl(sample_idxs)
        # predictive mean and var at x
        f_mean, f_var = self.predict(x, full_cov=False, sample_idxs=sample_idxs)
        prior_kl = prior_kl.sum(-2)
        if not self.tied_samples:
            prior_kl = prior_kl * (self.n_samples / sample_size)

        #(n_mc, n_samles, n)
        lik = self.likelihood.variational_expectation(y, f_mean, f_var)
        # scale is (m / batch_size) * (self.n_samples / sample size)
        # to compute an unbiased estimate of the likelihood of the full dataset
        m = (self.m if m is None else m)
        scale = (m / batch_size) * (self.n_samples / sample_size)
        lik = lik.sum(-2)
        lik = lik * scale
        return lik, prior_kl


    def predict(self,
                x: Tensor,
                full_cov: bool,
                sample_idxs=None) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor (single kernel) or Tensor list (product kernels)
            test input tensor(s) with dimensions (n_b x n_samples x d x m)
        full_cov : bool
            returns full covariance if true otherwise returns the diagonal

        Returns
        -------
        mu : Tensor
            mean of predictive density at test inputs [ s ]
        v : Tensor
            variance/covariance of predictive density at test inputs [ s ]
            if full_cov is true returns full covariance, otherwise
            returns diagonal variance

        """

        q_mu, q_sqrt = self.prms

        assert (q_mu.shape[0] == q_sqrt.shape[0])
        if (not self.tied_samples) and sample_idxs is not None:
            q_mu = q_mu[sample_idxs]
            q_sqrt = q_sqrt[sample_idxs]

        x = self.scale * self.dim_scale * x  #multiply each dimension by the prior scale

        mu = q_mu.matmul(x)  # n_b x n_samples x n x m
        if not full_cov and self.d < self.n:
            # Contract latent pairs instead of materializing a neuron-by-latent
            # activation for every trial/time bin. Both compute x^T L L^T x.
            covariance = q_sqrt.matmul(q_sqrt.transpose(-1, -2))
            latent_pairs = (x.unsqueeze(-2) * x.unsqueeze(-3)).flatten(-3, -2)
            variance = covariance.flatten(-2).matmul(latent_pairs)
            return mu, variance.clamp_min(0)
        l = x[..., None, :, :].transpose(-1, -2).matmul(
            q_sqrt)  # n_b x n_samples x m x d
        if not full_cov:
            return mu, torch.square(l).sum(-1)
        else:
            return mu, l.matmul(l.transpose(-1, -2))

    @property
    def prms(self) -> Tuple[Tensor, Tensor]:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt

        #multiply the posterior by a scale factor for each neuron
        q_mu, q_sqrt = self.neuron_scale * q_mu, self.neuron_scale[
            ..., None] * q_sqrt
        return q_mu, q_sqrt

    def g0_parameters(self):
        return [self._q_mu, self._q_sqrt]

    def g1_parameters(self):
        return list(
            itertools.chain.from_iterable([
                self.likelihood.parameters(),
                [self._scale, self._neuron_scale, self._dim_scale]
            ]))


# Circulant GP posterior

class GPbase(nn.Module):
    name = "GPbase"  # it is important that child classes have "GP" in their name, this is used in control flow

    def __init__(self,
                 d: int,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 _scale=0.9,
                 ell=None):
        """
        Parameters
        ----------
        d: int
            latent dimensionality
        m : int
            number of conditions/timepoints
        n_samples: int
            number of samples
        ts: Tensor
            input timepoints for each sample (n_samples x 1 x m)

        Notes
        -----
        Our GP has prior N(0, K)
        We parameterize our posterior as N(K2 v, K2 I^2 K2)
        where K2 K2 = K and I(s) is some inner matrix which can take different forms.
        s is a vector of scale parameters for each time point.

        """

        super().__init__()

        self.d = d
        self.m = m

        #initialize GP mean parameters
        nu = torch.randn((n_samples, self.d, m)) * 0.01
        self._nu = nn.Parameter(data=nu, requires_grad=True)  #m in the notes

        #initialize covariance parameters
        _scale = torch.ones(n_samples, self.d, m) * _scale  #n_diag x T
        self._scale = nn.Parameter(data=inv_softplus(_scale),
                                   requires_grad=True)

        #initialize length scale
        if ell is None:
            _ell = torch.ones(1, self.d,
                              1) * (torch.max(ts) - torch.min(ts)) / 20
        else:
            if type(ell) in [float, int]:
                _ell = torch.ones(1, self.d, 1) * ell
            else:
                _ell = ell
        self._ell = nn.Parameter(data=inv_softplus(_ell), requires_grad=True)

        #pre-compute time differences (only need one row for the toeplitz stuff)
        self.ts = ts
        dts_sq = torch.square(ts - ts[..., :1])  #(n_samples x 1 x m)
        #sum over _input_ dimension, add an axis for _output_ dimension
        dts_sq = dts_sq.sum(-2)[:, None, ...]  #(n_samples x 1 x m)
        self.dts_sq = nn.Parameter(data=dts_sq, requires_grad=False)

        self.dt = (ts[0, 0, 1] - ts[0, 0, 0]).item()  #scale by dt

    @property
    def scale(self) -> torch.Tensor:
        return softplus(self._scale)

    @property
    def nu(self) -> torch.Tensor:
        return self._nu

    @property
    def ell(self) -> torch.Tensor:
        return softplus(self._ell)

    @property
    def prms(self):
        return self.nu, self.scale, self.ell

    @property
    def lat_mu(self):
        """return variational mean mu = K_half @ nu"""
        nu = self.nu
        K_half = self.K_half()  #(n_samples x d x m)
        mu = sym_toeplitz_matmul(K_half, nu[..., None])[..., 0]
        return mu.transpose(-1, -2)  #(n_samples x m x d)

    def K_half(self, sample_idxs=None):
        """compute one column of the square root of the prior matrix"""
        #K^(1/2) has length scale ell/sqrt(2) if K has ell
        sqrt_two = self.ell.new_tensor(math.sqrt(2.0))
        ell_half = self.ell / sqrt_two

        #K^(1/2) has sig var sig*2^1/4*pi^(-1/4)*ell^(-1/2) if K has sig^2 (1 x d x 1)
        scale = self.ell.new_tensor((2**0.25) * (math.pi**-0.25) * (self.dt**0.5))
        sig_sqr_half = scale * self.ell.pow(-0.5)

        if (sample_idxs is None) or (self.dts_sq.shape[0] == 1):
            dts = self.dts_sq[:, ...]
        else:
            dts = self.dts_sq[sample_idxs, ...]

        # (n_samples x d x m)
        K_half = sig_sqr_half * torch.exp(-dts / (2 * torch.square(ell_half)))

        return K_half

    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        This should be implemented for each class separately.
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        raise NotImplementedError

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        raise NotImplementedError

    def sample(self,
               size,
               Y=None,
               batch_idxs=None,
               sample_idxs=None,
               kmax=5,
               analytic_kl=False,
               prior=None):
        """
        generate samples and computes its log entropy
        """

        #compute KL analytically
        lq = self.kl(batch_idxs=batch_idxs,
                     sample_idxs=sample_idxs)  #(n_samples x d)

        K_half = self.K_half(sample_idxs=sample_idxs)  #(n_samples x d x m)
        n_samples, d, m = K_half.shape

        # sample a batch with dims: (n_samples x d x m x n_mc)
        v = torch.randn(n_samples, d, m, size[0])  # v ~ N(0, 1)
        #compute I @ v (n_samples x d x m x n_mc)
        I_v = self.I_v(v, sample_idxs=sample_idxs)

        nu = self.nu  #mean parameter (n_samples, d, m)
        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
        samp = nu[..., None] + I_v  #add mean parameter to each sample

        #compute K@(I@v+nu)
        x = sym_toeplitz_matmul(K_half, samp)  #(n_samples x d x m x n_mc)
        x = x.permute(-1, 0, 2, 1)  #(n_mc x n_samples x m x d)

        if batch_idxs is not None:  #only select some time points
            x = x[..., batch_idxs, :]

        #(n_mc x n_samples x m x d), (n_samples x d)
        return x, lq

    def gmu_parameters(self):
        return [self._nu]

    def concentration_parameters(self):
        return [self._scale, self._ell]


class GP_circ(GPbase):
    name = "GP_circ"

    def __init__(self,
                 d: int,
                 m: int,
                 n_samples: int,
                 ts: torch.Tensor,
                 _scale=0.9,
                 ell=None):
        """
        Parameters
        ----------
        d: int
            latent dimensionality
        m : int
            number of conditions/timepoints
        n_samples: int
            number of samples
        ts: Tensor
            input timepoints for each sample (n_samples x 1 x m)

        Notes
        -----
        We parameterize our posterior as N(K2 v, K2 SCCS K2) where K2@K2 = Kprior, S is diagonal and C is circulant
        """

        super(GP_circ, self).__init__(d,
                                      m,
                                      n_samples,
                                      ts,
                                      _scale=_scale,
                                      ell=ell)

        #initialize circulant parameters
        if self.m % 2 == 0:
            _c = torch.ones(n_samples, self.d, int(m / 2) + 1)
        else:
            _c = torch.ones(n_samples, self.d, int((m + 1) / 2))
        self._c = nn.Parameter(data=inv_softplus(_c), requires_grad=True)

    @property
    def c(self) -> torch.Tensor:
        return softplus(self._c)

    @property
    def prms(self):
        return self.nu, self.scale, self.ell, self.c

    def I_v(self, v, sample_idxs=None):
        """
        Compute I @ v for some vector v.
        Here I = S C.
        v is (n_samples x d x m x n_mc) where n_samples is the number of sample_idxs
        """
        scale, c = self.scale, self.c
        if sample_idxs is not None:
            scale = scale[sample_idxs, ...]  #(n_samples x d x m)
            c = c[sample_idxs, ...]  #(n_samples x d x m/2)

        #Fourier transform (n_samples x d x n_mc x m/2)
        rv = rfft(v.transpose(-1, -2).to(scale.device))

        #inverse fourier transform of product (n_samples x d x m x n_mc)
        Cv = irfft(c[..., None, :] * rv, n=self.m).transpose(-1, -2)

        #multiply by diagonal scale
        SCv = scale[..., None] * Cv


        return SCv

    def kl(self, batch_idxs=None, sample_idxs=None):
        """
        Compute KL divergence between prior and posterior.
        This should be implemented for each class separately
        """
        #(n_samples x d x m), (n_samples x d x m), (n_samples x d x m/2)
        nu, S, c = self.nu, self.scale, self.c

        if sample_idxs is not None:
            nu = nu[sample_idxs, ...]
            S = S[sample_idxs, ...]
            c = c[sample_idxs, ...]

        #n_samples x d x m
        Cr = irfft(self.c,
                   n=self.m)  #first row of C given by inverse Fourier transform

        #(n_samples x d)
        TrTerm = torch.square(S).sum(-1) * torch.square(Cr).sum(-1)
        MeanTerm = torch.square(nu).sum(-1)  #(n_samples x d)
        DimTerm = S.shape[-1]
        LogSTerm = 2 * (torch.log(S)).sum(-1)  #(n_samples x d)

        #c[0] + 2*c[1:end] (n_samples x d)
        LogCTerm = 2 * (torch.log(c)).sum(-1) - torch.log(c[..., 0])
        if self.m % 2 == 0:
            #c[0] + c[-1] + 2*c[1:-1]
            LogCTerm = LogCTerm - torch.log(c[..., -1])
        LogCTerm = 2 * LogCTerm  #one for each C

        kl = 0.5 * (TrTerm + MeanTerm - DimTerm - LogSTerm - LogCTerm)
        if batch_idxs is not None:  #scale by batch size
            kl = kl * len(batch_idxs) / self.m

        return kl

    def gmu_parameters(self):
        return [self._nu, self._c]


# Joint latent and observation model

class Lvgplvm(nn.Module):
    """BGPFA's linear observation model and circulant GP latent posterior."""

    name = "Lvgplvm"

    def __init__(self, n: int, m: int, d: int, n_samples: int,
                 lat_dist: GP_circ, likelihood: Likelihood, *,
                 Y=None, learn_scale=None, ard=False, rel_scale=1):
        super().__init__()
        self.obs = Bvfa(n, d, m, n_samples, likelihood, Y=Y,
                        learn_scale=learn_scale, ard=ard, rel_scale=rel_scale)
        # Preserve the observation alias and state-dict keys of existing runs.
        self.svgp = self.obs
        self.n = n
        self.m = m
        self.n_samples = n_samples
        self.lat_dist = lat_dist

    def elbo(self,
             data,
             n_mc,
             kmax=5,
             batch_idxs=None,
             sample_idxs=None,
             neuron_idxs=None,
             m=None,
             analytic_kl=False):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_mc : int
            number of MC samples
        kmax : int
            Retained from the reference call signature; unused for GP_circ.
        batch_idxs : Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))
        sample_idxs : Optional int list
            if None then use all data
            otherwise, compute elbo only for selected samples
        neuron_idxs: Optional int list
            if None then use all data
            otherwise, compute only elbo for selected neurons
        m : Optional int
            used to scale the svgp likelihood and sgp prior.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        svgp_elbo : Tensor
            evidence lower bound of sparse GP per neuron, batch and sample (n_mc x n)
            note that this is the ELBO for the batch which is proportional to an unbiased estimator for the data.
        kl : Tensor
            estimated KL divergence per batch between variational distribution and prior (n_mc)

        Notes
        -----
        ELBO of the model per batch is [ svgp_elbo - kl ]
        """

        n_samples = self.n_samples
        m = (self.m if m is None else m)

        g, lq = self.lat_dist.sample(torch.Size([n_mc]),
                                     data,
                                     batch_idxs=batch_idxs,
                                     sample_idxs=sample_idxs,
                                     kmax=kmax,
                                     analytic_kl=analytic_kl)
        # g is shape (n_mc, n_samples, m, d)
        # lq is the analytic KL per trial and latent dimension.

        # note that [ obs.elbo ] recognizes inputs of dims (n_mc x d x m)
        # and so we need to permute [ g ] to have the right dimensions
        #(n_mc x n), (1 x n)
        svgp_lik, svgp_kl = self.obs.elbo(data,
                                          g.transpose(-1, -2),
                                          sample_idxs,
                                          m=m)  #p(Y|g)
        if neuron_idxs is not None:
            svgp_lik = svgp_lik[..., neuron_idxs]
            svgp_kl = svgp_kl[..., neuron_idxs]
        lik = svgp_lik - svgp_kl

        # GP_circ returns the analytic KL, shared by all Monte Carlo samples.
        kl = torch.ones(n_mc).to(data.device) * lq.sum()

        #rescale KL to entire dataset (basically structured conditions)
        batch_size = m if batch_idxs is None else len(batch_idxs)
        sample_size = n_samples if sample_idxs is None else len(sample_idxs)
        kl = (m / batch_size) * (n_samples / sample_size) * kl

        return lik, kl

    def forward(self,
                data,
                n_mc,
                kmax=5,
                batch_idxs=None,
                sample_idxs=None,
                neuron_idxs=None,
                m=None,
                analytic_kl=False):
        """
        Parameters
        ----------
        data : Tensor
            data with dimensionality (n_samples x n x m)
        n_mc : int
            number of MC samples
        kmax : int
            Retained from the reference call signature; unused for GP_circ.
        batch_idxs: Optional int list
            if None then use all data and (batch_size == m)
            otherwise, (batch_size == len(batch_idxs))
        sample_idxs : Optional int list
            if None then use all data
            otherwise, compute elbo only for selected samples
        neuron_idxs: Optional int list
            if None then use all data
            otherwise, compute only elbo for selected neurons
        m : Optional int
            used to scale the svgp likelihood and sgp prior.
            If not provided, self.m is used which is provided at initialization.
            This parameter is useful if we subsample data but want to weight the prior as if it was the full dataset.
            We use this e.g. in crossvalidation

        Returns
        -------
        elbo : Tensor
            evidence lower bound of the GPLVM model averaged across MC samples and summed over n, m, n_samples (scalar)
        """

        #(n_mc, n), (n_mc)
        lik, kl = self.elbo(data,
                            n_mc,
                            kmax=kmax,
                            batch_idxs=batch_idxs,
                            sample_idxs=sample_idxs,
                            neuron_idxs=neuron_idxs,
                            m=m,
                            analytic_kl=analytic_kl)
        #sum over neurons and mean over  MC samples
        lik = lik.sum(-1).mean()
        kl = kl.mean()

        return lik, kl  #mean across batches, sum across everything else


# Parameter grouping and held-out posterior optimization

def sort_params(model, hook):
    """Group mean and covariance parameters for the reference burn-in schedule."""
    hooks = [model.lat_dist.nu.register_hook(hook),
             model.lat_dist._scale.register_hook(hook)]
    params0 = [*model.lat_dist.gmu_parameters(), *model.svgp.g0_parameters()]
    params1 = [*model.lat_dist.concentration_parameters(), *model.svgp.g1_parameters()]
    return [{'params': params0}, {'params': params1}], hooks


def fit_latents(model, data, *, max_steps, n_mc, lrate, burnin):
    """Optimize an evaluation posterior with frozen observation/prior parameters."""
    params, hooks = sort_params(model, lambda grad: grad * 1)
    try:
        optimizer = torch.optim.Adam(params, lr=lrate)
        scheduler = LambdaLR(optimizer, lr_lambda=[
            lambda step: 1,
            lambda step: 1 - np.exp(-step / (3 * burnin)),
        ])
        # Match the full-batch indices supplied by the original BatchDataLoader.
        sample_idxs = list(range(data.shape[0]))
        batch_idxs = list(range(data.shape[-1]))
        for step in range(max_steps):
            elbo, kl = model(data, n_mc, sample_idxs=sample_idxs, batch_idxs=batch_idxs)
            loss = -elbo + (1 - np.exp(-step / burnin)) * kl
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    finally:
        for hook in hooks:
            hook.remove()
