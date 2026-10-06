"""Variational Bayesian factor analysis adapted from mgplvm (see LICENSE)."""

import itertools
from typing import List, Optional, Tuple

from sklearn import decomposition
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, kl_divergence, transform_to, constraints

from .likelihoods import Likelihood
from .numerics import softplus, inv_softplus


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
