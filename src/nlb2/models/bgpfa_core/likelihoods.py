"""Gaussian and Poisson likelihoods adapted from mgplvm (see LICENSE)."""

from typing import Optional

import numpy as np
from numpy.polynomial.hermite import hermgauss
from sklearn import decomposition
import torch
from torch import Tensor, nn
import torch.distributions as dists

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
