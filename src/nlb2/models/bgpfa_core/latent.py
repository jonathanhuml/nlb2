"""Circulant variational GP posterior adapted from mgplvm (see LICENSE)."""

import math

import torch
from torch import nn
from torch.fft import rfft, irfft

from .numerics import softplus, inv_softplus, sym_toeplitz_matmul


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
