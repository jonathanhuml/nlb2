"""Linear variational GP latent model adapted from mgplvm (see LICENSE)."""

import torch
from torch import nn

from .latent import GP_circ
from .likelihoods import Likelihood
from .observation import Bvfa


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
