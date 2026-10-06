"""BGPFA parameter groups and full-batch posterior inference.

Adapted from mgplvm's optimisers/svgp.py (MIT; see LICENSE). The inference
loop retains its KL ramp, parameter ordering, and covariance learning-rate
burn-in without importing the general cross-validation/data-loader framework.
"""

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


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
