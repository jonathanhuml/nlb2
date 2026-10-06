"""Numerical core used by NLB2's Bayesian GPFA model.

Adapted from tachukao/mgplvm-pytorch (MIT; see LICENSE). Only the circulant
Euclidean GP posterior, variational Bayesian factor analysis, Gaussian/Poisson
likelihoods, and full-batch optimization needed by BGPFA are retained. This is
an internal implementation, not a general-purpose mgplvm compatibility layer.
"""
