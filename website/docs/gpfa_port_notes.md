# GPFA Port Notes

The local MATLAB reference is `/Users/jonathanhuml/Desktop/gpfa-matlab`.
The Python Elephant reference is:
`https://github.com/NeuralEnsemble/elephant/tree/master/elephant/gpfa`.

The implementation in `nlb2.models.gpfa` is being moved toward the Elephant
Python structure while keeping tensors in PyTorch:

- `gpfaEngine.m`: initialize `gamma`, `eps`, `C`, `d`, and diagonal `R`.
- `fastfa.m`: initialize `C`, `d`, and `R` with factor analysis.
- `exactInferenceWithLL.m`: run the GPFA E-step and compute likelihood.
- `em.m`: update `C`, `d`, `R`, then learn GP kernel parameters.
- `learnGPparams.m` and `grad_betgam.m`: update each RBF `gamma` from
  posterior autocovariance sufficient statistics.

Current Elephant-aligned choices:

- Elephant uses `sklearn.decomposition.FactorAnalysis`; NLB2 ports the same
  FA EM objective in torch to avoid adding sklearn to the model dependency
  surface.
- Elephant uses SciPy `L-BFGS-B` for `grad_betgam`; NLB2 uses PyTorch
  autograd with `torch.optim.LBFGS` on the equivalent objective.
- Elephant exposes both raw latent trajectories and `latent_variable_orth`.
  NLB2 now returns raw latents in `ModelOutput.latents` and
  orthonormalized latents plus `Corth` in `ModelOutput.extras`.

Orthonormalization is postprocessing for latent visualization. It should not
change decoded firing-rate predictions unless the latent trajectories and the
loading matrix are transformed consistently.

The benchmark epoch definition is unchanged: one GPFA epoch is one full E-step,
one C/d/R M-step, and the associated `gamma` optimization using the E-step
sufficient statistics.
