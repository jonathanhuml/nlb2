# NLB2

<p align="center">
  <img src="https://zkunkworks.com/nlb2/assets/lorenz.png" alt="NLB2 Lorenz attractor logo" width="240">
</p>

NLB2 is a Python library for fitting and comparing
models of neural population dynamics through a shared PyTorch API.

## Models

| Model | Citation | Description |
| --- | --- | --- |
| bGPFA | [Jensen et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/58238e9ae2dd305d79c2ebc8c1883422-Abstract.html) | Bayesian Gaussian process factor analysis with automatic relevance determination, scalable variational inference, and support for non-Gaussian observation noise. |
| CASSM | [Huml et al., 2026](https://arxiv.org/abs/2606.01468) | Learns low-dimensional projections for Kalman filtering while accounting for uncertainty introduced by approximate computation. |
| GPFA | [Yu et al., 2009](https://doi.org/10.1152/jn.90941.2008) | Combines temporal smoothing and dimensionality reduction in a probabilistic model with Gaussian process latent trajectories. |
| iLQR-VAE | [Schimel et al., 2022](https://openreview.net/forum?id=wRODLDHaAiW) | Sequential variational autoencoder that uses iterative linear quadratic regulation to infer latent dynamics, initial conditions, and external inputs. |
| Kalman filter | [Kalman, 1960](https://doi.org/10.1115/1.3662552) | Exact inference in linear Gaussian state-space models; the NLB2 implementation uses Matérn dynamics to represent smooth neural trajectories. |
| LangevinFlow | [Song et al., 2025](https://arxiv.org/abs/2507.11531) | Sequential variational autoencoder with underdamped Langevin latent dynamics and a learned potential built from locally coupled oscillators. |
| LFADS | [Pandarinath et al., 2018](https://doi.org/10.1038/s41592-018-0109-9) | Recurrent sequential variational autoencoder that infers single-trial neural dynamics, initial conditions, and time-varying inputs. |
| MINT | [Perkins et al., 2025](https://elifesciences.org/articles/89421) | Matches observed spikes to a learned library of neural trajectories using Poisson likelihoods and interpolation. |
| NDT | [Ye & Pandarinath, 2021](https://doi.org/10.51628/001c.27358) | Transformer that learns neural activity representations by reconstructing masked spike counts with temporal self-attention. |
| PSTH | [Palmer & Ashby, 1992](https://pubmed.ncbi.nlm.nih.gov/20870522/) | Estimates event-aligned firing rates by averaging spike counts across repeated trials within each condition. |
| Gaussian smoothing | [Shimazaki & Shinomoto, 2007](https://doi.org/10.1162/neco.2007.19.6.1503) (histogram binning) | Estimates firing rates by convolving spike counts with a Gaussian kernel of configurable width. |
| STNDT | [Le & Shlizerman, 2022](https://arxiv.org/abs/2206.04727) | Extends NDT with attention across both time and neurons, combining masked modeling with contrastive learning. |

## Quickstart

Install from the repository root, including plotting dependencies:

```bash
python -m pip install -e ".[benchmarks]"
```

With a CUDA-enabled PyTorch installation, train GPFA on synthetic Lorenz data:

```bash
nlb2 run -d lorenz -m gpfa --device cuda \
  --epochs 20 --batch-size 8 --live-eval-interval 5 \
  --run-name gpfa_lorenz
```

This generates data locally, reports training and validation losses, and
evaluates reconstruction metrics every five epochs. Each run saves its resolved
configuration, learning history, fitted model, metrics, and predictions under
`runs/`.

Use a YAML recipe to specify the full experiment, with CLI overrides for the
device and training budget:

```bash
nlb2 run -c configs/experiment/synthetic/lorenz/gpfa/gpfa_lorenz.yaml \
  --device cuda --epochs 20
```

## Capabilities

Use the same interface for synthetic Lorenz and chaotic-RNN experiments, NLB
benchmarks, and prepared Allen VCN or CTD datasets.

| Task | Command |
| --- | --- |
| Discover available models and datasets | `nlb2 list models` / `nlb2 list datasets` |
| Train and evaluate a configured experiment | `nlb2 run -c experiment.yaml --device cuda` |
| Run several experiment configs sequentially | `nlb2 run -c first.yaml second.yaml third.yaml --device cuda` |
| Search hyperparameters with Ray Tune | `nlb2 tune -c study.yaml` |
| Continue an interrupted tuning study | `nlb2 tune --resume-from studies/my_study` |
| Download and prepare NLB validation data | `nlb2 prepare-nlb --datasets mc_maze --splits val --bin-sizes-ms 5 --download` |
| Score saved NLB predictions | `nlb2 score-nlb --run-dir runs/my_nlb_run` |

Use `nlb2 --help` or `nlb2 <command> --help` for options. Tuning supports random
search and Optuna TPE, repeated training seeds, and validation-based selection;
see the [tuning guide](website/docs/tuning.md) for installation and study recipes.

## Python API

`ExperimentConfig` combines dataset, model, preprocessing, and trainer settings.
Model configs build PyTorch modules, `DataModule` prepares data and batches, and
`Experiment` handles training, evaluation, and saved results. Python and the CLI
use the same experiment configuration and execution path:

```python
from nlb2 import Experiment

experiment = Experiment.from_config_path(
    "configs/experiment/synthetic/lorenz/gpfa/gpfa_lorenz.yaml"
)
experiment.config.trainer.device = "cuda"
result = experiment.run()
print(result.metrics)
```

`Study` adds a search space and validation objective to an experiment config.
It runs trials through Ray Tune and exports `best_config.yaml`, which can be
run through the ordinary `Experiment` API or `nlb2 run` command.

## Learn more

- [Lorenz tutorial](tutorials/lorenz.ipynb): configure an experiment, train a model,
  and plot learning curves and reconstructed activity.
- [Documentation](https://zkunkworks.com/nlb2/): model reference, configuration,
  and hyperparameter tuning.
