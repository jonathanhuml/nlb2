"""Pre-specified protocol for the v2 Figure 1 readout experiment.

Written before any v2 test metric existed. Everything that decides what gets reported lives
here. `protocol_sha256()` hashes this file; the lockfile records the hash and evaluate_v2.py
refuses to run if the file changed after locking.

Design
  Representations (frozen, from Stage 1): lfads, mtm, mtm_random, plus smoothed_spikes (control).
  Tasks
    wheel     per-bin wheel speed, z-scored with training statistics, MSE loss.
              Selection metric: validation R^2. Reported metric: test R^2.
    cosmooth  per-bin held-out neuron counts from held-in-driven features, Poisson loss.
              Selection metric: validation pooled co-bps. Reported metrics: test pooled co-bps
              (all 50 bins) and test mean-rate MAE in Hz (first 500 ms), as in Stage 1.
  Readout families (weights shared across the 50 bins)
    linear       wheel: closed-form ridge over RIDGE_ALPHA. cosmooth: Poisson GLM over GLM_L2.
    mlp          architecture from MLP_CONFIGS, optimizer settings from OPT_SEARCH.
    transformer  architecture from TRANSFORMER_CONFIGS, optimizer settings from OPT_SEARCH.
  Splits: the BWB TS1 causal train/val/test windows and the fixed SHA-256 held-out-unit mask
  from figure1_data.py. Neither is changed in v2.

Architecture screen (cross-fitted, leave-one-mouse-out)
  Every MLP and transformer architecture is fit at seed 0 on every session, scored on
  validation data only. For a session from mouse m, the architecture is the one with the
  highest equal-session mean validation metric over sessions from the other 12 mice. So no
  session's own data, or its mouse's data, chooses its architecture; only its optimizer
  settings (and ridge/GLM penalty) are tuned on its own validation split.

Pre-specified analysis (stats_v2.py)
  Unit: session (n = 29, 13 mice); seeds averaged within session first.
  Effect: MtM - LFADS paired per session. CI: 95% hierarchical bootstrap (mice, then
  sessions), 10,000 replicates. Test: two-sided Wilcoxon signed-rank over sessions.
  Panel A (co-smoothing): {co-bps, MAE} x {linear, mlp, transformer}: 6 tests, Holm over 6.
  Panel B (wheel R^2): {linear, mlp, transformer}: 3 tests, Holm over 3.
  Panel C (representation x readout interaction): for each outcome {wheel R^2, co-bps, MAE},
    a Friedman test across the three readouts of the per-session difference (MtM - LFADS):
    3 tests, Holm over 3. Secondary: pairwise Wilcoxon contrasts of that difference.
  Secondary (not Holm-adjusted with the above): readout main effect within each representation
  (Friedman across readouts), mouse-level Wilcoxon (n = 13) as a sensitivity check, controls
  (mtm_random, smoothed_spikes), and the Stage 1 native co-smoothing readouts.
"""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'v2'
FITS = V2 / 'fits'

TASKS = ['wheel', 'cosmooth']
FEATURES = ['lfads', 'mtm', 'mtm_random', 'smoothed_spikes']
PRIMARY = ('lfads', 'mtm')
FAMILIES = ['linear', 'mlp', 'transformer']
SEEDS = [0, 1, 2]
SCREEN_SEED = 0
SMOOTH_SIGMA_S = 0.05

RIDGE_ALPHA = [float(a) for a in np.logspace(-4, 4, 17)]
GLM_L2 = [float(a) for a in np.logspace(-8, 0, 9)]
OPT_SEARCH = [(1e-3, .01), (1e-3, .1), (3e-4, .01), (3e-4, .1)]  # (learning rate, weight decay)
MAX_EPOCHS, MIN_EPOCHS, PATIENCE, BATCH = 300, 40, 30, 32
DROPOUT = .1

MLP_CONFIGS = {
    'd2_w128': dict(depth=2, width=128),  # v1 readout
    'd1_w128': dict(depth=1, width=128),
    'd1_w256': dict(depth=1, width=256),
    'd2_w64': dict(depth=2, width=64),
    'd2_w256': dict(depth=2, width=256),
    'd3_w128': dict(depth=3, width=128),
}
TRANSFORMER_CONFIGS = {
    'd2_w256_h4': dict(depth=2, width=256, heads=4),  # v1 baseline
    'd1_w256_h4': dict(depth=1, width=256, heads=4),
    'd4_w256_h4': dict(depth=4, width=256, heads=4),
    'd6_w256_h4': dict(depth=6, width=256, heads=4),
    'd2_w128_h4': dict(depth=2, width=128, heads=4),
    'd2_w512_h4': dict(depth=2, width=512, heads=4),
    'd2_w256_h2': dict(depth=2, width=256, heads=2),
    'd2_w256_h8': dict(depth=2, width=256, heads=8),
}
SELECTION_METRIC = {'wheel': 'val_r2', 'cosmooth': 'val_co_bps'}
BOOTSTRAP_REPS, BOOTSTRAP_SEED = 10_000, 20261120


def names(family):
    if family == 'linear':
        return ['linear']
    configs = MLP_CONFIGS if family == 'mlp' else TRANSFORMER_CONFIGS
    return [f'{family}_{c}' for c in configs]


def family_of(name):
    return name.split('_', 1)[0]


def eids():
    return (ROOT / 'eval_recording_ids.txt').read_text().split()


def protocol_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
