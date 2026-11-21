"""Pre-specified extension (goal rule 6): more BWB TS1 behaviors on the same 29 sessions.

Frozen on 2026-11-20 before any v2 test metric existed, as for the primary protocol (protocol_v2.py).
Separate files keep the primary analysis unchanged: v2/fits_ext/, v2/selection_ext.json,
v2/lockfile_ext.json, v2/test_metrics_ext.csv, v2/final_table_ext.md.

Tasks: per-bin regression on the same movement-aligned 1 s windows (50 x 20 ms bins), with the
same frozen Stage 1 representations, readout families, search spaces, z-scoring, MSE loss and
leave-one-mouse-out architecture screen as the v2 wheel task.
  whisker  whisker/whisker_motion_energy
  rpaw     paws/right_paw_speed
  lpaw     paws/left_paw_speed
  licks    licks/licking_rate
Targets are linearly interpolated at bin centres, as for wheel in figure1_data.py. A window is used
for a task only if all 50 bin centres fall inside that signal's recorded domain. All 29 sessions
pass BWB behavior QC for all four signals (checked 2026-11-20).
Metric: test R^2 for all four. BWB scores licks with Poisson D^2; R^2 is used here so the readout
and loss are identical across behaviors. This is a stated deviation.
Not included: choice, contrast and reward. These are sequence-level classification targets and
would need a different readout form (pooling across bins), so they are not comparable with the
per-bin readouts.
Representations: lfads, mtm, mtm_random. smoothed_spikes is dropped here to save compute.

Analysis (stats on v2/test_metrics_ext.csv, same conventions as protocol_v2):
  Panel D: MtM - LFADS for each task x readout (4 x 3 = 12 tests), Wilcoxon, Holm over 12.
  Panel E: interaction. Per task, a Friedman test across readouts of the per-session gap; Holm over 4.
  Secondary: readout main effect within each representation; mtm vs mtm_random.
"""
import hashlib
from pathlib import Path

import protocol_v2 as P

TASKS = {
    'whisker': ('whisker', 'whisker_motion_energy'),
    'rpaw': ('paws', 'right_paw_speed'),
    'lpaw': ('paws', 'left_paw_speed'),
    'licks': ('licks', 'licking_rate'),
}
FEATURES = ['lfads', 'mtm', 'mtm_random']
FITS = P.V2 / 'fits_ext'


def protocol_sha256():
    """Hash of both protocol files: the extension inherits every primary setting."""
    h = hashlib.sha256(Path(__file__).read_bytes())
    h.update(Path(P.__file__).read_bytes())
    return h.hexdigest()
