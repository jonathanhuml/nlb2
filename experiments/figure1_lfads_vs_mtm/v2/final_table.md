| Panel | Metric | Readout | LFADS | MtM | MtM − LFADS [95% CI] | MtM better | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|---|---|
| A | co-bps | linear | 0.479 | 0.458 | -0.021 [-0.063, +0.035] | 9/29 | 0.019 | 0.095 |
| A | co-bps | mlp | 0.448 | 0.443 | -0.005 [-0.055, +0.067] | 11/29 | 0.15 | 0.45 |
| A | co-bps | transformer | 0.484 | 0.494 | +0.010 [-0.040, +0.068] | 11/29 | 0.43 | 0.61 |
| A | MAE (Hz) | linear | 6.987 | 7.455 | +0.468 [+0.072, +1.089] | 6/29 | 0.0065 | 0.039 |
| A | MAE (Hz) | mlp | 6.942 | 7.088 | +0.146 [-0.050, +0.336] | 8/29 | 0.039 | 0.15 |
| A | MAE (Hz) | transformer | 6.998 | 7.090 | +0.092 [-0.131, +0.336] | 10/29 | 0.3 | 0.61 |
| B | wheel R² | linear | 0.245 | 0.198 | -0.047 [-0.077, -0.012] | 5/29 | 0.0038 | 0.0047 |
| B | wheel R² | mlp | 0.271 | 0.228 | -0.042 [-0.073, -0.010] | 6/29 | 0.0023 | 0.0047 |
| B | wheel R² | transformer | 0.307 | 0.232 | -0.075 [-0.109, -0.044] | 4/29 | 1.7e-06 | 5.0e-06 |

| Panel C: interaction | gap (MtM − LFADS) linear / MLP / transformer | Friedman χ² | p | Holm p |
|---|---|---|---|---|
| co-bps | -0.021 / -0.005 / +0.010 | 3.38 | 0.18 | 0.55 |
| MAE (Hz) | +0.468 / +0.146 / +0.092 | 2.14 | 0.34 | 0.69 |
| wheel R² | -0.047 / -0.042 / -0.075 | 1.31 | 0.52 | 0.69 |

Contrasts of the gap (Wilcoxon, Holm within metric):
- co-bps: gap(mlp - linear) = +0.016 [-0.008, +0.045], p = 0.29, Holm p = 0.46
- co-bps: gap(transformer - linear) = +0.031 [-0.002, +0.073], p = 0.039, Holm p = 0.12
- co-bps: gap(transformer - mlp) = +0.015 [-0.035, +0.061], p = 0.23, Holm p = 0.46
- MAE (Hz): gap(mlp - linear) = -0.322 [-0.893, +0.008], p = 0.08, Holm p = 0.24
- MAE (Hz): gap(transformer - linear) = -0.376 [-1.051, +0.045], p = 0.13, Holm p = 0.26
- MAE (Hz): gap(transformer - mlp) = -0.054 [-0.204, +0.087], p = 0.62, Holm p = 0.62
- wheel R²: gap(mlp - linear) = +0.005 [-0.027, +0.028], p = 0.4, Holm p = 0.4
- wheel R²: gap(transformer - linear) = -0.028 [-0.068, +0.008], p = 0.19, Holm p = 0.38
- wheel R²: gap(transformer - mlp) = -0.033 [-0.070, +0.002], p = 0.13, Holm p = 0.38

Session = unit (n=29, 13 mice); seeds averaged within session. Diff = MtM - LFADS. CI = 95% hierarchical bootstrap (mice, then sessions). p = two-sided Wilcoxon signed-rank over sessions; Holm within panel. Panel C: Friedman test across readouts of the per-session MtM - LFADS gap.