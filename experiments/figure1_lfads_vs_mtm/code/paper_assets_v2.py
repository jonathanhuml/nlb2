"""Write Figure 1 (v2) paper assets from the locked stats outputs; presentation only, no fitting.
The Figure 1 caption is written by hand in the paper and is not generated here.

  python code/paper_assets_v2.py [--paper-dir DIR]     (default: figures/paper; `make paper PAPER_DIR=...`)

Outputs (all regenerated from v2/final_stats.json, v2/final_stats_ext.json, figures/figure1_v2.pdf):
  figure1_v2.pdf                 copy of the rendered figure
  table_figure1_primary.tex      co-smoothing (co-bps, MAE) and wheel R^2, LFADS vs MtM per readout
  table_figure1_ext.tex          whisker, right paw, left paw, licking R^2, LFADS vs MtM per readout
  table_figure1_interaction.tex  model x readout interaction tests for all seven outcomes
  SOURCES.json                   SHA-256 of every input, so the paper copy can be traced to the locked run
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['linear', 'mlp', 'transformer']
LABELS = {'linear': 'Linear', 'mlp': 'MLP', 'transformer': 'Transformer'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(x, nd):
    return f'${x:.{nd}f}$'


def signed(x, nd):
    return f'${x:+.{nd}f}$'


def pfmt(p, bold=False):
    if p < 1e-3:
        m, e = f'{p:.1e}'.split('e')
        s = rf'{m}\times10^{{{int(e)}}}'
    else:
        s = f'{p:#.2g}'  # two significant figures, keeping trailing zeros (0.30, not 0.3)
    return rf'$\mathbf{{{s}}}$' if bold else f'${s}$'


def comparison_rows(rows, nd):
    out = []
    for fam in FAMILIES:
        r = next(r for r in rows if r['readout'] == fam)
        lo, hi = r['ci']
        out.append(f"\\quad {LABELS[fam]} & {num(r['mean_a'], nd)} & {num(r['mean_b'], nd)} & "
                   f"{signed(r['diff'], nd)} [{signed(lo, nd)}, {signed(hi, nd)}] & {pfmt(r['p'])} & "
                   f"{pfmt(r['p_holm'], r['p_holm'] < .05)} \\\\")
    return out


def comparison_table(groups, caption, label):
    lines = [r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{6pt}',
             r'\renewcommand{\arraystretch}{1.1}', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\vspace{0.5em}', r'\begin{tabular}{@{}lrrlrr@{}}', r'\toprule',
             r'Readout & LFADS & MtM & MtM $-$ LFADS [95\% CI] & $p$ & Holm $p$ \\', r'\midrule']
    for i, (title, rows, nd) in enumerate(groups):
        if i:
            lines.append(r'\addlinespace[3pt]')
        lines.append(rf'\multicolumn{{6}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
        lines += comparison_rows(rows, nd)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def interaction_table(primary, ext, caption, label):
    lines = [r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{6pt}',
             r'\renewcommand{\arraystretch}{1.1}', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
             r'\vspace{0.5em}', r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule',
             r'& \multicolumn{3}{c}{MtM $-$ LFADS gap} & & & \\', r'\cmidrule(lr){2-4}',
             r'Outcome & Linear & MLP & Transformer & Friedman $\chi^2$ & $p$ & Holm $p$ \\', r'\midrule']
    for i, (title, rows) in enumerate([('Primary outcomes (Holm over 3)', primary),
                                       ('Extension tasks (Holm over 4)', ext)]):
        if i:
            lines.append(r'\addlinespace[3pt]')
        lines.append(rf'\multicolumn{{7}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
        for name, r, nd in rows:
            g = r['gap_by_readout']
            lines.append(f"\\quad {name} & " + ' & '.join(signed(g[f], nd) for f in FAMILIES) +
                         f" & ${r['friedman_chi2']:.2f}$ & {pfmt(r['p'])} & {pfmt(r['p_holm'], r['p_holm'] < .05)} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(lines)


def heldout_counts():
    units, mapped = set(), set()
    for f in (ROOT / 'results' / 'lfads').glob('*/seed0/neurons.csv'):
        for r in csv.DictReader(f.open()):
            units.add((r['eid'], r['unit_id']))
            if r['region'] not in ('root', 'void'):
                mapped.add((r['eid'], r['unit_id']))
    return len(units), len(mapped)


def main(paper_dir):
    v2 = ROOT / 'v2'
    S = json.loads((v2 / 'final_stats.json').read_text())
    E = json.loads((v2 / 'final_stats_ext.json').read_text())
    out = Path(paper_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    n_units, n_mapped = heldout_counts()
    n, mice = S['panel_B_wheel'][0]['n_sessions'], S['panel_B_wheel'][0]['n_mice']

    A = S['panel_A_cosmooth']; B = S['panel_B_wheel']; D = E['panel_D']
    (out / 'table_figure1_primary.tex').write_text(comparison_table(
        [('Co-smoothing: held-out co-bps (higher is better)', [r for r in A if r['metric'] == 'co-bps'], 3),
         ('Co-smoothing: MAE in window-mean rate, Hz (lower is better)', [r for r in A if r['metric'] == 'MAE (Hz)'], 2),
         (r'Wheel speed $R^2$', B, 3)],
        rf'Figure~1 primary outcomes: frozen LFADS vs.\ MtM representations under three readouts on the test split '
        rf'({n} sessions, {mice} mice; seeds averaged within session). CI: 95\% hierarchical bootstrap over mice, '
        r'then sessions (10{,}000 replicates). $p$: two-sided Wilcoxon signed-rank test over sessions. Holm '
        r'correction within co-smoothing (6 tests) and within wheel (3 tests). Bold: Holm $p<0.05$.',
        'tab:figure1-primary'))
    task_names = [('whisker', 'Whisker motion energy'), ('rpaw', 'Right paw speed'), ('lpaw', 'Left paw speed'),
                  ('licks', 'Licking rate')]
    (out / 'table_figure1_ext.tex').write_text(comparison_table(
        [(rf'{name} $R^2$', [r for r in D if r['task'] == t], 3) for t, name in task_names],
        r'Figure~1 extension tasks (BrainWideBench TS1 behaviors on the same sessions, frozen and locked '
        r'separately before testing): test $R^2$ for LFADS vs.\ MtM under three readouts. Statistics as in '
        r'Table~\ref{tab:figure1-primary}; Holm correction across all 12 tests. Bold: Holm $p<0.05$.',
        'tab:figure1-ext'))
    C = {r['metric']: r for r in S['panel_C_interaction']}; Et = {r['task']: r for r in E['panel_E']}
    (out / 'table_figure1_interaction.tex').write_text(interaction_table(
        [('Co-smoothing co-bps', C['co-bps'], 3), ('Co-smoothing MAE (Hz)', C['MAE (Hz)'], 2),
         (r'Wheel speed $R^2$', C['wheel R²'], 3)],
        [(f'{name} $R^2$', Et[t], 3) for t, name in task_names],
        r'Model $\times$ readout interaction. For each session, the MtM $-$ LFADS gap is computed under each '
        r'readout; the Friedman test asks whether the gap differs across the three readouts. Gaps are '
        r'equal-session means.', 'tab:figure1-interaction'))

    shutil.copy2(ROOT / 'figures' / 'figure1_v2.pdf', out / 'figure1_v2.pdf')
    inputs = ['v2/final_stats.json', 'v2/final_stats_ext.json', 'v2/lockfile.json', 'v2/lockfile_ext.json',
              'v2/test_metrics.csv', 'v2/test_metrics_ext.csv', 'figures/figure1_v2.pdf',
              'code/plot_figure1_v2.py', 'code/paper_assets_v2.py']
    (out / 'SOURCES.json').write_text(json.dumps(
        {'source': 'nlb2/experiments/figure1_lfads_vs_mtm', 'sha256': {p: sha(ROOT / p) for p in inputs}}, indent=1) + '\n')
    print('PAPER ASSETS WRITTEN', out, sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--paper-dir', default=str(ROOT / 'figures' / 'paper'))
    main(p.parse_args().paper_dir)
