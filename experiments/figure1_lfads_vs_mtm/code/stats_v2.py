"""Pre-specified statistics for v2 (see protocol_v2.py docstring). Reads v2/test_metrics.csv only.

Writes v2/final_stats.json, v2/final_table.md, v2/session_values.csv and prints the final table.
"""
import collections
import csv
import json

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

import protocol_v2 as P

OUTCOMES = [('cosmooth', 'test_co_bps', 'co-bps', 'higher'), ('cosmooth', 'test_mae_hz', 'MAE (Hz)', 'lower'),
            ('wheel', 'test_r2', 'wheel R²', 'higher')]
A, B = P.PRIMARY  # difference reported as B - A = MtM - LFADS


def load():
    rows = list(csv.DictReader((P.V2 / 'test_metrics.csv').open()))
    vals = collections.defaultdict(list); subject = {}
    for r in rows:
        subject[r['eid']] = r['subject']
        for task, key, _, _ in OUTCOMES:
            if r['task'] == task and r[key] != '':
                vals[(key, r['feature'], r['family'], r['eid'])].append((int(r['seed']), float(r[key])))
    return vals, subject


def session_table(vals, key, feature, family, eids):
    out = {}
    for e in eids:
        v = vals.get((key, feature, family, e), [])
        if len(v) != len(P.SEEDS):
            raise SystemExit(f'{key} {feature} {family} {e}: {len(v)} seeds (expected {len(P.SEEDS)})')
        out[e] = float(np.mean([x for _, x in v]))
    return out


def hier_boot(values, subjects, reps=P.BOOTSTRAP_REPS, seed=P.BOOTSTRAP_SEED):
    by = collections.defaultdict(list)
    for v, s in zip(values, subjects):
        by[s].append(v)
    groups = [np.asarray(g) for g in by.values()]
    rng = np.random.default_rng(seed); boots = np.empty(reps)
    for i in range(reps):
        picked = [groups[j][rng.integers(0, len(groups[j]), len(groups[j]))] for j in rng.integers(0, len(groups), len(groups))]
        boots[i] = np.concatenate(picked).mean()
    return [float(x) for x in np.quantile(boots, [.025, .975])]


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (m - rank) * ps[i])); adj[i] = running
    return [float(x) for x in adj]


def paired(da, db, eids, subject, direction):
    d = np.array([db[e] - da[e] for e in eids])
    lo, hi = hier_boot(d, [subject[e] for e in eids])
    p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.
    mice = collections.defaultdict(list)
    for e, x in zip(eids, d):
        mice[subject[e]].append(x)
    dm = np.array([np.mean(v) for v in mice.values()])
    p_mouse = float(wilcoxon(dm).pvalue) if np.any(dm != 0) else 1.
    b_better = int((d > 0).sum() if direction == 'higher' else (d < 0).sum())
    return {'mean_a': float(np.mean([da[e] for e in eids])), 'mean_b': float(np.mean([db[e] for e in eids])),
            'diff': float(d.mean()), 'ci': [lo, hi], 'p': p, 'n_sessions': len(eids), 'n_mice': len(mice),
            'mtm_better_sessions': b_better, 'p_mouse_level': p_mouse}


def main():
    vals, subject = load()
    eids = P.eids()
    S = {}  # (key, feature, family) -> {eid: seed-mean}
    for task, key, _, _ in OUTCOMES:
        fams = P.FAMILIES + (['native'] if task == 'cosmooth' else [])
        for f in P.FEATURES:
            for fam in fams:
                if fam == 'native' and f == 'smoothed_spikes':
                    continue
                S[(key, f, fam)] = session_table(vals, key, f, fam, eids)
    with (P.V2 / 'session_values.csv').open('w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['metric', 'feature', 'family', 'eid', 'subject', 'seed_mean'])
        for (key, f, fam), d in sorted(S.items()):
            for e, v in d.items():
                w.writerow([key, f, fam, e, subject[e], v])
    res = {'panel_A_cosmooth': [], 'panel_B_wheel': [], 'panel_C_interaction': [], 'secondary': {}}
    for task, key, label, direction in OUTCOMES:
        panel = 'panel_B_wheel' if task == 'wheel' else 'panel_A_cosmooth'
        for fam in P.FAMILIES:
            r = paired(S[(key, A, fam)], S[(key, B, fam)], eids, subject, direction)
            res[panel].append(dict(r, metric=label, readout=fam))
    for panel in ['panel_A_cosmooth', 'panel_B_wheel']:
        for r, a in zip(res[panel], holm([r['p'] for r in res[panel]])):
            r['p_holm'] = a
    for task, key, label, direction in OUTCOMES:
        D = np.array([[S[(key, B, fam)][e] - S[(key, A, fam)][e] for fam in P.FAMILIES] for e in eids])
        stat, p = friedmanchisquare(*D.T)
        contrasts = {}
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            c = D[:, j] - D[:, i]
            lo, hi = hier_boot(c, [subject[e] for e in eids])
            contrasts[f'{P.FAMILIES[j]} - {P.FAMILIES[i]}'] = {'mean': float(c.mean()), 'ci': [lo, hi],
                                                               'p': float(wilcoxon(c).pvalue)}
        for k, a in zip(contrasts, holm([c['p'] for c in contrasts.values()])):
            contrasts[k]['p_holm_within_outcome'] = a
        res['panel_C_interaction'].append({'metric': label, 'friedman_chi2': float(stat), 'p': float(p),
                                           'gap_by_readout': {fam: float(D[:, i].mean()) for i, fam in enumerate(P.FAMILIES)},
                                           'contrasts_of_gap': contrasts})
    for r, a in zip(res['panel_C_interaction'], holm([r['p'] for r in res['panel_C_interaction']])):
        r['p_holm'] = a
    # Secondary: readout main effect within each representation; controls; native readouts.
    for task, key, label, direction in OUTCOMES:
        for f in P.FEATURES:
            M = np.array([[S[(key, f, fam)][e] for fam in P.FAMILIES] for e in eids])
            stat, p = friedmanchisquare(*M.T)
            res['secondary'].setdefault('readout_main_effect', []).append(
                {'metric': label, 'feature': f, 'means': {fam: float(M[:, i].mean()) for i, fam in enumerate(P.FAMILIES)},
                 'friedman_chi2': float(stat), 'p': float(p)})
        fams = P.FAMILIES + (['native'] if task == 'cosmooth' else [])
        for a_, b_ in [('mtm_random', 'mtm'), ('smoothed_spikes', 'lfads'), ('smoothed_spikes', 'mtm')]:
            for fam in fams:
                if fam == 'native' and 'smoothed_spikes' in (a_, b_):
                    continue
                r = paired(S[(key, a_, fam)], S[(key, b_, fam)], eids, subject, direction)
                res['secondary'].setdefault('controls', []).append(dict(r, metric=label, readout=fam, comparison=f'{b_} - {a_}'))
        if task == 'cosmooth':
            r = paired(S[(key, A, 'native')], S[(key, B, 'native')], eids, subject, direction)
            res['secondary'].setdefault('native_stage1', []).append(dict(r, metric=label, readout='native'))
    res['note'] = ('Session = unit (n=29, 13 mice); seeds averaged within session. Diff = MtM - LFADS. CI = 95% '
                   'hierarchical bootstrap (mice, then sessions). p = two-sided Wilcoxon signed-rank over sessions; '
                   'Holm within panel. Panel C: Friedman test across readouts of the per-session MtM - LFADS gap.')
    (P.V2 / 'final_stats.json').write_text(json.dumps(res, indent=1))
    md = render(res)
    (P.V2 / 'final_table.md').write_text(md)
    print(md)


def fmt(x, nd=3):
    return f'{x:+.{nd}f}' if isinstance(x, float) else str(x)


def pf(p):
    return f'{p:.2g}' if p >= 1e-3 else f'{p:.1e}'


def render(res):
    out = ['| Panel | Metric | Readout | LFADS | MtM | MtM − LFADS [95% CI] | MtM better | Wilcoxon p | Holm p |',
           '|---|---|---|---|---|---|---|---|---|']
    for panel, tag in [('panel_A_cosmooth', 'A'), ('panel_B_wheel', 'B')]:
        for r in res[panel]:
            out.append(f"| {tag} | {r['metric']} | {r['readout']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | "
                       f"{fmt(r['diff'])} [{fmt(r['ci'][0])}, {fmt(r['ci'][1])}] | {r['mtm_better_sessions']}/{r['n_sessions']} | "
                       f"{pf(r['p'])} | {pf(r['p_holm'])} |")
    out += ['', '| Panel C: interaction | gap (MtM − LFADS) linear / MLP / transformer | Friedman χ² | p | Holm p |',
            '|---|---|---|---|---|']
    for r in res['panel_C_interaction']:
        g = r['gap_by_readout']
        out.append(f"| {r['metric']} | {fmt(g['linear'])} / {fmt(g['mlp'])} / {fmt(g['transformer'])} | "
                   f"{r['friedman_chi2']:.2f} | {pf(r['p'])} | {pf(r['p_holm'])} |")
    out += ['', 'Contrasts of the gap (Wilcoxon, Holm within metric):']
    for r in res['panel_C_interaction']:
        for k, c in r['contrasts_of_gap'].items():
            out.append(f"- {r['metric']}: gap({k}) = {fmt(c['mean'])} [{fmt(c['ci'][0])}, {fmt(c['ci'][1])}], "
                       f"p = {pf(c['p'])}, Holm p = {pf(c['p_holm_within_outcome'])}")
    out += ['', res['note']]
    return '\n'.join(out)


if __name__ == '__main__':
    main()
