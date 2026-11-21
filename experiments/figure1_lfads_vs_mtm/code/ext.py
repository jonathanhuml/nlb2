"""Extension pipeline for protocol_ext.py (extra TS1 behaviors). Reuses the v2 code unchanged.

  python code/ext.py fit --task whisker --feature mtm --eid E --seed 0 --names linear,...   (train/val only)
  python code/ext.py orchestrate screen|seeds
  python code/ext.py select | ledger | lock
  python code/ext.py evaluate        single test evaluation -> v2/test_metrics_ext.csv
  python code/ext.py stats           -> v2/final_table_ext.md, v2/final_stats_ext.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
import torch

import protocol_ext as X
import protocol_v2 as P
import readouts_v2 as R
from figure1_data import DT, NT, ROOT, prepare

SPLITS_FIT = ('train', 'val')


def behavior(eid, task, data, splits):
    """Per-bin target at the same bin centres as wheel, plus a per-window validity mask."""
    grp, key = X.TASKS[task]
    with h5py.File(ROOT / 'data' / f'{eid}.h5', 'r') as f:
        g = f[grp]; v = g[key][:].astype(np.float64)
        t = float(g['domain/start'][0]) + np.arange(len(v)) / float(g.attrs['sampling_rate'])
    out = {}
    for k in splits:
        centers = data['splits'][k]['starts'][:, None] + (np.arange(NT) + .5) * DT
        y = np.interp(centers, t, v)
        valid = (centers[:, 0] >= t[0]) & (centers[:, -1] <= t[-1]) & np.isfinite(y).all(1)
        out[k] = (y.astype(np.float32), valid)
    return out


def fit(task, feature, eid, seed, names):
    out = X.FITS / task / feature / eid / f'seed{seed}'; out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out / f'{n}.json').exists()]
    if not todo:
        return
    data = prepare(eid)
    tb = behavior(eid, task, data, SPLITS_FIT)
    raw = R.load_features(feature, eid, seed, data)
    raw = {k: raw[k][tb[k][1]] for k in SPLITS_FIT}
    yr = {k: tb[k][0][tb[k][1]] for k in SPLITS_FIT}
    norm = R.norm_stats(raw['train'])
    x = {k: R.apply_norm(v, norm) for k, v in raw.items()}
    stats = R.target_stats('wheel', yr['train'])
    y = {k: (v - stats['mean']) / stats['std'] for k, v in yr.items()}
    for name in todo:
        t0 = time.monotonic()
        # 'wheel' selects the regression path (MSE loss, R^2) inside readouts_v2.
        info, state = R.fit_ridge(x, y) if name == 'linear' else R.fit_net(name, 'wheel', x, y, stats, seed)
        info.update(task=task, feature=feature, eid=eid, subject=data['subject'], seed=seed, name=name,
                    family=P.family_of(name), config=name.split('_', 1)[1] if '_' in name else 'linear',
                    input_dim=int(x['train'].shape[-1]), train_windows=int(len(x['train'])), val_windows=int(len(x['val'])),
                    dropped_windows={k: int((~tb[k][1]).sum()) for k in SPLITS_FIT}, seconds=time.monotonic() - t0,
                    device=str(R.DEVICE), torch=torch.__version__, normalization_floor=norm['floor'],
                    protocol_sha256=X.protocol_sha256(), code_sha256=P.file_sha256(__file__),
                    readouts_v2_sha256=P.file_sha256(R.__file__))
        torch.save({'name': name, 'task': task, 'state': state, 'norm': norm, 'target_stats': stats,
                    'input_dim': int(x['train'].shape[-1]), 'out_dim': 1}, out / f'{name}_state.pt')
        (out / f'{name}.json').write_text(json.dumps(info, default=float))
        print('FIT', task, feature, eid[:8], 'seed', seed, name, 'val_r2', round(info['val_r2'], 4),
              f"{info['seconds']:.1f}s", flush=True)


# ---------------------------------------------------------------- orchestration
def orchestrate(phase):
    workers = int(os.environ.get('FIG1_WORKERS', 8))
    env = dict(os.environ, PYTHONPATH=f"{ROOT/'vendor/ibl-bwb/src'}:{ROOT/'vendor/nlb2/src'}:{ROOT/'code'}",
               FIG1_DEVICE=os.environ.get('FIG1_DEVICE', 'cuda'), OMP_NUM_THREADS='2', PYTHONWARNINGS='ignore')
    logs = P.V2 / 'logs_ext'; logs.mkdir(parents=True, exist_ok=True)
    if phase == 'screen':
        tr = P.names('transformer')
        jobs = [(t, f, e, P.SCREEN_SEED, g) for g in [tr[:4], tr[4:], P.names('mlp') + ['linear']]
                for t in X.TASKS for f in X.FEATURES for e in P.eids()]
    else:
        sel = json.loads((P.V2 / 'selection_ext.json').read_text())['chosen']
        jobs = [(t, f, e, s, ['linear'] + [sel[t][f][fam][e] for fam in ('mlp', 'transformer')])
                for s in P.SEEDS if s != P.SCREEN_SEED for t in X.TASKS for f in X.FEATURES for e in P.eids()]

    def run(j):
        t, f, e, s, names = j
        while not (ROOT / 'results' / f / e / f'seed{s}' / 'complete.json').exists():
            time.sleep(60)
        todo = [n for n in names if not (X.FITS / t / f / e / f'seed{s}' / f'{n}.json').exists()]
        if not todo:
            return 0
        cmd = [str(ROOT / '.venv/bin/python'), str(ROOT / 'code/ext.py'), 'fit', '--task', t, '--feature', f, '--eid', e,
               '--seed', str(s), '--names', ','.join(todo)]
        with (logs / f'{t}_{f}_{e}_seed{s}.log').open('a') as fh:
            rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        print(('OK  ' if rc == 0 else 'FAIL'), t, f, e, 'seed', s, flush=True)
        return rc
    with ThreadPoolExecutor(workers) as pool:
        rcs = list(pool.map(run, jobs))
    print(f'PHASE ext-{phase} FINISHED: {len(jobs)} jobs, {sum(r != 0 for r in rcs)} failed', flush=True)


# ---------------------------------------------------------------- select / ledger / lock (reuse v2 code)
def _patch_selection_module():
    import select_lock_v2 as SL
    P.TASKS = list(X.TASKS); P.FEATURES = list(X.FEATURES); P.FITS = X.FITS
    P.SELECTION_METRIC = {t: 'val_r2' for t in X.TASKS}
    P.protocol_sha256 = X.protocol_sha256
    SL.SELECTION = P.V2 / 'selection_ext.json'; SL.LOCK = P.V2 / 'lockfile_ext.json'
    SL.CODE = SL.CODE + ['protocol_ext.py', 'ext.py']
    return SL


def evaluate():
    SL = _patch_selection_module()
    import evaluate_v2 as EV
    out_csv = P.V2 / 'test_metrics_ext.csv'
    if out_csv.exists():
        sys.exit('extension test metrics already computed; refusing to recompute')
    lock = json.loads(SL.LOCK.read_text())
    EV.SELECTION = SL.SELECTION
    EV.verify(lock)
    pred_dir = P.V2 / 'test_predictions_ext'; pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []; cache = {}
    for e in sorted(lock['entries'], key=lambda e: (e['eid'], e['task'], e['feature'], e['seed'])):
        if e['eid'] not in cache:
            cache.clear(); cache[e['eid']] = (prepare(e['eid']), {})
        data, tcache = cache[e['eid']]
        if e['task'] not in tcache:
            tcache[e['task']] = behavior(e['eid'], e['task'], data, ('test',))['test']
        y, valid = tcache[e['task']]
        ck = torch.load(ROOT / e['state'], map_location='cpu', weights_only=False)
        raw = R.load_features(e['feature'], e['eid'], e['seed'], data, splits=('test',))['test'][valid]
        out = EV.predict(dict(e, task='wheel'), ck, R.apply_norm(raw, ck['norm']))
        pred = out * ck['target_stats']['std'] + ck['target_stats']['mean']
        rows.append({k: e[k] for k in ['task', 'feature', 'family', 'name', 'eid', 'seed']} |
                    {'subject': data['subject'], 'test_r2': R.r2(y[valid], pred), 'test_windows': int(valid.sum())})
        np.savez_compressed(pred_dir / f"{e['task']}_{e['feature']}_{e['family']}_{e['eid']}_seed{e['seed']}.npz",
                            pred=pred.astype(np.float32), target=y[valid])
    with out_csv.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stamp = {'evaluated': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': len(rows),
             'lockfile_sha256': P.file_sha256(SL.LOCK), 'test_metrics_sha256': P.file_sha256(out_csv)}
    (P.V2 / 'evaluated_ext.json').write_text(json.dumps(stamp, indent=1))
    print('EXT TEST EVALUATED ONCE', json.dumps(stamp))


def stats():
    import stats_v2 as S
    rows = list(csv.DictReader((P.V2 / 'test_metrics_ext.csv').open()))
    vals = {}; subject = {}
    for r in rows:
        subject[r['eid']] = r['subject']
        vals.setdefault(('test_r2', r['task'] + '|' + r['feature'], r['family'], r['eid']), []).append((int(r['seed']), float(r['test_r2'])))
    eids = P.eids(); A, B = P.PRIMARY
    T = {(t, f, fam): S.session_table(vals, 'test_r2', f'{t}|{f}', fam, eids)
         for t in X.TASKS for f in X.FEATURES for fam in P.FAMILIES}
    res = {'panel_D': [], 'panel_E': [], 'secondary': []}
    for t in X.TASKS:
        for fam in P.FAMILIES:
            res['panel_D'].append(dict(S.paired(T[(t, A, fam)], T[(t, B, fam)], eids, subject, 'higher'), task=t, readout=fam))
    for r, a in zip(res['panel_D'], S.holm([r['p'] for r in res['panel_D']])):
        r['p_holm'] = a
    from scipy.stats import friedmanchisquare
    for t in X.TASKS:
        D = np.array([[T[(t, B, fam)][e] - T[(t, A, fam)][e] for fam in P.FAMILIES] for e in eids])
        stat, p = friedmanchisquare(*D.T)
        res['panel_E'].append({'task': t, 'gap_by_readout': {fam: float(D[:, i].mean()) for i, fam in enumerate(P.FAMILIES)},
                               'friedman_chi2': float(stat), 'p': float(p)})
        for f in X.FEATURES:
            M = np.array([[T[(t, f, fam)][e] for fam in P.FAMILIES] for e in eids])
            s2, p2 = friedmanchisquare(*M.T)
            res['secondary'].append({'task': t, 'feature': f, 'means': {fam: float(M[:, i].mean()) for i, fam in enumerate(P.FAMILIES)},
                                     'readout_main_effect_friedman_p': float(p2)})
    for r, a in zip(res['panel_E'], S.holm([r['p'] for r in res['panel_E']])):
        r['p_holm'] = a
    (P.V2 / 'final_stats_ext.json').write_text(json.dumps(res, indent=1))
    md = ['| Panel | Task | Readout | LFADS R² | MtM R² | MtM − LFADS [95% CI] | MtM better | p | Holm p (12) |', '|---|---|---|---|---|---|---|---|---|']
    for r in res['panel_D']:
        md.append(f"| D | {r['task']} | {r['readout']} | {r['mean_a']:.3f} | {r['mean_b']:.3f} | {S.fmt(r['diff'])} "
                  f"[{S.fmt(r['ci'][0])}, {S.fmt(r['ci'][1])}] | {r['mtm_better_sessions']}/{r['n_sessions']} | {S.pf(r['p'])} | {S.pf(r['p_holm'])} |")
    md += ['', '| Panel E: interaction | gap linear / MLP / transformer | Friedman χ² | p | Holm p (4) |', '|---|---|---|---|---|']
    for r in res['panel_E']:
        g = r['gap_by_readout']
        md.append(f"| {r['task']} | {S.fmt(g['linear'])} / {S.fmt(g['mlp'])} / {S.fmt(g['transformer'])} | {r['friedman_chi2']:.2f} | {S.pf(r['p'])} | {S.pf(r['p_holm'])} |")
    text = '\n'.join(md)
    (P.V2 / 'final_table_ext.md').write_text(text)
    print(text)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'fit':
        p = argparse.ArgumentParser()
        p.add_argument('--task', choices=list(X.TASKS), required=True); p.add_argument('--feature', choices=X.FEATURES, required=True)
        p.add_argument('--eid', required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--names', required=True)
        a = p.parse_args(sys.argv[2:])
        fit(a.task, a.feature, a.eid, a.seed, a.names.split(','))
    elif cmd == 'orchestrate':
        orchestrate(sys.argv[2])
    elif cmd in ('select', 'ledger', 'lock'):
        getattr(_patch_selection_module(), cmd)()
    elif cmd == 'evaluate':
        evaluate()
    elif cmd == 'stats':
        stats()
