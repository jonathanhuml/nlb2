"""Cross-fitted architecture selection, the ledger of every variant, and the lockfile.

  python code/select_lock_v2.py select   needs every seed-0 screen fit; writes v2/selection.json
  python code/select_lock_v2.py ledger   writes v2/ledger.csv (every fit and every optimizer run)
  python code/select_lock_v2.py lock     needs every selected fit for seeds 0-2; writes v2/lockfile.json

Selection uses validation metrics only. The lockfile records SHA-256 hashes of the protocol, the
fitting/evaluation/statistics code, and every weight file to be evaluated, so evaluate_v2.py can
prove nothing changed between locking and the single test evaluation.
"""
import collections
import csv
import json
import sys
import time

import numpy as np

import protocol_v2 as P

LOCK = P.V2 / 'lockfile.json'
SELECTION = P.V2 / 'selection.json'
CODE = ['protocol_v2.py', 'readouts_v2.py', 'evaluate_v2.py', 'stats_v2.py', 'figure1_data.py', 'select_lock_v2.py']


def load_fit(task, feature, eid, seed, name):
    f = P.FITS / task / feature / eid / f'seed{seed}' / f'{name}.json'
    return json.loads(f.read_text()) if f.exists() else None


def select():
    metric = P.SELECTION_METRIC
    missing = []
    scores = {}  # task -> feature -> name -> eid -> val metric
    subject = {}
    for t in P.TASKS:
        for f in P.FEATURES:
            for fam in ('mlp', 'transformer'):
                for n in P.names(fam):
                    for e in P.eids():
                        r = load_fit(t, f, e, P.SCREEN_SEED, n)
                        if r is None:
                            missing.append((t, f, e, n)); continue
                        scores.setdefault(t, {}).setdefault(f, {}).setdefault(n, {})[e] = r[metric[t]]
                        subject[e] = r['subject']
    if missing:
        sys.exit(f'{len(missing)} screen fits missing, e.g. {missing[:3]}')
    chosen = {}; overall = {}; lomo = {}
    for t in P.TASKS:
        for f in P.FEATURES:
            for fam in ('mlp', 'transformer'):
                cand = P.names(fam)
                overall.setdefault(t, {}).setdefault(f, {})[fam] = {
                    n: float(np.mean([scores[t][f][n][e] for e in P.eids()])) for n in cand}
                for e in P.eids():
                    others = [o for o in P.eids() if subject[o] != subject[e]]
                    s = {n: float(np.mean([scores[t][f][n][o] for o in others])) for n in cand}
                    best = max(cand, key=lambda n: (s[n], -cand.index(n)))
                    chosen.setdefault(t, {}).setdefault(f, {}).setdefault(fam, {})[e] = best
                    lomo.setdefault(t, {}).setdefault(f, {}).setdefault(fam, {})[e] = s
    out = {'rule': 'leave-one-mouse-out: for a session from mouse m, the architecture with the highest '
                   'equal-session mean seed-0 validation metric over sessions of the other mice; test never read',
           'metric': metric, 'chosen': chosen, 'overall_mean_val': overall, 'lomo_scores': lomo,
           'subjects': subject, 'protocol_sha256': P.protocol_sha256(), 'created': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    SELECTION.write_text(json.dumps(out, indent=1))
    for t in P.TASKS:
        for f in P.FEATURES:
            for fam in ('mlp', 'transformer'):
                c = collections.Counter(chosen[t][f][fam].values())
                ranked = sorted(overall[t][f][fam].items(), key=lambda kv: -kv[1])
                print(f'{t:9s} {f:16s} {fam:12s} chosen {dict(c)}  | all-session val ranking: ' +
                      ', '.join(f'{n.split("_", 1)[1]}={v:.4f}' for n, v in ranked[:4]))


def ledger():
    rows = []
    for f in sorted(P.FITS.glob('*/*/*/seed*/*.json')):
        r = json.loads(f.read_text())
        base = {k: r.get(k) for k in ['task', 'feature', 'eid', 'subject', 'seed', 'name', 'family', 'config']}
        sel_metric = r.get('val_r2', r.get('val_co_bps'))
        if r['name'] == 'linear':
            for s in r['search']:
                rows.append(dict(base, setting=json.dumps({k: v for k, v in s.items() if k in ('alpha', 'l2')}),
                                 val_loss=s.get('val_mse', ''), val_metric=s.get('val_r2', s.get('val_co_bps')),
                                 chosen_setting=int(s.get('alpha', s.get('l2')) == r.get('alpha', r.get('l2'))),
                                 fit_val_metric=sel_metric))
        else:
            for s in r['search']:
                rows.append(dict(base, setting=json.dumps({'lr': s['lr'], 'weight_decay': s['weight_decay'],
                                                           'selected_epoch': s['selected_epoch']}),
                                 val_loss=s['best_val_loss'], val_metric='',
                                 chosen_setting=int(s['lr'] == r['lr'] and s['weight_decay'] == r['weight_decay']),
                                 fit_val_metric=sel_metric))
    with (P.V2 / 'ledger.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    fits = len({(r['task'], r['feature'], r['eid'], r['seed'], r['name']) for r in rows})
    print(f'ledger: {len(rows)} optimizer/penalty runs across {fits} fitted readouts -> v2/ledger.csv')


def lock():
    if LOCK.exists():
        sys.exit('lockfile already exists; refusing to overwrite')
    sel = json.loads(SELECTION.read_text())
    if sel['protocol_sha256'] != P.protocol_sha256():
        sys.exit('protocol changed since selection')
    entries = []; missing = []
    for t in P.TASKS:
        for f in P.FEATURES:
            for e in P.eids():
                for seed in P.SEEDS:
                    for fam in P.FAMILIES:
                        name = 'linear' if fam == 'linear' else sel['chosen'][t][f][fam][e]
                        d = P.FITS / t / f / e / f'seed{seed}'
                        js, st = d / f'{name}.json', d / f'{name}_state.pt'
                        if not (js.exists() and st.exists()):
                            missing.append((t, f, e, seed, name)); continue
                        entries.append({'task': t, 'feature': f, 'eid': e, 'seed': seed, 'family': fam, 'name': name,
                                        'json': str(js.relative_to(P.ROOT)), 'state': str(st.relative_to(P.ROOT)),
                                        'json_sha256': P.file_sha256(js), 'state_sha256': P.file_sha256(st)})
    if missing:
        sys.exit(f'{len(missing)} selected fits missing, e.g. {missing[:3]}')
    code = {c: P.file_sha256(P.ROOT / 'code' / c) for c in CODE}
    LOCK.write_text(json.dumps({'created': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'protocol_sha256': P.protocol_sha256(),
                                'selection_sha256': P.file_sha256(SELECTION), 'code_sha256': code,
                                'n_entries': len(entries), 'entries': entries}, indent=1))
    print(f'LOCKED {len(entries)} readouts; protocol {P.protocol_sha256()[:12]}; lockfile sha {P.file_sha256(LOCK)[:12]}')


if __name__ == '__main__':
    {'select': select, 'ledger': ledger, 'lock': lock}[sys.argv[1]]()
