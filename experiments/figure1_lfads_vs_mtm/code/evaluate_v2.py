"""The single test evaluation of every locked v2 readout.

Refuses to run if v2/test_metrics.csv already exists, if the protocol, selection or code changed
since v2/lockfile.json was written, or if any locked weight file's SHA-256 differs. Writes
  v2/test_metrics.csv           one row per (task, feature, family, session, seed)
  v2/test_predictions/...npz    test predictions per locked readout
  v2/evaluated.json             timestamp and hashes
Also scores the Stage 1 native co-smoothing readouts (results/<model>/<eid>/seed<k>/rates.npz) with
the same functions, as reference rows with family 'native'.
"""
import csv
import json
import sys
import time

import numpy as np
import torch

import protocol_v2 as P
from common import DEVICE
from figure1_data import DT, prepare
from readouts_v2 import apply_norm, build, forward_all, load_features, pooled_co_bps, r2
from select_lock_v2 import CODE, LOCK, SELECTION

OUT = P.V2 / 'test_metrics.csv'
PRED = P.V2 / 'test_predictions'


def mae_hz(counts, rates_hz):
    """Stage 1 MAE: per held-out unit, mean over windows of |predicted - observed| mean rate in the
    first 500 ms (25 bins), then the mean over units."""
    observed = np.asarray(counts, np.float64)[:, :25].sum(1) / .5
    predicted = np.asarray(rates_hz, np.float64)[:, :25].mean(1)
    return float(np.abs(predicted - observed).mean(0).mean())


def verify(lock):
    if lock['protocol_sha256'] != P.protocol_sha256():
        sys.exit('protocol_v2.py changed after locking')
    if lock['selection_sha256'] != P.file_sha256(SELECTION):
        sys.exit('selection.json changed after locking')
    for c, h in lock['code_sha256'].items():
        if c == 'evaluate_v2.py':
            continue  # this file is hashed into evaluated.json instead
        if P.file_sha256(P.ROOT / 'code' / c) != h:
            sys.exit(f'{c} changed after locking')
    bad = [e['state'] for e in lock['entries'] if P.file_sha256(P.ROOT / e['state']) != e['state_sha256']]
    if bad:
        sys.exit(f'{len(bad)} locked weight files changed, e.g. {bad[:2]}')


def predict(entry, ck, x):
    if entry['family'] == 'linear':
        return x @ ck['state']['coef'].astype(np.float32) + np.asarray(ck['state']['intercept'], np.float32)
    model = build(entry['name'], ck['input_dim'], ck['out_dim'])
    model.load_state_dict(ck['state']); model = model.to(DEVICE)
    out = forward_all(model, torch.from_numpy(x).to(DEVICE)).float().cpu().numpy()
    return out[..., 0] if entry['task'] == 'wheel' else out


def evaluate(entries, split='test', native=True):
    """Score locked readouts on `split`. Only main() calls this with split='test'; the val split is a
    dry run that must reproduce the validation metrics recorded at fit time."""
    rows = []; cache = {}
    tag = '' if split == 'test' else f'{split}_'
    for e in sorted(entries, key=lambda e: (e['eid'], e['feature'], e['seed'])):
        if e['eid'] not in cache:
            cache.clear(); cache[e['eid']] = prepare(e['eid'])
        data = cache[e['eid']]
        ck = torch.load(P.ROOT / e['state'], map_location='cpu', weights_only=False)
        raw = load_features(e['feature'], e['eid'], e['seed'], data, splits=(split,))[split]
        out = predict(e, ck, apply_norm(raw, ck['norm']))
        row = {k: e[k] for k in ['task', 'feature', 'family', 'name', 'eid', 'seed']}
        row['subject'] = data['subject']
        stem = f"{tag}{e['task']}_{e['feature']}_{e['family']}_{e['eid']}_seed{e['seed']}.npz"
        if e['task'] == 'wheel':
            y = data['splits'][split]['behavior']
            pred = out * ck['target_stats']['std'] + ck['target_stats']['mean']
            row.update(test_r2=r2(y, pred), test_co_bps='', test_mae_hz='')
            if split == 'test':
                np.savez_compressed(PRED / stem, pred=pred.astype(np.float32), target=y)
        else:
            y = data['splits'][split]['counts'][:, :, data['heldout']]
            base = ck['target_stats']['train_mean_count']
            rates = np.exp(np.minimum(out, 10.)) / DT
            row.update(test_r2='', test_co_bps=pooled_co_bps(y, out, base), test_mae_hz=mae_hz(y, rates))
            if split == 'test':
                np.savez_compressed(PRED / stem, rates_hz=rates.astype(np.float32))
        rows.append(row)
    if not native:
        return rows
    # Stage 1 native co-smoothing readouts (the models' own output layers), same scoring.
    for feature in ['lfads', 'mtm', 'mtm_random']:
        for eid in sorted({e['eid'] for e in entries}):
            data = prepare(eid)
            base = data['splits']['train']['counts'].mean((0, 1))[data['heldout']]
            for seed in sorted({e['seed'] for e in entries}):
                f = P.ROOT / 'results' / feature / eid / f'seed{seed}' / 'rates.npz'
                with np.load(f) as z:
                    idx = z['heldout']; rates = z[split][:, :, idx]; y = z[f'{split}_counts'][:, :, idx]
                rows.append({'task': 'cosmooth', 'feature': feature, 'family': 'native', 'name': 'native', 'eid': eid,
                             'seed': seed, 'subject': data['subject'], 'test_r2': '',
                             'test_co_bps': pooled_co_bps(y, np.log(np.maximum(rates * DT, 1e-8)), base),
                             'test_mae_hz': mae_hz(y, rates)})
    return rows


def main():
    if OUT.exists():
        sys.exit('test metrics already computed; refusing to recompute')
    lock = json.loads(LOCK.read_text())
    verify(lock)
    PRED.mkdir(parents=True, exist_ok=True)
    rows = evaluate(lock['entries'], 'test')
    with OUT.open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    stamp = {'evaluated': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'rows': len(rows),
             'lockfile_sha256': P.file_sha256(LOCK), 'test_metrics_sha256': P.file_sha256(OUT),
             'evaluate_v2_sha256': P.file_sha256(__file__), 'device': str(DEVICE), 'torch': torch.__version__}
    (P.V2 / 'evaluated.json').write_text(json.dumps(stamp, indent=1))
    print('TEST EVALUATED ONCE', json.dumps(stamp))

if __name__ == '__main__':
    main()
