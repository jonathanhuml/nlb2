"""Check evaluate_v2.py end to end on the VALIDATION split (never test).

For finished fits, re-predicts the val split through the evaluation code path and checks the
result reproduces the validation metric recorded at fit time. Prints the max discrepancy.
Usage: python code/dryrun_val_v2.py [max_fits]
"""
import json
import sys

import numpy as np

import protocol_v2 as P
from evaluate_v2 import evaluate

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
entries = []
for f in sorted(P.FITS.glob('*/*/*/seed*/*.json')):
    r = json.loads(f.read_text())
    st = f.with_name(f"{r['name']}_state.pt")
    if st.exists():
        entries.append({'task': r['task'], 'feature': r['feature'], 'eid': r['eid'], 'seed': r['seed'],
                        'family': r['family'], 'name': r['name'], 'state': str(st.relative_to(P.ROOT)),
                        'recorded': r.get('val_r2', r.get('val_co_bps'))})
rng = np.random.default_rng(0)
entries = [entries[i] for i in sorted(rng.choice(len(entries), min(limit, len(entries)), replace=False))]
rows = evaluate(entries, split='val', native=False)
worst = 0.
for e, row in zip(sorted(entries, key=lambda e: (e['eid'], e['feature'], e['seed'])), rows):
    got = float(row['test_r2'] if e['task'] == 'wheel' else row['test_co_bps'])
    worst = max(worst, abs(got - e['recorded']))
print(f'dry run on val: {len(rows)} readouts re-scored through evaluate(); max |evaluate - recorded| = {worst:.2e}')
nat = evaluate(entries[:1], split='val', native=True)[1:]
print('native rows (val):', len(nat), 'example', {k: nat[0][k] for k in ['feature', 'test_co_bps', 'test_mae_hz']})
