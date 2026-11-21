"""Verify the committed Figure 1 record without data, fits or a GPU.

  python code/verify.py

1. Lock integrity. For the primary lock (v2/lockfile.json) and the extension lock (v2/lockfile_ext.json):
   every locked code file, the protocol, and the selection still hash to the values recorded at lock time,
   and the single test evaluation (v2/evaluated*.json) points at this lockfile and this test_metrics file.
2. Statistics. Re-runs stats_v2.py and `ext.py stats` from the committed test metrics in a scratch copy,
   then compares every number with the committed final_stats*.json / session_values.csv and every line
   of final_table*.md.
3. Ledgers. Every fitted readout variant is listed in v2/ledger*.csv.gz.
4. Pretraining overlap (rule 1), only if vendor/ibl-bwb is present (`sh fetch_inputs.sh`): none of the
   29 evaluation sessions is in MtM's pretraining set.
Exit code 0 only if every check passes.
"""
import csv
import gzip
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'v2'
FAILED = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(ok, label, detail=''):
    print(f"{'PASS' if ok else 'FAIL'}  {label}{(' - ' + detail) if detail else ''}", flush=True)
    if not ok:
        FAILED.append(label)


def protocol_hash(module):
    sys.path.insert(0, str(ROOT / 'code'))
    try:
        return __import__(module).protocol_sha256()
    finally:
        sys.path.pop(0)


def verify_lock(name, lock_file, evaluated_file, selection_file, protocol_module, metrics_file, n_expected):
    lock = json.loads((V2 / lock_file).read_text())
    evaluated = json.loads((V2 / evaluated_file).read_text())
    bad = [c for c, h in lock['code_sha256'].items() if sha(ROOT / 'code' / c) != h]
    check(not bad, f'{name}: {len(lock["code_sha256"])} locked code files unchanged', ', '.join(bad))
    check(protocol_hash(protocol_module) == lock['protocol_sha256'], f'{name}: protocol hash',
          lock['protocol_sha256'][:12])
    check(sha(V2 / selection_file) == lock['selection_sha256'], f'{name}: selection hash')
    check(lock['n_entries'] == n_expected == len(lock['entries']), f'{name}: {n_expected} locked readouts')
    check(evaluated['lockfile_sha256'] == sha(V2 / lock_file), f'{name}: test evaluation used this lockfile')
    check(evaluated['test_metrics_sha256'] == sha(V2 / metrics_file), f'{name}: test metrics unchanged since evaluation')
    if 'evaluate_v2_sha256' in evaluated:
        check(evaluated['evaluate_v2_sha256'] == sha(ROOT / 'code' / 'evaluate_v2.py'), f'{name}: evaluate_v2.py unchanged')


def numbers_equal(a, b, path='', tol=1e-9):
    """Recursive comparison; returns a list of mismatching paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return [f'{path}: keys differ']
        return [m for k in a for m in numbers_equal(a[k], b[k], f'{path}.{k}', tol)]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f'{path}: lengths differ']
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in numbers_equal(x, y, f'{path}[{i}]', tol)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if (math.isnan(a) and math.isnan(b)) or abs(a - b) <= tol * max(1., abs(a), abs(b)):
            return []
        return [f'{path}: {a} != {b}']
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def csv_equal(a, b):
    ra, rb = list(csv.DictReader(io.StringIO(a))), list(csv.DictReader(io.StringIO(b)))
    if len(ra) != len(rb):
        return [f'{len(ra)} rows != {len(rb)} rows']
    out = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        for k in x:
            try:
                out += numbers_equal(float(x[k]), float(y[k]), f'row {i} {k}')
            except ValueError:
                out += [] if x[k] == y[k] else [f'row {i} {k}: {x[k]} != {y[k]}']
    return out


def verify_stats():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(ROOT / 'code', tmp / 'code', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'eval_recording_ids.txt', tmp)
        (tmp / 'v2').mkdir()
        for f in ['test_metrics.csv', 'test_metrics_ext.csv']:
            shutil.copy2(V2 / f, tmp / 'v2' / f)
        env = dict(os.environ, PYTHONPATH=str(tmp / 'code'), PYTHONWARNINGS='ignore', FIG1_DEVICE='cpu')
        for cmd in (['code/stats_v2.py'], ['code/ext.py', 'stats']):
            r = subprocess.run([sys.executable, *cmd], cwd=tmp, env=env, capture_output=True, text=True)
            check(r.returncode == 0, f're-ran {" ".join(cmd)}', r.stderr.strip().splitlines()[-1] if r.returncode else '')
            if r.returncode:
                return
        for f in ['final_stats.json', 'final_stats_ext.json']:
            diff = numbers_equal(json.loads((tmp / 'v2' / f).read_text()), json.loads((V2 / f).read_text()))
            check(not diff, f'recomputed {f} matches the record', '; '.join(diff[:3]))
        for f in ['final_table.md', 'final_table_ext.md']:
            same = (tmp / 'v2' / f).read_text() == (V2 / f).read_text()
            check(same, f'recomputed {f} matches the record')
        diff = csv_equal((tmp / 'v2' / 'session_values.csv').read_text(), (V2 / 'session_values.csv').read_text())
        check(not diff, 'recomputed session_values.csv matches the record', '; '.join(diff[:3]))


def verify_ledgers():
    for f, fits_expected in [('ledger.csv.gz', 4872), ('ledger_ext.csv.gz', 7308)]:
        with gzip.open(V2 / f, 'rt') as fh:
            rows = list(csv.DictReader(fh))
        fits = {(r['task'], r['feature'], r['eid'], r['seed'], r['name']) for r in rows}
        check(len(fits) == fits_expected, f'{f}: {len(fits)} fitted readouts, {len(rows)} optimizer/penalty runs')


def verify_pretraining_overlap():
    pre = ROOT / 'vendor/ibl-bwb/src/core/data/pretrain_recording_ids.txt'
    if not pre.exists():
        print('SKIP  pretraining overlap (run `sh fetch_inputs.sh` to fetch vendor/ibl-bwb)')
        return
    ev = set((ROOT / 'eval_recording_ids.txt').read_text().split())
    pt = set(pre.read_text().split())
    check(len(ev) == 29 and not ev & pt, f'0 of {len(ev)} evaluation sessions in the {len(pt)}-session MtM pretraining set')


if __name__ == '__main__':
    verify_lock('primary', 'lockfile.json', 'evaluated.json', 'selection.json', 'protocol_v2', 'test_metrics.csv', 2088)
    verify_lock('extension', 'lockfile_ext.json', 'evaluated_ext.json', 'selection_ext.json', 'protocol_ext',
                'test_metrics_ext.csv', 3132)
    verify_ledgers()
    verify_stats()
    verify_pretraining_overlap()
    print(f'\n{"ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} CHECK(S) FAILED"}')
    sys.exit(1 if FAILED else 0)
