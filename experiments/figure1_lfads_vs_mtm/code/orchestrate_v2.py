"""Run v2 readout fits on HAL (resumable; finished fits are skipped).

  python code/orchestrate_v2.py screen   seed 0: linear + every MLP and transformer architecture,
                                         all tasks x features x 29 sessions (validation only)
  python code/orchestrate_v2.py seeds    seeds 1-2: linear + the cross-fitted architecture chosen
                                         for each session in v2/selection.json (validation only)
Environment: FIG1_WORKERS (default 10), FIG1_DEVICE (default cuda).
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
import protocol_v2 as P  # noqa: E402

WORKERS = int(os.environ.get('FIG1_WORKERS', 10))
ENV = dict(os.environ, PYTHONPATH=f"{ROOT/'vendor/ibl-bwb/src'}:{ROOT/'vendor/nlb2/src'}:{ROOT/'code'}",
           FIG1_DEVICE=os.environ.get('FIG1_DEVICE', 'cuda'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
           PYTHONWARNINGS='ignore')
LOGS = P.V2 / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)


def stage1_done(feature, eid, seed):
    if feature == 'smoothed_spikes':
        return True
    return (ROOT / 'results' / feature / eid / f'seed{seed}' / 'complete.json').exists()


def fit_done(task, feature, eid, seed, name):
    return (P.FITS / task / feature / eid / f'seed{seed}' / f'{name}.json').exists()


def running_elsewhere(task, feature, eid, seed):
    """A fit launched by an earlier orchestrator keeps running after that orchestrator is stopped."""
    pat = f'readouts_v2.py --task {task} --feature {feature} --eid {eid} --seed {seed} '
    return bool(subprocess.run(['pgrep', '-f', pat], capture_output=True, text=True).stdout.strip())


def job(task, feature, eid, seed, names):
    while not stage1_done(feature, eid, seed):
        time.sleep(60)
    while running_elsewhere(task, feature, eid, seed):
        time.sleep(20)
    todo = [n for n in names if not fit_done(task, feature, eid, seed, n)]
    if not todo:
        return 0
    log = LOGS / f'{task}_{feature}_{eid}_seed{seed}.log'
    cmd = [str(ROOT / '.venv/bin/python'), str(ROOT / 'code/readouts_v2.py'), '--task', task, '--feature', feature,
           '--eid', eid, '--seed', str(seed), '--names', ','.join(todo)]
    with log.open('a') as fh:
        rc = subprocess.call(cmd, env=ENV, stdout=fh, stderr=subprocess.STDOUT)
    print(('OK  ' if rc == 0 else 'FAIL'), task, feature, eid, 'seed', seed, ','.join(todo), flush=True)
    return rc


def screen_jobs():
    tr = P.names('transformer')
    groups = [tr[:4], tr[4:], P.names('mlp') + ['linear']]  # heaviest first
    return [(t, f, e, P.SCREEN_SEED, g) for g in groups for t in P.TASKS for f in P.FEATURES for e in P.eids()]


def seed_jobs():
    sel = json.loads((P.V2 / 'selection.json').read_text())['chosen']
    jobs = []
    for seed in [s for s in P.SEEDS if s != P.SCREEN_SEED]:
        for t in P.TASKS:
            for f in P.FEATURES:
                for e in P.eids():
                    jobs.append((t, f, e, seed, ['linear'] + [sel[t][f][fam][e] for fam in ('mlp', 'transformer')]))
    return jobs


if __name__ == '__main__':
    phase = sys.argv[1]
    jobs = screen_jobs() if phase == 'screen' else seed_jobs()
    with ThreadPoolExecutor(WORKERS) as pool:
        rcs = list(pool.map(lambda j: job(*j), jobs))
    print(f'PHASE {phase} FINISHED: {len(jobs)} jobs, {sum(r != 0 for r in rcs)} failed', flush=True)
