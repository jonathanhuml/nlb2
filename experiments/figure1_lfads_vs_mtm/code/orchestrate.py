"""Run Stage 1 for every session/method/seed as data arrives. Resumable: completed runs are skipped.

Seed 0 (two learning-rate candidates) must finish before seeds 1-2 of the same session/method.
"""
import os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['lfads', 'mtm', 'mtm_random']
WORKERS = int(os.environ.get('FIG1_WORKERS', 6))
SIZES = dict(line.split() for line in (ROOT / 'data_sizes.txt').read_text().splitlines())
ENV = dict(os.environ, PYTHONPATH=f"{ROOT/'vendor/ibl-bwb/src'}:{ROOT/'vendor/nlb2/src'}:{ROOT/'code'}",
           FIG1_DEVICE='cpu', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONWARNINGS='ignore')
(ROOT / 'logs').mkdir(exist_ok=True)


def ready(eid):
    f = ROOT / 'data' / f'{eid}.h5'
    return f.exists() and f.stat().st_size == int(SIZES[eid])


def done(eid, method, seed):
    return (ROOT / 'results' / method / eid / f'seed{seed}' / 'complete.json').exists()


def running_elsewhere(eid, method, seed):
    # A fit started by an earlier orchestrator keeps running after that orchestrator exits.
    out = subprocess.run(['pgrep', '-f', f'stage1_represent.* {eid} {method} {seed}$'], capture_output=True, text=True)
    return bool(out.stdout.strip())


def job(eid, method, seed):
    while running_elsewhere(eid, method, seed):
        time.sleep(30)
    if done(eid, method, seed):
        return 0
    log = ROOT / 'logs' / f'{method}_{eid}_seed{seed}.log'
    cmd = [str(ROOT / '.venv/bin/python'), '-c',
           'import torch;torch.set_num_threads(2);import sys;sys.argv=sys.argv[:1]+sys.argv[1:];'
           'import stage1_represent as s;s.run(sys.argv[1],sys.argv[2],int(sys.argv[3]))', eid, method, str(seed)]
    # Per-method device override (HAL: FIG1_DEVICE_MTM=cuda etc.); default stays cpu as in seed 0.
    env = dict(ENV, FIG1_DEVICE=os.environ.get(f'FIG1_DEVICE_{method.upper()}', os.environ.get('FIG1_DEVICE', 'cpu')))
    with log.open('a') as fh:
        rc = subprocess.call(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(('OK  ' if rc == 0 else 'FAIL'), method, eid, 'seed', seed, flush=True)
    return rc


def wait_job(eid, method, seed):
    while not ready(eid):
        time.sleep(30)
    if seed > 0 and not done(eid, method, 0):
        return
    job(eid, method, seed)


eids = (ROOT / 'eval_recording_ids.txt').read_text().split()
# Order by data availability so finished downloads are used first.
eids.sort(key=lambda e: (not ready(e), e))
# Phase 1: seed 0 everywhere (full-cohort result first). Phase 2: seeds 1, 2.
for seeds in [(0,), (1, 2)]:
    with ThreadPoolExecutor(WORKERS) as pool:
        for seed in seeds:
            for eid in eids:
                for method in METHODS:
                    pool.submit(wait_job, eid, method, seed)
print('ALL SUBMITTED JOBS FINISHED', flush=True)
