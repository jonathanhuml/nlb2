"""Stage 1: fit a representation on neural targets only, then freeze and cache it.

Methods
  lfads       NLB2 LFADS (unchanged nlb2 code), same hyperparameters as the original Figure 1 run.
  mtm         Released BrainWideBench MtM, shared backbone frozen, new-session stitchers learned.
  mtm_random  Control: identical to mtm but the backbone keeps its random initialization.

Everything needed to re-analyse later is saved under results/<method>/<eid>/seed<k>/:
  trial_lr*.pt            selected-epoch checkpoints per learning-rate candidate
  history_lr*.json        per-epoch train loss / validation held-out NLL
  selection.json          chosen learning rate (seed 0 selects; seeds 1, 2 reuse it)
  latents.npz             raw (unnormalized) latents for train/val/test, shape (windows, 50, D)
  rates.npz               predicted rates (Hz) for ALL units, val and test, plus observed counts
  neurons.csv             per held-out unit co-smoothing metrics on test
  complete.json           summary and provenance
"""
import argparse
import copy
import csv
import gc
import json
import subprocess
import time
import numpy as np
import torch

from common import DEVICE
from figure1_data import ROOT, DT, NT, prepare, batch

EPOCHS = {'lfads': 400, 'mtm': 120, 'mtm_random': 120}
BATCH = {'lfads': 16, 'mtm': 8, 'mtm_random': 8}
WEIGHT_DECAY = {'lfads': 0., 'mtm': .01, 'mtm_random': .01}
LRS = [1e-3]  # 3e-4 dropped 2026-11-20: won 3/19 early selections by <=0.84% val NLL; see learning-curve check


def git_rev(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def build(method, data, epochs, batch_size):
    if method == 'lfads':
        from lfads_model import make_lfads
        return make_lfads(data, epochs, batch_size)
    from mtm_model import MtMNeural, load_checkpoint
    return MtMNeural(data, load_checkpoint(), random_backbone=(method == 'mtm_random'))


def run_forward(model, method, b):
    if method == 'lfads':
        out = model(b['x'])
        return out.rates, out.latents, out
    rates, z = model(b)
    return rates, z, None


def loss_fn(model, method, b, rates, out):
    if method == 'lfads':
        return model.loss({'spikes': b['x'], 'reconstruction_spikes': b['counts']}, out).total
    means = (rates * DT).clamp_min(1e-8)
    return (means - b['counts'] * means.log()).mean()


def chunks(n, size):
    return [np.arange(i, min(i + size, n)) for i in range(0, n, size)]


def val_heldout_nll(model, method, data, batch_size):
    model.eval(); total = 0.; n = 0
    with torch.no_grad():
        for ids in chunks(len(data['splits']['val']['counts']), batch_size):
            b = batch(data, 'val', ids, DEVICE)
            rates, z, out = run_forward(model, method, b)
            mean = (rates[:, :, data['heldout']] * DT).clamp_min(1e-8)
            loss = mean - b['counts'][:, :, data['heldout']] * mean.log()
            total += loss.sum().item(); n += loss.numel()
    return total / n


def fit(method, data, seed, lr, epochs, batch_size, outdir):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build(method, data, epochs, batch_size).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                            weight_decay=WEIGHT_DECAY[method])
    rng = np.random.default_rng(seed)
    best = np.inf; best_state = None; stale = 0; history = []; start = time.monotonic()
    stop_after = max(12, int(.1 * epochs)); minimum = max(10, int(.35 * epochs))
    selected_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train(); total = 0.; nb = 0
        order = rng.permutation(len(data['splits']['train']['counts']))
        for first in range(0, len(order), batch_size):
            b = batch(data, 'train', order[first:first + batch_size], DEVICE)
            opt.zero_grad(set_to_none=True)
            rates, z, out = run_forward(model, method, b)
            loss = loss_fn(model, method, b, rates, out)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            if not torch.isfinite(norm):
                raise RuntimeError('Non-finite gradient')
            opt.step(); total += loss.item(); nb += 1
        score = val_heldout_nll(model, method, data, batch_size)
        history.append({'epoch': epoch, 'train_loss': total / nb, 'val_heldout_nll': score,
                        'seconds': time.monotonic() - start})
        if score < best - 1e-6:
            best = score; stale = 0; selected_epoch = epoch
            best_state = {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print('NEURAL', method, data['eid'][:8], 'seed', seed, 'lr', lr, json.dumps(history[-1]), flush=True)
            (outdir / f'history_lr{lr:g}.json').write_text(json.dumps(history, indent=1))
        if epoch >= minimum and stale >= stop_after:
            break
    (outdir / f'history_lr{lr:g}.json').write_text(json.dumps(history, indent=1))
    ckpt = outdir / f'trial_lr{lr:g}.pt'
    torch.save({'model': best_state, 'method': method, 'seed': seed, 'lr': lr,
                'selected_epoch': selected_epoch, 'val_heldout_nll': best, 'max_epochs': epochs}, ckpt)
    return {'lr': lr, 'selected_epoch': selected_epoch, 'trained_epochs': epoch,
            'val_heldout_nll': best, 'seconds': time.monotonic() - start, 'checkpoint': str(ckpt)}


def extract(model, method, data, batch_size):
    latents = {}; rates = {}
    model.eval()
    with torch.no_grad():
        for split in ['train', 'val', 'test']:
            zz = []; rr = []
            for ids in chunks(len(data['splits'][split]['counts']), batch_size):
                b = batch(data, split, ids, DEVICE)
                r, z, _ = run_forward(model, method, b)
                zz.append(z.float().cpu().numpy()); rr.append(r.float().cpu().numpy())
            latents[split] = np.concatenate(zz); rates[split] = np.concatenate(rr)
    return latents, rates


def neuron_metrics(data, rates, split):
    """Identical to the original Figure 1 scoring (first 500 ms window-mean rate MAE, co-bps)."""
    idx = data['heldout']
    observed = data['splits'][split]['counts'][:, :25, idx].astype(np.float64)
    predicted = rates[:, :25, idx].astype(np.float64)
    baseline = data['splits']['train']['counts'].mean((0, 1))[idx] / DT
    observed_window = observed.sum(1) / .5; predicted_window = predicted.mean(1)
    mae = np.abs(predicted_window - observed_window).mean(0)
    baseline_mae = np.abs(baseline - observed_window).mean(0)
    global_bias = np.abs(predicted_window.mean(0) - observed_window.mean(0))
    means = np.maximum(predicted * DT, 1e-8); base = np.maximum(baseline * DT, 1e-8)
    ll = (observed * np.log(means / base) - means + base).sum((0, 1))
    spike_sum = observed.sum((0, 1))
    bps = np.divide(ll, spike_sum * np.log(2), out=np.full(len(idx), np.nan), where=spike_sum > 0)
    # Full-window (all 50 bins) co-bps as well, the standard NLB-style quantity.
    obs_full = data['splits'][split]['counts'][:, :, idx].astype(np.float64)
    m_full = np.maximum(rates[:, :, idx].astype(np.float64) * DT, 1e-8)
    ll_full = (obs_full * np.log(m_full / base) - m_full + base).sum((0, 1))
    s_full = obs_full.sum((0, 1))
    bps_full = np.divide(ll_full, s_full * np.log(2), out=np.full(len(idx), np.nan), where=s_full > 0)
    rows = []
    for j, i in enumerate(idx):
        rows.append({'eid': data['eid'], 'subject': data['subject'], 'unit_id': data['unit_ids'][i],
                     'region': data['regions'][i], 'cosmos': data['cosmos'][i],
                     'x': data['xyz'][i, 0], 'y': data['xyz'][i, 1], 'z': data['xyz'][i, 2],
                     'mean_rate_mae_hz': mae[j], 'baseline_mae_hz': baseline_mae[j],
                     'global_mean_bias_hz': global_bias[j], 'co_bps': bps[j], 'co_bps_full_window': bps_full[j],
                     'observed_mean_hz': observed_window[:, j].mean(),
                     'predicted_mean_hz': predicted_window[:, j].mean()})
    return rows


def run(eid, method, seed):
    data = prepare(eid)
    out = ROOT / 'results' / method / eid / f'seed{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        print('SKIP', out, flush=True); return
    epochs, batch_size = EPOCHS[method], BATCH[method]
    lrs = LRS
    if seed > 0:
        lrs = [json.loads((out.parent / 'seed0' / 'selection.json').read_text())['chosen']['lr']]
    candidates = []
    for lr in lrs:
        ck = out / f'trial_lr{lr:g}.pt'
        if ck.exists():
            c = torch.load(ck, map_location='cpu', weights_only=False)
            candidates.append({k: c[k] for k in ['lr', 'selected_epoch', 'val_heldout_nll']} | {'checkpoint': str(ck)})
        else:
            candidates.append(fit(method, data, seed, lr, epochs, batch_size, out))
        gc.collect()
    chosen = min(candidates, key=lambda c: c['val_heldout_nll'])
    (out / 'selection.json').write_text(json.dumps({'chosen': chosen, 'candidates': candidates}, indent=2))
    model = build(method, data, epochs, batch_size)
    model.load_state_dict(torch.load(chosen['checkpoint'], map_location='cpu', weights_only=False)['model'])
    model = model.to(DEVICE)
    latents, rates = extract(model, method, data, batch_size)
    report = getattr(model, 'load_report', None)
    del model; gc.collect()
    np.savez_compressed(out / 'latents.npz', **latents)
    np.savez_compressed(out / 'rates.npz', val=rates['val'], test=rates['test'],
                        val_counts=data['splits']['val']['counts'], test_counts=data['splits']['test']['counts'],
                        heldout=data['heldout'], heldin=data['heldin'])
    rows = neuron_metrics(data, rates['test'], 'test')
    with (out / 'neurons.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {'eid': eid, 'subject': data['subject'], 'method': method, 'seed': seed, 'device': str(DEVICE),
              'neural_selection': chosen, 'heldout_units': len(rows), 'latent_shape': list(latents['test'].shape[1:]),
              'mean_rate_mae_hz': float(np.mean([r['mean_rate_mae_hz'] for r in rows])),
              'baseline_mae_hz': float(np.mean([r['baseline_mae_hz'] for r in rows])),
              'mean_unit_co_bps': float(np.nanmean([r['co_bps'] for r in rows])),
              'mean_unit_co_bps_full_window': float(np.nanmean([r['co_bps_full_window'] for r in rows])),
              'load_report': report,
              'nlb2_commit': git_rev(ROOT / 'vendor' / 'nlb2'), 'ibl_bwb_commit': git_rev(ROOT / 'vendor' / 'ibl-bwb')}
    (out / 'complete.json').write_text(json.dumps(result, indent=2, default=str))
    print('COMPLETE', json.dumps({k: result[k] for k in ['eid', 'method', 'seed', 'mean_rate_mae_hz',
                                                         'baseline_mae_hz', 'mean_unit_co_bps']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--eid', required=True)
    p.add_argument('--method', choices=list(EPOCHS), required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    run(a.eid, a.method, a.seed)
