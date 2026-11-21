"""v2 readout fitting on frozen features. Uses the train and val splits only.

Test features and test targets are never read here; evaluate_v2.py computes test metrics once,
after lock_v2.py has frozen every selection. Outputs under v2/fits/<task>/<feature>/<eid>/seed<k>/:
  <name>.json      validation metrics, chosen optimizer settings / penalty, full search record
  <name>_state.pt  weights, feature normalization, target statistics (everything needed to predict)
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn
from torch.nn import functional as F

import protocol_v2 as P
from common import DEVICE
from figure1_data import DT, ROOT, prepare

FIT_SPLITS = ('train', 'val')
LOGR_MAX = 10.


# ----------------------------------------------------------------------------- data
def load_features(feature, eid, seed, data, splits=FIT_SPLITS):
    if feature == 'smoothed_spikes':
        return {k: gaussian_filter1d(data['splits'][k]['counts'][:, :, data['heldin']].astype(np.float32),
                                     P.SMOOTH_SIGMA_S / DT, axis=1) / DT for k in splits}
    with np.load(ROOT / 'results' / feature / eid / f'seed{seed}' / 'latents.npz') as z:
        return {k: z[k].astype(np.float32) for k in splits}


def norm_stats(train):
    """Per-channel z-scoring with a floored scale.

    v1 only guarded exactly-zero variance; a unit firing once in training had a tiny nonzero std
    and exploded at test time (smoothed_spikes MLP R^2 of -35). Flooring at 10% of the median
    positive std removes that failure and is a no-op for dense latents.
    """
    flat = train.reshape(-1, train.shape[-1]).astype(np.float64)
    mean = flat.mean(0); std = flat.std(0)
    positive = std[std > 1e-8]
    floor = .1 * float(np.median(positive)) if len(positive) else 1.
    scale = np.maximum(std, floor)
    return {'mean': mean.astype(np.float32), 'scale': scale.astype(np.float32), 'floor': floor,
            'floored_channels': int((std < floor).sum())}


def apply_norm(x, norm):
    return ((x - norm['mean']) / norm['scale']).astype(np.float32)


def targets(task, data, splits=FIT_SPLITS):
    if task == 'wheel':
        y = {k: data['splits'][k]['behavior'].astype(np.float32) for k in splits}
        return y
    idx = data['heldout']
    return {k: data['splits'][k]['counts'][:, :, idx].astype(np.float32) for k in splits}


def target_stats(task, y_train):
    if task == 'wheel':
        return {'mean': float(y_train.mean()), 'std': max(float(y_train.std()), 1e-6)}
    return {'train_mean_count': y_train.mean((0, 1)).astype(np.float64)}


def r2(y, p):
    y = np.asarray(y, np.float64).ravel(); p = np.asarray(p, np.float64).ravel()
    return float(1. - np.square(y - p).sum() / np.square(y - y.mean()).sum())


def pooled_co_bps(counts, log_rates, base_count):
    """Sum over windows, bins and held-out units of the Poisson log-likelihood gain over the
    training mean rate, divided by total spikes * ln 2. Same quantity as aggregate.pooled_bps."""
    y = np.asarray(counts, np.float64)
    lam = np.exp(np.minimum(np.asarray(log_rates, np.float64), LOGR_MAX))
    lam = np.maximum(lam, 1e-8)
    base = np.maximum(np.asarray(base_count, np.float64), 1e-8)
    ll = (y * np.log(lam / base) - lam + base).sum()
    return float(ll / (y.sum() * np.log(2)))


# ----------------------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, dim, out, depth, width, dropout=P.DROPOUT):
        super().__init__()
        layers, d = [], dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout)]; d = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out)

    def forward(self, x):
        return self.head(self.body(x))


class Transformer(nn.Module):
    """Bidirectional encoder across the 50 bins, then a shared per-bin linear head (v1 PerBinTransformer)."""

    def __init__(self, dim, out, depth, width, heads, tokens=50, dropout=P.DROPOUT):
        super().__init__()
        self.input = nn.Linear(dim, width) if dim != width else nn.Identity()
        self.time = nn.Parameter(torch.randn(1, tokens, width) * .02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=dropout,
                                                                batch_first=True, norm_first=True, activation='gelu')
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, out)

    def forward(self, x):
        h = self.input(x) + self.time
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


def build(name, dim, out):
    family, config = name.split('_', 1)
    if family == 'mlp':
        return MLP(dim, out, **P.MLP_CONFIGS[config])
    return Transformer(dim, out, **P.TRANSFORMER_CONFIGS[config])


def init_head(model, task, stats):
    with torch.no_grad():
        if task == 'cosmooth':
            model.head.bias.copy_(torch.from_numpy(np.log(np.maximum(stats['train_mean_count'], 1e-4))).float())
        else:
            model.head.bias.zero_()


# ----------------------------------------------------------------------------- losses
def task_loss(task, out, y):
    if task == 'wheel':
        return F.mse_loss(out.squeeze(-1), y)
    logr = out.clamp(max=LOGR_MAX)
    return (logr.exp() - y * logr).mean()


def forward_all(model, x, bs=128):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(xx) for xx in x.split(bs)])


# ----------------------------------------------------------------------------- fitting
def fit_ridge(x, y):
    D = x['train'].shape[-1]
    X = x['train'].reshape(-1, D).astype(np.float64); Y = y['train'].reshape(-1).astype(np.float64)
    xm = X.mean(0); ym = Y.mean(); Xc = X - xm; Yc = Y - ym
    gram = Xc.T @ Xc / len(Xc); rhs = Xc.T @ Yc / len(Xc)
    Xv = x['val'].reshape(-1, D).astype(np.float64)
    best = None; search = []
    for a in P.RIDGE_ALPHA:
        w = np.linalg.solve(gram + a * np.eye(D), rhs)
        pv = (Xv - xm) @ w + ym
        rec = {'alpha': a, 'val_mse': float(np.mean((pv - y['val'].reshape(-1)) ** 2)), 'val_r2': r2(y['val'], pv)}
        search.append(rec)
        if best is None or rec['val_mse'] < best[0]['val_mse']:
            best = (rec, w)
    rec, w = best
    state = {'coef': w, 'intercept': float(ym - xm @ w)}
    info = {'alpha': rec['alpha'], 'val_r2': rec['val_r2'], 'val_mse': rec['val_mse'], 'parameters': D + 1, 'search': search}
    return info, state


def fit_glm(x, y, stats):
    D = x['train'].shape[-1]; N = y['train'].shape[-1]
    dev = DEVICE; dt = torch.float64 if dev.type == 'cpu' else torch.float32  # consumer GPUs: fp64 is ~64x slower
    X = torch.from_numpy(x['train'].reshape(-1, D)).to(dev, dt); Y = torch.from_numpy(y['train'].reshape(-1, N)).to(dev, dt)
    Xv = torch.from_numpy(x['val'].reshape(-1, D)).to(dev, dt)
    base = stats['train_mean_count']
    W = torch.zeros(D, N, device=dev, dtype=dt, requires_grad=True)
    b = torch.tensor(np.log(np.maximum(base, 1e-4)), device=dev, dtype=dt).requires_grad_(True)
    best = None; search = []
    for alpha in sorted(P.GLM_L2, reverse=True):  # warm start from the strongest penalty
        opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn='strong_wolfe',
                                tolerance_grad=1e-7, tolerance_change=1e-10)

        def closure():
            opt.zero_grad()
            logr = (X @ W + b).clamp(max=LOGR_MAX)
            loss = (logr.exp() - Y * logr).mean() + alpha * W.square().sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            lv = (Xv @ W + b).cpu().numpy().reshape(*x['val'].shape[:2], N)
        rec = {'l2': alpha, 'val_co_bps': pooled_co_bps(y['val'], lv, base)}
        search.append(rec)
        if not math.isfinite(rec['val_co_bps']):
            continue
        if best is None or rec['val_co_bps'] > best[0]['val_co_bps']:
            best = (rec, W.detach().cpu().numpy().copy(), b.detach().cpu().numpy().copy())
    rec, w, bb = best
    return {'l2': rec['l2'], 'val_co_bps': rec['val_co_bps'], 'parameters': D * N + N, 'search': search}, \
        {'coef': w, 'intercept': bb}


def fit_net(name, task, x, y, stats, seed):
    xt = {k: torch.from_numpy(v).to(DEVICE) for k, v in x.items()}
    yt = {k: torch.from_numpy(v).to(DEVICE) for k, v in y.items()}
    dim = x['train'].shape[-1]; out = 1 if task == 'wheel' else y['train'].shape[-1]
    runs = []; best_all = None
    for lr, wd in P.OPT_SEARCH:
        torch.manual_seed(seed)
        model = build(name, dim, out).to(DEVICE); init_head(model, task, stats)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        g = torch.Generator().manual_seed(seed)
        best = float('inf'); state = None; stale = 0; chosen = 0; hist = []
        for epoch in range(1, P.MAX_EPOCHS + 1):
            model.train(); n = 0
            # Same permutation and update order as before; host-device syncs reduced from two per
            # minibatch to one per epoch (performance only, 2026-11-20 22:15; see LEDGER.md).
            perm = torch.randperm(len(yt['train']), generator=g).to(DEVICE)
            tot = torch.zeros((), device=DEVICE)
            for idx in perm.split(P.BATCH):
                opt.zero_grad(set_to_none=True)
                loss = task_loss(task, model(xt['train'][idx]), yt['train'][idx])
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
                tot += loss.detach() * len(idx); n += len(idx)
            tot = float(tot)
            if not math.isfinite(tot):
                raise RuntimeError(f'Non-finite readout loss ({name}, lr={lr}, wd={wd})')
            val = float(task_loss(task, forward_all(model, xt['val']), yt['val']))
            hist.append([epoch, round(tot / n, 6), round(val, 6)])
            if val < best - 1e-7:
                best = val; stale = 0; chosen = epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if epoch >= P.MIN_EPOCHS and stale >= P.PATIENCE:
                break
        runs.append({'lr': lr, 'weight_decay': wd, 'best_val_loss': best, 'selected_epoch': chosen,
                     'trained_epochs': epoch, 'history': hist})
        if best_all is None or best < best_all[0]:
            best_all = (best, state, lr, wd, chosen)
    best, state, lr, wd, chosen = best_all
    model = build(name, dim, out); model.load_state_dict(state); model = model.to(DEVICE)
    pv = forward_all(model, xt['val']).float().cpu().numpy()
    info = {'lr': lr, 'weight_decay': wd, 'selected_epoch': chosen, 'val_loss': best,
            'parameters': sum(p.numel() for p in model.parameters()), 'search': runs}
    if task == 'wheel':
        info['val_r2'] = r2(y['val'], pv[..., 0])
    else:
        info['val_co_bps'] = pooled_co_bps(y['val'], pv, stats['train_mean_count'])
    return info, state


def out_dir(task, feature, eid, seed):
    return P.FITS / task / feature / eid / f'seed{seed}'


def run(task, feature, eid, seed, names):
    out = out_dir(task, feature, eid, seed); out.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out / f'{n}.json').exists()]
    if not todo:
        return
    data = prepare(eid)
    raw = load_features(feature, eid, seed, data)
    norm = norm_stats(raw['train'])
    x = {k: apply_norm(v, norm) for k, v in raw.items()}
    y_raw = targets(task, data)
    stats = target_stats(task, y_raw['train'])
    y = {k: (v - stats['mean']) / stats['std'] for k, v in y_raw.items()} if task == 'wheel' else y_raw
    for name in todo:
        t0 = time.monotonic()
        if name == 'linear':
            info, state = fit_ridge(x, y) if task == 'wheel' else fit_glm(x, y, stats)
        else:
            info, state = fit_net(name, task, x, y, stats, seed)
        info.update(task=task, feature=feature, eid=eid, subject=data['subject'], seed=seed, name=name,
                    family=P.family_of(name), config=name.split('_', 1)[1] if '_' in name else 'linear',
                    input_dim=int(x['train'].shape[-1]), heldout_units=int(len(data['heldout'])),
                    train_windows=int(len(x['train'])), val_windows=int(len(x['val'])), seconds=time.monotonic() - t0,
                    device=str(DEVICE), torch=torch.__version__, normalization_floor=norm['floor'],
                    floored_channels=norm['floored_channels'], protocol_sha256=P.protocol_sha256(),
                    code_sha256=P.file_sha256(__file__))
        torch.save({'name': name, 'task': task, 'state': state, 'norm': norm, 'target_stats': stats,
                    'input_dim': int(x['train'].shape[-1]), 'out_dim': 1 if task == 'wheel' else int(len(data['heldout']))},
                   out / f'{name}_state.pt')
        (out / f'{name}.json').write_text(json.dumps(info, default=float))
        metric = info.get('val_r2', info.get('val_co_bps'))
        print('FIT', task, feature, eid[:8], 'seed', seed, name, P.SELECTION_METRIC[task], round(metric, 4),
              f"{info['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=P.TASKS, required=True)
    p.add_argument('--feature', choices=P.FEATURES, required=True)
    p.add_argument('--eid', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--names', required=True, help='comma list, e.g. linear,mlp_d2_w128,transformer_d2_w256_h4')
    a = p.parse_args()
    run(a.task, a.feature, a.eid, a.seed, a.names.split(','))
