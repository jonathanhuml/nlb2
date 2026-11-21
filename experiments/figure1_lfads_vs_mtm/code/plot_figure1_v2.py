"""Figure 1 (v2): the original Figure 1 design (code/plot_figure1.py), filled with the locked v2 results.

  a  Native co-smoothing error map (Stage 1 output layers), per-unit MAE averaged over 3 seeds, on a
     green (low error) to red (high error) scale; mu_pop = mean over all 757 held-out units +/- s.e.m. across
     units, printed in green for the lower-error model and red for the higher.
  b  Behavioral decoding: wheel-speed R^2 (v2 lock, top) and licking-rate R^2 (extension lock, bottom), LFADS vs
     MtM under linear / MLP / transformer readouts; IBL task cartoon on the right, as in the original figure.
Bars: equal-session mean over 29 sessions (seeds averaged within session). Whiskers: 95% hierarchical
bootstrap (mice, then sessions). Text above each readout: Holm-adjusted Wilcoxon p for MtM - LFADS
(Holm within wheel [3 tests] and within the extension tasks [12 tests]). Sessions, mice, seeds and the
model x readout interaction tests are reported in the appendix, not on the figure.
Reads only stats outputs (v2/final_stats*.json, v2/session_values.csv, v2/test_metrics_ext.csv) plus
Stage 1 neurons.csv; presentation only, no fitting. Bar means are asserted equal to the stats files.
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ['linear', 'mlp', 'transformer']
LABELS = {'linear': 'Linear', 'mlp': 'MLP', 'transformer': 'Transformer'}
COLORS = {'lfads': '#28658b', 'mtm': '#c36b42'}
NAMES = {'lfads': 'LFADS', 'mtm': 'MtM'}
ATLAS_EXTENT = (5.4, -7.8, -7.668, .332)
GOOD, BAD = '#2e7d4f', '#b8372f'
ERROR_CMAP = LinearSegmentedColormap.from_list('error_green_red', [GOOD, '#8cc084', '#efe6b4', '#e39a56', BAD])
W, H = 3.25, 4.0  # inches; original Figure 1 width, footer removed


def diffuse_neurons(rows, background, norm, cmap=ERROR_CMAP, sigma_mm=.55):
    """Interpolate measured neuron errors over nearby tissue for display (as in plot_figure1.py)."""
    height, width = background.shape[:2]
    counts = np.zeros((height, width), dtype=float)
    weighted = np.zeros_like(counts)
    for row in rows:
        px = int(round((ATLAS_EXTENT[0] - float(row['y']) * 1000) / 13.2 * (width - 1)))
        py = int(round((ATLAS_EXTENT[3] - float(row['z']) * 1000) / 8.0 * (height - 1)))
        if 0 <= px < width and 0 <= py < height:
            counts[py, px] += 1
            weighted[py, px] += float(row['mean_rate_mae_hz'])
    sigma = (sigma_mm * height / 8.0, sigma_mm * width / 13.2)
    density = gaussian_filter(counts, sigma=sigma, mode='constant')
    blended = gaussian_filter(weighted, sigma=sigma, mode='constant') / np.maximum(density, 1e-12)
    rgba = cmap(norm(blended))
    texture = background[:, :, 0].astype(float) / 255.0
    rgba[:, :, :3] *= (.2 + .8 * texture[:, :, None])
    tissue = background[:, :, 3].astype(float) / 255.0
    tissue /= max(tissue.max(), 1e-12)
    rgba[:, :, 3] = .96 * density / (density + .008 * density.max()) * tissue
    return rgba


def unit_means(seeds=(0, 1, 2)):
    g = collections.defaultdict(list); meta = {}  # rows carry eid so mu_pop's s.e. can resample by session
    for m in COLORS:
        for f in sorted((ROOT / 'results' / m).glob('*/seed*/neurons.csv')):
            if int(f.parent.name[4:]) not in seeds:
                continue
            for r in csv.DictReader(f.open()):
                key = (m, r['eid'], r['unit_id']); g[key].append(float(r['mean_rate_mae_hz'])); meta[key] = r
    rows = []
    for key, v in g.items():
        r = meta[key]
        rows.append({'method': key[0], 'eid': key[1], 'region': r['region'], 'y': r['y'], 'z': r['z'],
                     'mean_rate_mae_hz': float(np.mean(v)), 'seeds': len(v)})
    return rows


def ext_session_values(v2, task):
    """Seed-mean test R^2 per session for one extension task, in session_values.csv row format."""
    acc = collections.defaultdict(list); subj = {}
    for r in csv.DictReader((v2 / 'test_metrics_ext.csv').open()):
        if r['task'] == task:
            acc[(r['feature'], r['family'], r['eid'])].append(float(r['test_r2'])); subj[r['eid']] = r['subject']
    return [{'metric': f'test_r2_{task}', 'feature': f, 'family': fam, 'eid': e, 'subject': subj[e],
             'seed_mean': float(np.mean(v))} for (f, fam, e), v in acc.items()]


def boot_ci(values, subjects, reps=10_000, seed=20261120):
    by = collections.defaultdict(list)
    for v, s in zip(values, subjects):
        by[s].append(v)
    groups = [np.asarray(x) for x in by.values()]
    rng = np.random.default_rng(seed); out = np.empty(reps)
    for i in range(reps):
        out[i] = np.concatenate([groups[j][rng.integers(0, len(groups[j]), len(groups[j]))]
                                 for j in rng.integers(0, len(groups), len(groups))]).mean()
    return np.quantile(out, [.025, .975])


def ptext(p):
    return 'p < 0.001' if p < 1e-3 else f'p = {p:.3f}' if p < .01 else f'p = {p:.2f}'


def bars(ax, sv, metric_key, pvals, ylabel, check_means, xlabels=True):
    """pvals: {family: Holm p}. check_means: {(feature, family): mean} from the stats file, asserted."""
    vals = collections.defaultdict(dict); subj = {}
    for r in sv:
        if r['metric'] == metric_key and r['feature'] in COLORS and r['family'] in FAMILIES:
            vals[(r['feature'], r['family'])][r['eid']] = float(r['seed_mean']); subj[r['eid']] = r['subject']
    tops = []
    for i, fam in enumerate(FAMILIES):
        for j, m in enumerate(COLORS):
            d = vals[(m, fam)]; eids = sorted(d)
            assert len(eids) == 29, (metric_key, m, fam, len(eids))
            mean = float(np.mean([d[e] for e in eids]))
            assert abs(mean - check_means[(m, fam)]) < 1e-9, (metric_key, m, fam, mean, check_means[(m, fam)])
            lo, hi = boot_ci([d[e] for e in eids], [subj[e] for e in eids])
            x = i + (j - .5) * .36
            ax.bar(x, mean, width=.34, color=COLORS[m], zorder=3)
            ax.errorbar(x, mean, yerr=[[max(0, mean - lo)], [max(0, hi - mean)]], fmt='none', ecolor='#25313b',
                        elinewidth=.5, capsize=1.3, zorder=4)
            tops.append(hi)
    ymax = max(tops)
    for i, fam in enumerate(FAMILIES):
        p = pvals[fam]
        ax.text(i, ymax * 1.04, ptext(p), ha='center', va='bottom', fontsize=4.3,
                color='#25313b' if p < .05 else '#8a969d')
    ax.set_ylim(0, ymax * 1.22)
    ax.set_yticks([0, .2, .4])
    ax.set_xticks(range(3), [LABELS[f] for f in FAMILIES] if xlabels else [], fontsize=5)
    ax.set_xlim(-.55, 2.55)
    ax.set_ylabel(ylabel, fontsize=5.5, labelpad=1.5)
    ax.tick_params(length=1.8, pad=1.2, labelsize=4.8)
    ax.spines[['top', 'right']].set_visible(False); ax.yaxis.grid(True, color='#e4e9ec', lw=.4, zorder=0)
    ax.set_axisbelow(True)


def stats_means(rows):
    out = {}
    for r in rows:
        out[('lfads', r['readout'])] = r['mean_a']; out[('mtm', r['readout'])] = r['mean_b']
    return out


def Y(d):
    """Figure-fraction y for a distance d (inches) below the top edge."""
    return 1 - d / H


def main(v2_dir, output_dir):
    v2 = ROOT / v2_dir
    stats = json.loads((v2 / 'final_stats.json').read_text())
    ext = json.loads((v2 / 'final_stats_ext.json').read_text())
    sv = list(csv.DictReader((v2 / 'session_values.csv').open())) + ext_session_values(v2, 'licks')
    neurons = unit_means()
    seeds_per_unit = sorted({r['seeds'] for r in neurons})
    # mu_pop weights every held-out unit equally (mapped or not); +/- s.e.m. across units.
    pop = {}
    for m in COLORS:
        v = np.array([r['mean_rate_mae_hz'] for r in neurons if r['method'] == m])
        pop[m] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), len(v))
    best = min(COLORS, key=lambda m: pop[m][0])
    mapped = [r for r in neurons if r['region'] not in ('root', 'void')]
    anatomy = np.asarray(Image.open(ROOT / 'assets/allen_sagittal_slice_ml_minus2mm.png').convert('RGBA'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.linewidth': .5, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(W, H))

    # a: native co-smoothing map (positions as in plot_figure1.py)
    vmax = float(5 * np.ceil(np.quantile([r['mean_rate_mae_hz'] for r in mapped], .95) / 5))
    norm = Normalize(0, max(1, vmax))
    fig.text(.02, .985, 'a', weight='bold', fontsize=9, va='top')
    fig.text(.07, .985, 'Co-smoothing', weight='bold', fontsize=8.5, va='top')
    for x, m in [(0., 'lfads'), (.5, 'mtm')]:
        fig.text(x + .25, Y(.28), 'Task-Specific' if m == 'lfads' else 'Foundation Model', ha='center', va='center',
                 fontsize=8, fontstyle='italic')
        fig.text(x + .25, Y(.42), NAMES[m], ha='center', va='center', color=COLORS[m], fontsize=8, weight='bold')
        ax = fig.add_axes([x + .01, Y(1.32), .48, .871 / H])
        ax.imshow(anatomy, extent=ATLAS_EXTENT, origin='upper', interpolation='bilinear')
        ax.imshow(diffuse_neurons([r for r in mapped if r['method'] == m], anatomy, norm), extent=ATLAS_EXTENT,
                  origin='upper', interpolation='bilinear', zorder=3)
        ax.set(xlim=(5.8, -8.2), ylim=(-8., 1.), aspect='equal'); ax.axis('off')
        fig.text(x + .25, Y(1.45), rf"$\mu_{{\mathrm{{pop}}}} = {pop[m][0]:.2f} \pm {pop[m][1]:.2f}$ Hz", ha='center',
                 va='center', fontsize=8, color=GOOD if m == best else BAD)
    cax = fig.add_axes([.25, Y(1.6175), .5, .0595 / H])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=ERROR_CMAP), cax=cax, orientation='horizontal', extend='max')
    cb.ax.tick_params(labelsize=5.5, length=2, pad=1); cb.outline.set_visible(False)
    fig.text(.5, Y(1.82), 'Mean absolute error in window-mean rate (Hz)', ha='center', fontsize=6)

    # b: behavioral decoding: wheel (top) and licks (bottom) stacked on the left, task cartoon on the right
    fig.text(.02, Y(2.02), 'b', weight='bold', fontsize=9, va='top')
    fig.text(.07, Y(2.02), 'Behavioral decoding', weight='bold', fontsize=8.5, va='top')
    for j, m in enumerate(COLORS):
        fig.patches.append(plt.Rectangle((.665 + j * .16, Y(2.125)), .018, .05 / H, color=COLORS[m],
                                         transform=fig.transFigure))
        fig.text(.69 + j * .16, Y(2.10), NAMES[m], fontsize=6, va='center', color=COLORS[m], weight='bold')
    licks = [r for r in ext['panel_D'] if r['task'] == 'licks']
    panels = [
        (2.22, 'Wheel speed', 'test_r2', {r['readout']: r['p_holm'] for r in stats['panel_B_wheel']},
         stats_means(stats['panel_B_wheel']), False),
        (3.08, 'Licking rate', 'test_r2_licks', {r['readout']: r['p_holm'] for r in licks}, stats_means(licks), True),
    ]
    for d0, title, metric_key, pvals, check, xlabels in panels:
        fig.text(.13, Y(d0), title, weight='bold', fontsize=6, va='top')
        ax = fig.add_axes([.13, Y(d0 + .73), .49, .60 / H])
        bars(ax, sv, metric_key, pvals, r'Test $R^2$', check, xlabels)
    inset = fig.add_axes([.655, Y(3.92), .34, 1.70 / H])
    inset.imshow(np.rot90(plt.imread(ROOT / 'assets/ibl_wheel_task.png'), k=-1)); inset.axis('off')
    if seeds_per_unit != [3]:
        fig.text(.5, Y(3.98), f'PREVIEW: map uses {seeds_per_unit} seeds per unit', ha='center', fontsize=5,
                 color='#9d5551')
    out = ROOT / output_dir; out.mkdir(parents=True, exist_ok=True)
    for e in ['png', 'pdf']:
        fig.savefig(out / f'figure1_v2.{e}', dpi=600)
    plt.close(fig)
    print('FIGURE WRITTEN', out / 'figure1_v2.pdf', flush=True)
    for m in COLORS:
        print(f'  mu_pop {m}: {pop[m][0]:.3f} +/- {pop[m][1]:.3f} Hz (s.e.m. over {pop[m][2]} units)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--v2-dir', default='v2'); p.add_argument('--output-dir', default='figures')
    a = p.parse_args(); main(a.v2_dir, a.output_dir)
