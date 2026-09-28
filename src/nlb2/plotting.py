"""Shared plotting style for NLB2 benchmark artifacts."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


_MODEL_COLORS = {
    "true": "#111111",
    "target": "#111111",
    "prediction": "#0072B2",
    "pred": "#0072B2",
    "mint": "#CC79A7",
    "ilqr_vae": "#D55E00",
    "gpfa": "#009E73",
    "bgpfa": "#56B4E9",
    "kalman": "#E69F00",
    "langevin_flow": "#332288",
    "cassm": "#0072B2",
    "lfads": "#7F3C8D",
    "ndt": "#666666",
    "stndt": "#A6761D",
}

_MODEL_MARKERS = {
    "mint": "o",
    "ilqr_vae": "s",
    "gpfa": "^",
    "bgpfa": "v",
    "kalman": "D",
    "langevin_flow": "<",
    "cassm": "P",
    "lfads": "X",
    "ndt": "h",
    "stndt": "*",
}

_MODEL_LABELS = {
    "bgpfa": "bGPFA",
    "cassm": "CASSM",
    "gpfa": "GPFA",
    "ilqr_vae": "iLQR-VAE",
    "kalman": "Kalman",
    "langevin_flow": "LangevinFlow",
    "lfads": "LFADS",
    "mint": "MINT",
    "ndt": "NDT",
    "stndt": "STNDT",
    "true": "true",
    "pred": "predicted",
    "prediction": "predicted",
}

_FALLBACK_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F3C8D",
    "#666666",
]
_FALLBACK_MARKERS = ["o", "s", "^", "v", "D", "P", "X", "h"]


def plot_style(
    *,
    nrows: int = 1,
    ncols: int = 1,
    rel_width: float = 1.0,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
) -> dict[str, object]:
    """Return publication-style Matplotlib rcParams.

    TUEplots is used when available. The fallback intentionally mirrors the
    same rcParam contract so plotting code does not need a second path on
    Python versions where TUEplots cannot be installed.
    """

    fallback = _fallback_style(
        nrows=nrows,
        ncols=ncols,
        rel_width=rel_width,
        width_scale=width_scale,
        height_scale=height_scale,
    )
    try:
        from tueplots import axes, bundles, cycler
        from tueplots.constants.color import palettes, rgb
    except ImportError:
        return fallback

    try:
        style: dict[str, object] = {}
        bundle = _select_neurips_bundle(bundles)
        style.update(
            bundle(
                usetex=False,
                rel_width=rel_width,
                nrows=nrows,
                ncols=ncols,
                family="serif",
            )
        )
        style.update(axes.lines())
        style.update(axes.grid())
        style.update(axes.color(base=rgb.tue_dark))
        style.update(cycler.cycler(color=palettes.tue_plot))
        style.update(_common_style_overrides())
        _scale_figsize(style, width_scale=width_scale, height_scale=height_scale)
        return style
    except Exception:
        return fallback


@contextmanager
def plot_context(
    *,
    nrows: int = 1,
    ncols: int = 1,
    rel_width: float = 1.0,
    width_scale: float = 1.0,
    height_scale: float = 1.0,
) -> Iterator[None]:
    """Apply the shared NLB2 plotting style within a Matplotlib context."""

    import matplotlib.pyplot as plt

    with plt.rc_context(
        plot_style(
            nrows=nrows,
            ncols=ncols,
            rel_width=rel_width,
            width_scale=width_scale,
            height_scale=height_scale,
        )
    ):
        yield


def model_color(name: str) -> str:
    """Return a stable color for a model or trace label."""

    key = str(name).lower()
    if key in _MODEL_COLORS:
        return _MODEL_COLORS[key]
    return _FALLBACK_CYCLE[_stable_index(key, len(_FALLBACK_CYCLE))]


def model_marker(name: str) -> str:
    """Return a stable marker for a model label."""

    key = str(name).lower()
    if key in _MODEL_MARKERS:
        return _MODEL_MARKERS[key]
    return _FALLBACK_MARKERS[_stable_index(key, len(_FALLBACK_MARKERS))]


def model_label(name: str) -> str:
    """Return the display spelling for a model label."""

    key = str(name).lower()
    return _MODEL_LABELS.get(key, str(name))


def style_axis(ax, *, which: str = "major") -> None:
    """Apply final per-axis polish that rcParams cannot cover consistently."""

    ax.grid(True, which=which, alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_outside(ax, *, ncols: int = 1):
    """Place a compact legend outside the plotting area."""

    return ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        frameon=False,
        fontsize=6.5,
        handlelength=1.15,
        labelspacing=0.35,
        ncol=ncols,
    )


def save_figure(fig, path, *, dpi: int = 300) -> None:
    """Save a figure with consistent export settings."""

    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def _fallback_style(
    *,
    nrows: int,
    ncols: int,
    rel_width: float,
    width_scale: float,
    height_scale: float,
) -> dict[str, object]:
    from cycler import cycler

    width = 5.5 * float(rel_width) * float(width_scale)
    panel_height = 1.75
    height = max(2.25, panel_height * max(1, int(nrows))) * float(height_scale)
    if ncols > 1:
        height = max(2.25, height * 1.15)
    return {
        **_common_style_overrides(),
        "figure.figsize": (width, height),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "axes.prop_cycle": cycler(color=_FALLBACK_CYCLE),
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }


def _common_style_overrides() -> dict[str, object]:
    return {
        "text.usetex": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.pad_inches": 0.03,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.0,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    }


def _stable_index(value: str, length: int) -> int:
    return sum(ord(char) for char in value) % length


def _scale_figsize(style: dict[str, object], *, width_scale: float, height_scale: float) -> None:
    size = style.get("figure.figsize")
    if size is None:
        return
    width, height = size
    style["figure.figsize"] = (
        float(width) * float(width_scale),
        float(height) * float(height_scale),
    )


def _select_neurips_bundle(bundles):
    for name in ("neurips2024", "neurips2023", "neurips2022", "neurips2021"):
        bundle = getattr(bundles, name, None)
        if bundle is not None:
            return bundle
    raise AttributeError("No supported TUEplots NeurIPS bundle is available.")
