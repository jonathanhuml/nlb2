from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nlb2.plotting import (
    model_color,
    model_label,
    model_marker,
    plot_context,
    plot_style,
    save_figure,
)


def test_plot_style_returns_uniform_rcparams():
    style = plot_style(nrows=2, ncols=2, rel_width=0.8, height_scale=1.4)

    assert style["text.usetex"] is False
    assert "figure.figsize" in style
    assert "axes.prop_cycle" in style
    assert float(style["figure.figsize"][1]) > float(style["figure.figsize"][0]) * 0.5
    assert model_color("gpfa") != model_color("kalman")
    assert model_marker("gpfa") == "^"
    assert model_label("bgpfa") == "bGPFA"
    assert model_label("ilqr_vae") == "iLQR-VAE"


def test_plot_context_can_write_png(tmp_path: Path):
    path = tmp_path / "figure.png"

    with plot_context(nrows=1, ncols=1):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1], color=model_color("gpfa"))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        save_figure(fig, path)
        plt.close(fig)

    assert path.exists()
    assert path.stat().st_size > 0


def test_experiment_plots_preserve_display_backend_and_open_figures(tmp_path: Path):
    from nlb2.experiment import _write_history_plots
    from nlb2.training import EpochReport
    from nlb2.types import StepResult

    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use("svg", force=True)
        user_figure, user_axis = plt.subplots()
        user_axis.plot([0, 1], [1, 0])
        open_figures = plt.get_fignums()
        history = [
            EpochReport(epoch=0, train=StepResult(loss=2.0), valid=StepResult(loss=2.5), seconds=0.1),
            EpochReport(epoch=1, train=StepResult(loss=1.0), valid=StepResult(loss=1.5), seconds=0.1),
        ]

        paths = _write_history_plots(tmp_path, history)

        assert matplotlib.get_backend().lower() == "svg"
        assert plt.get_fignums() == open_figures
        assert len(paths) == 2
        for path in paths.values():
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close("all")
        matplotlib.use(original_backend, force=True)
