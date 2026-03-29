from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


EXPLORATORY_DPI = 200
FINAL_DPI = 300

ENSO_ORDER = ["El Nino", "Neutral", "La Nina"]
ENSO_COLORS = {
    "El Nino": "#E74C3C",
    "Neutral": "#95A5A6",
    "La Nina": "#3498DB",
}

DEFAULT_FIGSIZE_SPATIAL = (14, 4.5)
DEFAULT_FIGSIZE_STAT = (12, 6)
DEFAULT_FIGSIZE_SUMMARY = (6, 8)

DEFAULT_SIG_ALPHA = 0.05


def apply_publication_style() -> None:
    """Apply repository-wide plotting defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "axes.unicode_minus": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "savefig.bbox": "tight",
        }
    )


def symmetric_levels(data: np.ndarray, n_levels: int = 21) -> tuple[np.ndarray, TwoSlopeNorm]:
    """Build symmetric levels and a zero-centered norm for signed fields."""
    vmax = float(np.nanmax(np.abs(data)))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    levels = np.linspace(-vmax, vmax, n_levels)
    return levels, TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def add_horizontal_colorbar(
    fig: plt.Figure,
    mappable,
    label: str,
    rect: tuple[float, float, float, float] = (0.12, 0.06, 0.78, 0.03),
):
    """Add a standard horizontal colorbar below the figure."""
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_label(label, fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return cbar


def subsample_sig_mask(sig_mask: np.ndarray, stride: int = 2) -> np.ndarray:
    """Thin dense significance masks so stippling does not dominate the field."""
    if stride <= 1:
        return sig_mask
    out = np.zeros_like(sig_mask, dtype=bool)
    out[::stride, ::stride] = sig_mask[::stride, ::stride]
    return out


def add_stippling(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    sig_mask: np.ndarray | None,
    *,
    color: str = "black",
    size: float = 10,
    alpha: float = 0.75,
    stride: int = 2,
) -> int:
    """Overlay sparse stippling for significant points on a 2D field."""
    if sig_mask is None:
        return 0
    mask = subsample_sig_mask(np.asarray(sig_mask, dtype=bool), stride=stride)
    iy, ix = np.where(mask)
    if len(iy) == 0:
        return 0
    ax.scatter(x[ix], y[iy], c=color, s=size, marker=".", alpha=alpha, linewidths=0)
    return len(iy)


def format_spatial_axes(
    ax: plt.Axes,
    *,
    xlabel: str = "Longitude (degE)",
    ylabel: str = "Latitude (degN)",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Apply the default spatial-axis styling."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(direction="in")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def format_pressure_axis(
    ax: plt.Axes,
    *,
    ylabel: str = "Pressure (hPa)",
    ticks: Iterable[int] = (1000, 850, 700, 500, 300, 200),
) -> None:
    """Format a pressure axis with pressure increasing downward."""
    ax.set_ylabel(ylabel)
    ax.set_yticks(list(ticks))
    ax.invert_yaxis()
    ax.tick_params(direction="in")


def add_group_counts_to_xticklabels(labels: list[str], counts: list[int]) -> list[str]:
    """Append sample sizes to categorical tick labels."""
    return [f"{label}\n(n={count})" for label, count in zip(labels, counts)]


def add_pvalue_brackets(
    ax: plt.Axes,
    comparisons: list[tuple[int, int, float]],
    *,
    fontsize: float = 9,
) -> None:
    """Add stacked p-value brackets above a grouped statistical plot."""
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    start = ymax + 0.02 * yrange
    step = 0.12 * yrange
    cap = 0.03 * yrange

    for idx, (left, right, pval) in enumerate(comparisons):
        y = start + idx * step
        ax.plot([left, left, right, right], [y, y + cap, y + cap, y], color="black", lw=1.0)
        if pval < 0.001:
            text = "p<0.001"
        else:
            text = f"p={pval:.3f}" if pval < 0.01 else f"p={pval:.2f}"
        ax.text((left + right) / 2, y + cap + 0.01 * yrange, text, ha="center", va="bottom", fontsize=fontsize)

    ax.set_ylim(ymin, start + max(len(comparisons), 1) * step + 0.08 * yrange)


def save_figure(
    fig: plt.Figure,
    out_path: str | Path,
    *,
    final: bool = False,
    close: bool = True,
) -> Path:
    """Save a figure with repository-standard DPI and bbox handling."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dpi = FINAL_DPI if final else EXPLORATORY_DPI
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path

