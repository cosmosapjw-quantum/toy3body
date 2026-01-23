from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# Matplotlib is an optional dependency for the core pipeline;
# plotting scripts import this module.
import matplotlib
matplotlib.use("Agg", force=True)  # headless
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class FigureConfig:
    fmt: str = "pdf"          # "pdf", "png", "svg"
    dpi: int = 300            # used for raster formats only
    fontsize: float = 10.0
    use_tex: bool = False
    tight: bool = True
    pad_inches: float = 0.02

    # A compact default figure size suitable for single-column papers
    figsize: Tuple[float, float] = (3.4, 2.6)


def set_paper_style(cfg: FigureConfig) -> None:
    """Set Matplotlib rcParams for publication-friendly figures.

    Keep this conservative and dependency-free (no seaborn/scienceplots).
    """
    plt.rcParams.update({
        "figure.figsize": cfg.figsize,
        "figure.dpi": 120,
        "savefig.dpi": cfg.dpi,
        "font.size": cfg.fontsize,
        "axes.labelsize": cfg.fontsize,
        "axes.titlesize": cfg.fontsize,
        "legend.fontsize": max(6.0, cfg.fontsize - 2.0),
        "xtick.labelsize": max(6.0, cfg.fontsize - 2.0),
        "ytick.labelsize": max(6.0, cfg.fontsize - 2.0),
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        "pdf.fonttype": 42,  # editable text in many PDF editors
        "ps.fonttype": 42,
    })
    if cfg.use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def savefig(fig: plt.Figure, path: str, cfg: FigureConfig) -> None:
    """Save figure with publication-friendly defaults.

    Matplotlib's bbox_inches='tight' is often needed to avoid clipped labels/titles.
    """
    kwargs = {}
    if cfg.tight:
        kwargs["bbox_inches"] = "tight"
        kwargs["pad_inches"] = cfg.pad_inches
    # DPI matters for raster formats; for PDF/SVG it is mostly ignored.
    if path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        kwargs["dpi"] = cfg.dpi
    fig.savefig(path, **kwargs)
