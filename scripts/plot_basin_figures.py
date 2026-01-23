#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
import json
import numpy as np

from toy3body_topo.chaos_metrics import basin_entropy_boxed
from toy3body_topo.plotting import FigureConfig, set_paper_style, savefig, ensure_dir

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


INT_TO_LABEL = {
    0: "escape",
    1: "exchange",
    2: "merger_cut",
    3: "timeout",
    4: "unknown",
}


def _boundary_mask(grid: np.ndarray) -> np.ndarray:
    """Return boolean mask of boundary pixels where neighbor labels differ."""
    g = np.asarray(grid, dtype=int)
    b = np.zeros_like(g, dtype=bool)
    # 4-neighborhood
    b[:-1, :] |= (g[:-1, :] != g[1:, :])
    b[1:, :] |= (g[1:, :] != g[:-1, :])
    b[:, :-1] |= (g[:, :-1] != g[:, 1:])
    b[:, 1:] |= (g[:, 1:] != g[:, :-1])
    return b


def plot_basin_map(grid: np.ndarray, b_vals: np.ndarray, phi_vals: np.ndarray, out_path: str, cfg: FigureConfig) -> None:
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111)
    extent = [phi_vals.min(), phi_vals.max(), b_vals.min(), b_vals.max()]
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel(r"phase $\phi$")
    ax.set_ylabel(r"impact parameter $b$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # label ticks if small number of outcomes
    ticks = sorted(set(int(x) for x in np.unique(grid)))
    cb.set_ticks(ticks)
    cb.set_ticklabels([INT_TO_LABEL.get(t, str(t)) for t in ticks])
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_boundary(boundary: np.ndarray, b_vals: np.ndarray, phi_vals: np.ndarray, out_path: str, cfg: FigureConfig) -> None:
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111)
    extent = [phi_vals.min(), phi_vals.max(), b_vals.min(), b_vals.max()]
    im = ax.imshow(boundary.astype(int), origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel(r"phase $\phi$")
    ax.set_ylabel(r"impact parameter $b$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["interior", "boundary"])
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_tend(tend: np.ndarray, b_vals: np.ndarray, phi_vals: np.ndarray, out_path: str, cfg: FigureConfig) -> None:
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(111)
    extent = [phi_vals.min(), phi_vals.max(), b_vals.min(), b_vals.max()]
    im = ax.imshow(tend, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
    ax.set_xlabel(r"phase $\phi$")
    ax.set_ylabel(r"impact parameter $b$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$t_{\rm end}$")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot exit-basin maps + boundary + t_end heatmap + basin entropy.")
    ap.add_argument("--basin_dir", required=True, help="Directory containing basin_grid.npy, b_vals.npy, phi_vals.npy")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fontsize", type=float, default=10.0)
    ap.add_argument("--use_tex", action="store_true")
    ap.add_argument("--box", type=int, default=8, help="Box size for basin entropy")
    ap.add_argument("--n_outcomes", type=int, default=4)
    args = ap.parse_args()

    cfg_fig = FigureConfig(fmt=args.fmt, dpi=args.dpi, fontsize=args.fontsize, use_tex=args.use_tex, figsize=(3.6, 3.0))
    set_paper_style(cfg_fig)

    basin_dir = args.basin_dir
    grid = np.load(os.path.join(basin_dir, "basin_grid.npy"))
    b_vals = np.load(os.path.join(basin_dir, "b_vals.npy"))
    phi_vals = np.load(os.path.join(basin_dir, "phi_vals.npy"))
    tend_path = os.path.join(basin_dir, "basin_tend.npy")
    tend = np.load(tend_path) if os.path.exists(tend_path) else None

    out_dir = args.out_dir or os.path.join(basin_dir, "figs_basin")
    ensure_dir(out_dir)
    ext = args.fmt

    plot_basin_map(grid, b_vals, phi_vals, os.path.join(out_dir, f"basin_map.{ext}"), cfg_fig)

    boundary = _boundary_mask(grid)
    plot_boundary(boundary, b_vals, phi_vals, os.path.join(out_dir, f"basin_boundary.{ext}"), cfg_fig)

    if tend is not None:
        plot_tend(tend, b_vals, phi_vals, os.path.join(out_dir, f"basin_tend.{ext}"), cfg_fig)

    # basin entropy (boxed)
    ent = basin_entropy_boxed(grid, n_outcomes=args.n_outcomes, box=args.box)
    with open(os.path.join(out_dir, "basin_entropy.json"), "w", encoding="utf-8") as f:
        json.dump(ent, f, indent=2)

    print("Saved basin figures to:", out_dir)
    print("Basin entropy:", ent)


if __name__ == "__main__":
    main()
