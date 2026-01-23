#!/usr/bin/env python
from __future__ import annotations

import os
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from toy3body_topo.chaos_metrics import survival_function, fit_escape_rate
from toy3body_topo.plotting import FigureConfig, set_paper_style, savefig, ensure_dir

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


DEFAULT_FEATURES = [
    "t_end",
    "t_int",
    "r_min",
    "d3_min",
    "n_loops",
    "loop_mean_gap",
    "loop_max_dur",
    "link_ok_frac",
    "link_nz_frac",
    "link_mean_abs",
    "link_raw_mean_abs",
]


def plot_outcome_fractions(df: pd.DataFrame, out_path: str, cfg: FigureConfig) -> None:
    counts = df["label"].value_counts()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(counts.index.astype(str), counts.values.astype(float) / len(df))
    ax.set_ylabel("fraction")
    ax.set_xlabel("outcome")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_survival(df: pd.DataFrame, out_path: str, cfg: FigureConfig, tmin: float = 0.0) -> None:
    t = df["t_end"].to_numpy(dtype=float)
    uniq, S = survival_function(t)
    fit = fit_escape_rate(t, t_min=tmin)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(uniq, S, marker=".", linestyle="None", label="empirical survival")

    if np.isfinite(fit.get("kappa", np.nan)):
        kappa = fit["kappa"]
        # plot fitted exp tail from tmin to max
        tt = np.linspace(max(tmin, uniq.min()), uniq.max(), 200)
        # y = exp(intercept - kappa t)
        intercept = fit.get("intercept", 0.0)
        yy = np.exp(intercept - kappa*tt)
        ax.semilogy(tt, yy, linestyle="-", label=fr"fit: $\kappa\approx{float(kappa):.3g}$")

    ax.set_xlabel("t (dwell time)")
    ax.set_ylabel("S(t) = P(T ≥ t)")
    ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)

    # Save fit JSON for record
    meta = {"fit": fit, "tmin": float(tmin), "n": int(len(df))}
    with open(os.path.splitext(out_path)[0] + "_fit.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def plot_feature_box_by_outcome(df: pd.DataFrame, feature: str, out_path: str, cfg: FigureConfig, yscale: str = "linear") -> None:
    labs = sorted(df["label"].unique().tolist())
    data = [df[df["label"] == lab][feature].to_numpy(dtype=float) for lab in labs]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot(data, labels=labs, showfliers=False)
    ax.set_ylabel(feature)
    ax.set_xlabel("outcome")
    if yscale != "linear":
        ax.set_yscale(yscale)
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, x: str, y: str, out_path: str, cfg: FigureConfig,
                 xscale: str = "linear", yscale: str = "linear") -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lab, g in df.groupby("label"):
        ax.plot(g[x].to_numpy(dtype=float), g[y].to_numpy(dtype=float), marker=".", linestyle="None", label=str(lab))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, features: List[str], out_path: str, cfg: FigureConfig) -> None:
    # numeric only
    X = df[features].apply(pd.to_numeric, errors="coerce")
    C = X.corr().to_numpy()
    fig = plt.figure(figsize=(cfg.figsize[0]*1.2, cfg.figsize[0]*1.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(C, origin="lower", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=90)
    ax.set_yticklabels(features)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out_path, cfg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate ensemble-level analysis figures from runs.csv")
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fontsize", type=float, default=10.0)
    ap.add_argument("--use_tex", action="store_true")
    ap.add_argument("--tmin", type=float, default=0.0, help="Lower cutoff for escape-rate fit")
    args = ap.parse_args()

    cfg_fig = FigureConfig(fmt=args.fmt, dpi=args.dpi, fontsize=args.fontsize, use_tex=args.use_tex, figsize=(3.4, 2.6))
    set_paper_style(cfg_fig)

    df = pd.read_csv(args.runs_csv)
    runs_dir = os.path.dirname(args.runs_csv)
    out_dir = args.out_dir or os.path.join(runs_dir, "figs_ensemble")
    ensure_dir(out_dir)

    ext = args.fmt

    # 1) outcome fractions
    plot_outcome_fractions(df, os.path.join(out_dir, f"outcome_fractions.{ext}"), cfg_fig)

    # 2) survival + escape-rate fit
    plot_survival(df, os.path.join(out_dir, f"survival_escape_rate.{ext}"), cfg_fig, tmin=args.tmin)

    # 3) key topological features by outcome
    for feat, yscale in [
        ("n_loops", "linear"),
        ("link_mean_abs", "linear"),
        ("link_raw_mean_abs", "linear"),
        ("r_min", "log"),
        ("t_end", "log"),
    ]:
        if feat in df.columns:
            plot_feature_box_by_outcome(df, feat, os.path.join(out_dir, f"box_{feat}_by_outcome.{ext}"), cfg_fig, yscale=yscale)

    # 4) scatter: topology vs dwell time
    if ("t_end" in df.columns) and ("link_raw_mean_abs" in df.columns):
        plot_scatter(df, "t_end", "link_raw_mean_abs", os.path.join(out_dir, f"scatter_linkraw_vs_tend.{ext}"), cfg_fig, xscale="log", yscale="linear")

    if ("t_end" in df.columns) and ("link_mean_abs" in df.columns):
        plot_scatter(df, "t_end", "link_mean_abs", os.path.join(out_dir, f"scatter_linkint_vs_tend.{ext}"), cfg_fig, xscale="log", yscale="linear")

    if ("r_min" in df.columns) and ("link_raw_mean_abs" in df.columns):
        plot_scatter(df, "r_min", "link_raw_mean_abs", os.path.join(out_dir, f"scatter_linkraw_vs_rmin.{ext}"), cfg_fig, xscale="log", yscale="linear")

    # 5) correlation heatmap (numeric features)
    feats = [f for f in DEFAULT_FEATURES if f in df.columns]
    if len(feats) >= 4:
        plot_correlation_heatmap(df, feats, os.path.join(out_dir, f"corr_heatmap.{ext}"), cfg_fig)

    print("Saved ensemble figures to:", out_dir)


if __name__ == "__main__":
    main()
