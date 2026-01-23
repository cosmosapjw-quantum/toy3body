#!/usr/bin/env python
from __future__ import annotations

import os
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from toy3body_topo.config import PhysicalParams, SimParams, ICParams, LoopParams, Units
from toy3body_topo.ic import make_initial_state
from toy3body_topo.integrate import integrate_scatter
from toy3body_topo.features import extract_topo_features
from toy3body_topo.shape import positions_from_Y, pairwise_distances, intruder_distance_to_binary_com, interaction_mask, shape_logdist, binary_com
from toy3body_topo.outcomes import classify_final_pair, outcome_label
from toy3body_topo.plotting import FigureConfig, set_paper_style, savefig, ensure_dir

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _float(x) -> float:
    try:
        return float(x)
    except Exception:
        # handle NaN, None, empty
        return float("nan")


def _build_params_from_row(row: pd.Series) -> tuple[PhysicalParams, SimParams, ICParams, LoopParams]:
    # Physical
    phys = PhysicalParams(
        m1=_float(row["phys_m1"]),
        m2=_float(row["phys_m2"]),
        m3=_float(row["phys_m3"]),
        epsilon=_float(row.get("phys_epsilon", 0.0)),
        k_warn=_float(row.get("phys_k_warn", 50.0)),
        k_merge=_float(row.get("phys_k_merge", 10.0)),
    )

    # Sim
    max_step = row.get("sim_max_step", float("inf"))
    try:
        max_step = float(max_step)
    except Exception:
        max_step = float("inf")

    sim = SimParams(
        method=str(row.get("sim_method", "DOP853")),
        rtol=_float(row.get("sim_rtol", 1e-10)),
        atol=_float(row.get("sim_atol", 1e-12)),
        max_step=max_step,
        t_max=_float(row.get("sim_t_max", 3000.0)),
        dt_chunk=_float(row.get("sim_dt_chunk", 50.0)),
        R_int=_float(row.get("sim_R_int", 20.0)),
        R_end=_float(row.get("sim_R_end", 200.0)),
        hier_ratio=_float(row.get("sim_hier_ratio", 5.0)),
        max_store_points=int(float(row.get("sim_max_store_points", 20000))),
        decimate_to=int(float(row.get("sim_decimate_to", 5000))),
    )

    # IC
    ic = ICParams(
        a0=_float(row.get("ic_a0", 1.0)),
        e0=_float(row.get("ic_e0", 0.0)),
        phase=_float(row.get("ic_phase", 0.0)),
        v_inf=_float(row.get("ic_v_inf", 1.0)),
        b=_float(row.get("ic_b", 0.0)),
        R0=_float(row.get("ic_R0", 100.0)),
        theta=_float(row.get("ic_theta", 0.0)),
    )

    # Loop params
    loop = LoopParams(
        eps_close=_float(row.get("loop_eps_close", 1e-2)),
        min_dt=_float(row.get("loop_min_dt", 10.0)),
        k_nn=int(float(row.get("loop_k_nn", 16))),
        max_loops_base=int(float(row.get("loop_max_loops_base", 1))),
        max_loops_long=int(float(row.get("loop_max_loops_long", 8))),
        t_long=_float(row.get("loop_t_long", 300.0)),
        resample_n=int(float(row.get("loop_resample_n", 256))),
        link_round_tol=_float(row.get("loop_link_round_tol", 0.1)),
        link_min_sep=_float(row.get("loop_link_min_sep", 1e-3)),
    )
    return phys, sim, ic, loop


def _energy_series_newton(Y: np.ndarray, units: Units, phys: PhysicalParams) -> np.ndarray:
    """Vectorized Newtonian energy time series for diagnostics."""
    G = units.G
    m = np.array([phys.m1, phys.m2, phys.m3], dtype=float)
    r = Y[:, :6].reshape(-1, 3, 2)
    v = Y[:, 6:].reshape(-1, 3, 2)
    v2 = np.sum(v*v, axis=2)  # (T,3)
    KE = 0.5 * np.sum(v2 * m[None, :], axis=1)  # (T,)

    # pairwise distances (T,3): 12, 23, 31
    d = pairwise_distances(r)
    PE = -(G * m[0]*m[1]/d[:,0] + G * m[1]*m[2]/d[:,1] + G * m[2]*m[0]/d[:,2])
    return KE + PE


def _load_or_rerun(runs_csv: str, seed: int, raw_dir: Optional[str], rerun_if_missing: bool) -> tuple[pd.Series, np.ndarray, np.ndarray, PhysicalParams, SimParams, ICParams, LoopParams]:
    df = pd.read_csv(runs_csv)
    match = df[df["seed"] == seed]
    if len(match) != 1:
        raise ValueError(f"seed {seed} not found uniquely in {runs_csv} (found {len(match)})")
    row = match.iloc[0]

    phys, sim, ic, loop = _build_params_from_row(row)

    # raw path
    if raw_dir is None:
        raw_dir = os.path.join(os.path.dirname(runs_csv), "raw")
    raw_path = os.path.join(raw_dir, f"run_seed{seed}.npz")

    if os.path.exists(raw_path):
        dat = np.load(raw_path)
        T = dat["T"].astype(float)
        Y = dat["Y"].astype(float)
        return row, T, Y, phys, sim, ic, loop

    if not rerun_if_missing:
        raise FileNotFoundError(f"Raw trajectory not found: {raw_path}. Enable --rerun_if_missing or store raw runs.")
    # rerun
    units = Units()
    y0 = make_initial_state(units, phys, ic)
    res = integrate_scatter(units, phys, sim, y0)
    T = res["T"]
    Y = res["Y"]
    return row, T, Y, phys, sim, ic, loop


def plot_xy_trajectories(T: np.ndarray, Y: np.ndarray, out_path: str, cfg: FigureConfig, title: str = "") -> None:
    r = positions_from_Y(Y)  # (T,3,2)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(3):
        ax.plot(r[:, i, 0], r[:, i, 1], label=f"body {i}")
        ax.plot([r[0, i, 0]], [r[0, i, 1]], marker="o")
        ax.plot([r[-1, i, 0]], [r[-1, i, 1]], marker="x")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)




def plot_intruder_xy_bincom(T: np.ndarray, Y: np.ndarray, phys: PhysicalParams, out_path: str, cfg: FigureConfig) -> None:
    """Intruder trajectory in the frame of the binary COM (configuration space)."""
    r = positions_from_Y(Y)
    com12 = binary_com(r, phys.m1, phys.m2)
    rel = r[:, 2, :] - com12
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(rel[:, 0], rel[:, 1])
    ax.plot([rel[0, 0]], [rel[0, 1]], marker="o")
    ax.plot([rel[-1, 0]], [rel[-1, 1]], marker="x")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$x_3 - x_{\rm COM,12}$")
    ax.set_ylabel(r"$y_3 - y_{\rm COM,12}$")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_intruder_timeseries(T: np.ndarray, Y: np.ndarray, out_path_xy: str, out_path_vxy: str, cfg: FigureConfig) -> None:
    """Sky-plane observable: intruder x,y and vx,vy time series."""
    r = Y[:, :6].reshape(-1, 3, 2)
    v = Y[:, 6:].reshape(-1, 3, 2)
    x3 = r[:, 2, 0]; y3 = r[:, 2, 1]
    vx3 = v[:, 2, 0]; vy3 = v[:, 2, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, x3, label=r"$x_3(t)$")
    ax.plot(T, y3, label=r"$y_3(t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("position (sky-plane)")
    ax.legend(loc="best")
    savefig(fig, out_path_xy, cfg)
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, vx3, label=r"$v_{x,3}(t)$")
    ax.plot(T, vy3, label=r"$v_{y,3}(t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("velocity (sky-plane)")
    ax.legend(loc="best")
    savefig(fig, out_path_vxy, cfg)
    plt.close(fig)


def plot_separations(T: np.ndarray, Y: np.ndarray, out_path: str, cfg: FigureConfig) -> None:
    r = positions_from_Y(Y)
    d = pairwise_distances(r)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, d[:, 0], label=r"$r_{12}$")
    ax.plot(T, d[:, 1], label=r"$r_{23}$")
    ax.plot(T, d[:, 2], label=r"$r_{31}$")
    ax.set_xlabel("t")
    ax.set_ylabel("pair separation")
    ax.set_yscale("log")
    ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_intruder_distance(T: np.ndarray, Y: np.ndarray, phys: PhysicalParams, sim: SimParams, out_path: str, cfg: FigureConfig) -> None:
    r = positions_from_Y(Y)
    d3 = intruder_distance_to_binary_com(r, phys.m1, phys.m2)
    mask = interaction_mask(r, phys.m1, phys.m2, sim.R_int, pad=10)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, d3, label=r"$d_3(t)$")
    ax.axhline(sim.R_int, linestyle="--", linewidth=1.0, label=r"$R_{\rm int}$")
    # highlight interaction window
    if np.any(mask):
        ax.plot(T[mask], d3[mask], linewidth=2.0)
    ax.set_xlabel("t")
    ax.set_ylabel("intruder distance to binary COM")
    ax.set_yscale("log")
    ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_energy_error(T: np.ndarray, Y: np.ndarray, phys: PhysicalParams, out_path: str, cfg: FigureConfig) -> None:
    units = Units()
    E = _energy_series_newton(Y, units, phys)
    E0 = E[0]
    rel = (E - E0) / (abs(E0) + 1e-30)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(T, rel)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$(E(t)-E_0)/|E_0|$")
    ax.set_yscale("symlog", linthresh=1e-12)
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_shape_3d(T: np.ndarray, Y: np.ndarray, phys: PhysicalParams, sim: SimParams, out_path: str, cfg: FigureConfig, decimate: int = 5) -> None:
    r = positions_from_Y(Y)
    # use interaction window if available
    mask = interaction_mask(r, phys.m1, phys.m2, sim.R_int, pad=10)
    if np.sum(mask) >= 50:
        rr = r[mask]
    else:
        rr = r
    S = shape_logdist(rr)
    if decimate > 1 and len(S) > decimate:
        S = S[::decimate]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(S[:, 0], S[:, 1], S[:, 2])
    ax.set_xlabel(r"$\log r_{12}$")
    ax.set_ylabel(r"$\log r_{23}$")
    ax.set_zlabel(r"$\log r_{31}$")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_shape_scale_free(T: np.ndarray, Y: np.ndarray, phys: PhysicalParams, sim: SimParams, out_path: str, cfg: FigureConfig, decimate: int = 5) -> None:
    r = positions_from_Y(Y)
    mask = interaction_mask(r, phys.m1, phys.m2, sim.R_int, pad=10)
    if np.sum(mask) >= 50:
        rr = r[mask]
    else:
        rr = r
    d = pairwise_distances(rr)  # (T,3)
    # scale-free log ratios
    u = np.log((d[:, 0] + 1e-30) / (d[:, 1] + 1e-30))
    v = np.log((d[:, 1] + 1e-30) / (d[:, 2] + 1e-30))
    if decimate > 1 and len(u) > decimate:
        u = u[::decimate]
        v = v[::decimate]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(u, v)
    ax.set_xlabel(r"$\log(r_{12}/r_{23})$")
    ax.set_ylabel(r"$\log(r_{23}/r_{31})$")
    ax.set_aspect("equal", adjustable="datalim")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_loops_3d(loops, out_path: str, cfg: FigureConfig) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for k, L in enumerate(loops):
        C = L.curve
        ax.plot(C[:, 0], C[:, 1], C[:, 2], label=f"loop {k} (gap={L.gap:.2g})")
    ax.set_xlabel(r"$\log r_{12}$")
    ax.set_ylabel(r"$\log r_{23}$")
    ax.set_zlabel(r"$\log r_{31}$")
    if len(loops) <= 6:
        ax.legend(loc="best")
    savefig(fig, out_path, cfg)
    plt.close(fig)


def plot_link_matrix(Lmat: Optional[np.ndarray], out_path: str, cfg: FigureConfig) -> None:
    if Lmat is None or Lmat.size == 0:
        return
    K = Lmat.shape[0]
    fig = plt.figure(figsize=(cfg.figsize[0], cfg.figsize[0]))
    ax = fig.add_subplot(111)
    im = ax.imshow(Lmat, origin="lower", interpolation="nearest")
    ax.set_xlabel("loop index")
    ax.set_ylabel("loop index")
    # annotate integers if small
    if K <= 10:
        for i in range(K):
            for j in range(K):
                ax.text(j, i, str(int(Lmat[i, j])), ha="center", va="center", fontsize=max(6.0, cfg.fontsize-4))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out_path, cfg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate per-run figures (trajectory/shape/loops/linking) for a given seed.")
    ap.add_argument("--runs_csv", required=True, help="Path to out_runs/runs.csv")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out_dir", default=None, help="Directory to save figures (default: figs_seed<seed> next to runs.csv)")
    ap.add_argument("--raw_dir", default=None, help="Directory containing run_seed<seed>.npz (default: <runs_dir>/raw)")
    ap.add_argument("--rerun_if_missing", action="store_true", help="Rerun simulation if raw npz is missing")

    ap.add_argument("--fmt", default="pdf", choices=["pdf", "png", "svg"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--fontsize", type=float, default=10.0)
    ap.add_argument("--use_tex", action="store_true")
    ap.add_argument("--decimate", type=int, default=5, help="Decimation factor for plotting dense curves")

    args = ap.parse_args()

    cfg_fig = FigureConfig(fmt=args.fmt, dpi=args.dpi, fontsize=args.fontsize, use_tex=args.use_tex)
    set_paper_style(cfg_fig)

    runs_dir = os.path.dirname(args.runs_csv)
    out_dir = args.out_dir or os.path.join(runs_dir, f"figs_seed{args.seed}")
    ensure_dir(out_dir)

    row, T, Y, phys, sim, ic, loop = _load_or_rerun(args.runs_csv, args.seed, args.raw_dir, args.rerun_if_missing)

    # derive label from current Y too (for robustness)
    units = Units()
    fin = classify_final_pair(units, phys, Y[-1])
    label_now = "unknown"
    if "merged" in row and bool(row["merged"]):
        label_now = "merger_cut"
    else:
        label_now = outcome_label(fin.get("binary_pair"), initial_binary=(0,1))

    # Extract topo loops + linking matrix (recomputed from raw trajectory for plotting)
    topo, loops, Linfo = extract_topo_features(T, Y, phys, sim, loop)

    # Save summary JSON
    summary = {
        "seed": int(args.seed),
        "label_csv": str(row.get("label", "")),
        "label_now": str(label_now),
        "t_end": float(T[-1]),
        "topo": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in topo.items()},
        "linking": {
            "K": int(Linfo.get("K", 0)),
            "ok_frac": float(Linfo.get("ok_frac", 0.0)),
            "nz_frac": float(Linfo.get("nz_frac", 0.0)),
            "mean_abs": float(Linfo.get("mean_abs", 0.0)),
            "raw_mean_abs": float(Linfo.get("raw_mean_abs", 0.0)),
        },
    }
    if Linfo.get("L") is not None:
        summary["linking"]["L"] = Linfo["L"].tolist()

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figures
    ext = args.fmt
    plot_xy_trajectories(T, Y, os.path.join(out_dir, f"traj_xy.{ext}"), cfg_fig, title=f"seed={args.seed}, {label_now}")
    plot_intruder_xy_bincom(T, Y, phys, os.path.join(out_dir, f"intruder_xy_bincom.{ext}"), cfg_fig)
    plot_intruder_timeseries(T, Y, os.path.join(out_dir, f"obs_intruder_xy_vs_t.{ext}"), os.path.join(out_dir, f"obs_intruder_vxy_vs_t.{ext}"), cfg_fig)
    plot_separations(T, Y, os.path.join(out_dir, f"separations_vs_t.{ext}"), cfg_fig)
    plot_intruder_distance(T, Y, phys, sim, os.path.join(out_dir, f"intruder_dist_vs_t.{ext}"), cfg_fig)
    plot_energy_error(T, Y, phys, os.path.join(out_dir, f"energy_rel_error.{ext}"), cfg_fig)
    plot_shape_3d(T, Y, phys, sim, os.path.join(out_dir, f"shape_space_3d.{ext}"), cfg_fig, decimate=args.decimate)
    plot_shape_scale_free(T, Y, phys, sim, os.path.join(out_dir, f"shape_space_scalefree_2d.{ext}"), cfg_fig, decimate=args.decimate)

    if len(loops) > 0:
        plot_loops_3d(loops, os.path.join(out_dir, f"loops_3d.{ext}"), cfg_fig)
    if Linfo.get("L") is not None:
        plot_link_matrix(Linfo["L"], os.path.join(out_dir, f"link_matrix.{ext}"), cfg_fig)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()