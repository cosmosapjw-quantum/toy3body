from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import PhysicalParams, SimParams, LoopParams
from .shape import (
    positions_from_Y,
    shape_logdist,
    shape_uvrho,
    interaction_mask,
    intruder_distance_to_binary_com,
    binary_com,
)
from .embedding import delay_embed_1d
from .loops import extract_loops
from .linking import linking_fingerprint, gauss_writhe


def _min_dist_over_time(r: NDArray[np.float64]) -> float:
    d01 = np.linalg.norm(r[:, 0, :] - r[:, 1, :], axis=1)
    d12 = np.linalg.norm(r[:, 1, :] - r[:, 2, :], axis=1)
    d20 = np.linalg.norm(r[:, 2, :] - r[:, 0, :], axis=1)
    return float(np.min(np.vstack([d01, d12, d20])))


def _interaction_time(T: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    if len(T) < 2:
        return 0.0
    dt = np.diff(T)
    m = mask.astype(float)
    m_mid = 0.5 * (m[:-1] + m[1:])
    return float(np.sum(dt * m_mid))


def _standardize(S: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    mu = np.mean(S, axis=0)
    sig = np.std(S, axis=0)
    sig = np.where(sig > eps, sig, 1.0)
    return (S - mu) / sig


def _winding_metrics(r: NDArray[np.float64],
                     T: NDArray[np.float64],
                     m1: float,
                     m2: float,
                     mask: NDArray[np.bool_] | None = None) -> dict:
    """Intruder winding around the binary COM during the interaction window."""
    if mask is None or np.sum(mask) < 5:
        rr = r
        tt = T
    else:
        rr = r[mask]
        tt = T[mask]

    com = binary_com(rr, m1, m2)
    rel = rr[:, 2, :] - com
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    ang_u = np.unwrap(ang)

    if len(ang_u) < 2:
        return {"wind_raw": 0.0, "wind_abs": 0.0, "wind_total": 0.0}

    wind_raw = float((ang_u[-1] - ang_u[0]) / (2.0 * np.pi))
    wind_abs = float(abs(wind_raw))
    wind_total = float((np.max(ang_u) - np.min(ang_u)) / (2.0 * np.pi))
    return {"wind_raw": wind_raw, "wind_abs": wind_abs, "wind_total": wind_total}


def extract_topo_features(T: NDArray[np.float64],
                          Y: NDArray[np.float64],
                          phys: PhysicalParams,
                          sim: SimParams,
                          loop_params: LoopParams) -> tuple[dict, list, dict]:
    """Compute interaction / loop / linking features from a single trajectory.

    Returns
    -------
    features : dict
        Scalar features that go into CSV.
    loops : list[Loop]
        Extracted loops (each has a closed curve).
    Linfo : dict
        Linking fingerprint summary + the matrix (if K>=2).
    """
    r = positions_from_Y(Y)

    # basic distances
    rmin = _min_dist_over_time(r)
    d3 = intruder_distance_to_binary_com(r, phys.m1, phys.m2)

    mask = interaction_mask(r, phys.m1, phys.m2, sim.R_int, pad=10)
    tint = _interaction_time(T, mask)

    # choose analysis window (interaction window if long enough)
    if np.sum(mask) >= 50:
        rr = r[mask]
        tt = T[mask]
        d3_sub = d3[mask]
    else:
        rr = r
        tt = T
        d3_sub = d3

    # build embedding S(tt)
    mode = str(getattr(loop_params, "shape_mode", "uvrho"))
    if mode == "logdist":
        S = shape_logdist(rr)
    elif mode == "uvrho":
        S = shape_uvrho(rr)
    elif mode == "delay_d3":
        x = np.log(d3_sub + 1e-15)
        tt, S = delay_embed_1d(tt, x, tau=float(loop_params.delay_tau), dim=3, n_uniform=int(loop_params.delay_uniform_n))
    else:
        # fallback
        S = shape_uvrho(rr)

    # standardize for numerical stability (affine transforms preserve linking)
    if bool(getattr(loop_params, "standardize", True)) and S.shape[1] >= 2:
        S = _standardize(S)

    # downweight rho for NN search (but keep it nonzero so the curve is not planar)
    if mode == "uvrho" and S.shape[1] >= 3:
        w = float(getattr(loop_params, "nn_rho_weight", 1.0))
        S = S.copy()
        S[:, 2] *= max(1e-6, w)

    # loop extraction: base vs long dwell
    t_end = float(T[-1])
    max_loops = loop_params.max_loops_long if t_end >= loop_params.t_long else loop_params.max_loops_base

    loops = extract_loops(
        S, tt,
        eps_close=float(loop_params.eps_close),
        min_dt=float(loop_params.min_dt),
        k_nn=int(loop_params.k_nn),
        max_loops=int(max_loops),
        resample_n=int(loop_params.resample_n),
        max_pairs_per_i=int(getattr(loop_params, "max_pairs_per_i", 1)),
        max_overlap_frac=float(getattr(loop_params, "max_overlap_frac", 0.2)),
    )

    curves = [L.curve for L in loops]
    Linfo = linking_fingerprint(
        curves,
        round_tol=float(loop_params.link_round_tol),
        min_sep=float(loop_params.link_min_sep),
    )

    # loop stats
    if loops:
        mean_gap = float(np.mean([L.gap for L in loops]))
        max_dur = float(np.max([L.duration for L in loops]))
        wr = [gauss_writhe(L.curve) for L in loops]
        loop_writhe_mean = float(np.mean(wr))
        loop_writhe_abs_mean = float(np.mean(np.abs(wr)))
        loop_writhe_std = float(np.std(wr))
    else:
        mean_gap = 0.0
        max_dur = 0.0
        loop_writhe_mean = 0.0
        loop_writhe_abs_mean = 0.0
        loop_writhe_std = 0.0

    wind = _winding_metrics(r, T, phys.m1, phys.m2, mask=mask)

    feats = {
        "t_end": t_end,
        "t_int": float(tint),
        "r_min": float(rmin),
        "d3_min": float(np.min(d3)),
        "n_loops": int(Linfo["K"]),
        "loop_mean_gap": float(mean_gap),
        "loop_max_dur": float(max_dur),
        "loop_writhe_mean": float(loop_writhe_mean),
        "loop_writhe_abs_mean": float(loop_writhe_abs_mean),
        "loop_writhe_std": float(loop_writhe_std),
        "link_ok_frac": float(Linfo["ok_frac"]),
        "link_nz_frac": float(Linfo["nz_frac"]),
        "link_mean_abs": float(Linfo["mean_abs"]),
        "link_raw_mean_abs": float(Linfo["raw_mean_abs"]),
        "wind_raw": float(wind["wind_raw"]),
        "wind_abs": float(wind["wind_abs"]),
        "wind_total": float(wind["wind_total"]),
    }

    return feats, loops, Linfo
