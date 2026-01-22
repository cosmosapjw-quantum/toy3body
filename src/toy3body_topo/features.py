from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .config import PhysicalParams, SimParams, LoopParams
from .shape import positions_from_Y, shape_logdist, interaction_mask, pairwise_distances, intruder_distance_to_binary_com
from .loops import extract_loops
from .linking import linking_fingerprint


def _min_dist_over_time(r: NDArray[np.float64]) -> float:
    d01 = np.linalg.norm(r[:,0,:]-r[:,1,:], axis=1)
    d12 = np.linalg.norm(r[:,1,:]-r[:,2,:], axis=1)
    d20 = np.linalg.norm(r[:,2,:]-r[:,0,:], axis=1)
    return float(np.min(np.vstack([d01, d12, d20])))


def _interaction_time(T: NDArray[np.float64], mask: NDArray[np.bool_]) -> float:
    if len(T) < 2:
        return 0.0
    # approximate integral by trapezoid rule on mask
    dt = np.diff(T)
    m = mask.astype(float)
    # average mask on intervals
    m_mid = 0.5*(m[:-1] + m[1:])
    return float(np.sum(dt * m_mid))


def extract_topo_features(T: NDArray[np.float64],
                          Y: NDArray[np.float64],
                          phys: PhysicalParams,
                          sim: SimParams,
                          loop_params: LoopParams) -> dict:
    r = positions_from_Y(Y)

    # basic distances
    rmin = _min_dist_over_time(r)
    d3 = intruder_distance_to_binary_com(r, phys.m1, phys.m2)

    mask = interaction_mask(r, phys.m1, phys.m2, sim.R_int, pad=10)
    tint = _interaction_time(T, mask)

    # shape series (restrict to interaction window if possible)
    if np.sum(mask) >= 50:
        S = shape_logdist(r[mask])
        tt = T[mask]
    else:
        S = shape_logdist(r)
        tt = T

    # loop extraction: mixed strategy (base vs long dwell)
    t_end = float(T[-1])
    max_loops = loop_params.max_loops_long if t_end >= loop_params.t_long else loop_params.max_loops_base

    loops = extract_loops(S, tt,
                          eps_close=loop_params.eps_close,
                          min_dt=loop_params.min_dt,
                          k_nn=loop_params.k_nn,
                          max_loops=max_loops,
                          resample_n=loop_params.resample_n)

    curves = [L.curve for L in loops]
    Linfo = linking_fingerprint(curves,
                                round_tol=loop_params.link_round_tol,
                                min_sep=loop_params.link_min_sep)

    # additional loop stats
    if loops:
        mean_gap = float(np.mean([L.gap for L in loops]))
        max_dur = float(np.max([L.duration for L in loops]))
    else:
        mean_gap = 0.0
        max_dur = 0.0

    return {
        "t_end": t_end,
        "t_int": float(tint),
        "r_min": rmin,
        "d3_min": float(np.min(d3)),
        "n_loops": int(Linfo["K"]),
        "loop_mean_gap": mean_gap,
        "loop_max_dur": max_dur,
        "link_ok_frac": float(Linfo["ok_frac"]),
        "link_nz_frac": float(Linfo["nz_frac"]),
        "link_mean_abs": float(Linfo["mean_abs"]),
        "link_raw_mean_abs": float(Linfo["raw_mean_abs"]),
        # do not store full matrix in CSV; store separately if needed
    }, loops, Linfo
