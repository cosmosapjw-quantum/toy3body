from __future__ import annotations

import time
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .config import Units, PhysicalParams, SimParams
from .rhs import rhs_newton, min_pair_distance, energy_newton


def _rg_max_pair(units: Units, phys: PhysicalParams) -> float:
    """Conservative r_g scale using max pair mass (m_i+m_j)/c^2."""
    c = phys.c
    if c is None:
        return 0.0
    m = np.array([phys.m1, phys.m2, phys.m3], dtype=float)
    msum_max = float(np.max([m[0]+m[1], m[1]+m[2], m[2]+m[0]]))
    return units.G * msum_max / (c*c)


def _is_hierarchical(y: NDArray[np.float64], ratio: float) -> bool:
    x1, y1, x2, y2, x3, y3 = y[0], y[1], y[2], y[3], y[4], y[5]
    d12 = (x2-x1)**2 + (y2-y1)**2
    d13 = (x3-x1)**2 + (y3-y1)**2
    d23 = (x3-x2)**2 + (y3-y2)**2
    ds = np.sort([d12, d13, d23])
    return (ds[1] / ds[0]) > (ratio*ratio)


def _max_radius_from_com(y: NDArray[np.float64], phys: PhysicalParams) -> float:
    m1, m2, m3 = phys.m1, phys.m2, phys.m3
    M = m1+m2+m3
    x1, y1, x2, y2, x3, y3 = y[0], y[1], y[2], y[3], y[4], y[5]
    xcom = (m1*x1 + m2*x2 + m3*x3)/M
    ycom = (m1*y1 + m2*y2 + m3*y3)/M
    r1 = (x1-xcom)**2 + (y1-ycom)**2
    r2 = (x2-xcom)**2 + (y2-ycom)**2
    r3 = (x3-xcom)**2 + (y3-ycom)**2
    return float(np.sqrt(max(r1, r2, r3)))


def integrate_scatter(units: Units,
                      phys: PhysicalParams,
                      sim: SimParams,
                      y0: NDArray[np.float64],
                      rhs_fun=None) -> dict:
    """Chunked integration with early stopping and merger-cut event."""
    t0 = 0.0
    y = y0.copy()

    rg = _rg_max_pair(units, phys)
    r_merge = phys.k_merge * rg if rg > 0 else -np.inf
    r_warn = phys.k_warn * rg if rg > 0 else -np.inf

    pn_warning = False
    merged = False
    timeout = False

    # diagnostics
    E0 = energy_newton(y, units, phys)

    t_list: list[float] = []
    y_chunks: list[NDArray[np.float64]] = []

    start = time.time()
    while t0 < sim.t_max:
        t1 = min(t0 + sim.dt_chunk, sim.t_max)

        def fun(t, yy):
            if rhs_fun is None:
                return rhs_newton(t, yy, units, phys)
            return rhs_fun(t, yy, units, phys)

        # merger event: terminate if min pair distance crosses r_merge downward
        def ev_merge(t, yy):
            return min_pair_distance(yy) - r_merge
        ev_merge.terminal = True
        ev_merge.direction = -1

        sol = solve_ivp(
            fun, (t0, t1), y,
            method=sim.method,
            rtol=sim.rtol, atol=sim.atol,
            max_step=sim.max_step,
            events=[ev_merge],
        )

        # append (avoid duplicating first point across chunks)
        if not t_list:
            t_list.extend(sol.t.tolist())
            y_chunks.append(sol.y.T)
        else:
            t_list.extend(sol.t[1:].tolist())
            y_chunks.append(sol.y.T[1:])

        y = sol.y[:, -1].copy()
        t0 = float(sol.t[-1])

        if rg > 0 and min_pair_distance(y) < r_warn:
            pn_warning = True

        if sol.status == 1:
            merged = True
            break

        # early stop: clearly separated + far field
        if _is_hierarchical(y, sim.hier_ratio) and (_max_radius_from_com(y, phys) > sim.R_end):
            break

        # memory guard
        npts = sum(arr.shape[0] for arr in y_chunks)
        if npts > sim.max_store_points:
            Y = np.vstack(y_chunks)
            T = np.array(t_list, dtype=float)
            idx = np.linspace(0, len(T)-1, sim.decimate_to).astype(int)
            t_list = T[idx].tolist()
            y_chunks = [Y[idx]]

    if t0 >= sim.t_max and (not merged):
        timeout = True

    T = np.array(t_list, dtype=float)
    Y = np.vstack(y_chunks)

    E_end = energy_newton(Y[-1], units, phys)

    return {
        "T": T,
        "Y": Y,
        "t_end": float(T[-1]),
        "merged": merged,
        "timeout": timeout,
        "pn_warning": pn_warning,
        "E0": float(E0),
        "E_end": float(E_end),
        "runtime_sec": float(time.time() - start),
    }
