from __future__ import annotations

import time
from typing import Optional
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
                      max_runtime_sec=None,
                      rhs_fun=None) -> dict:
    """Chunked integration with early stopping and safety rails.

    Basin-map sweeps can contain pathological initial conditions:
      * extremely close encounters that make the solver stall,
      * solver failures (status=-1) that must terminate the loop.
    """

    t0 = 0.0
    y = y0.copy()

    rg = _rg_max_pair(units, phys)
    r_merge = phys.k_merge * rg if rg > 0 else -np.inf
    r_warn = phys.k_warn * rg if rg > 0 else -np.inf

    # numerical collision cut
    r_coll = float(getattr(sim, "r_coll", 0.0) or 0.0)
    r_stop = max((r_merge if np.isfinite(r_merge) and r_merge > 0 else 0.0), r_coll)
    use_stop_event = (r_stop > 0.0)

    pn_warning = False
    merged = False
    timeout = False
    stop_reason = ""

    E0 = energy_newton(y, units, phys)

    # keep at least initial point
    t_list: list[float] = [0.0]
    y_chunks: list[NDArray[np.float64]] = [y[None, :]]

    start = time.time()
    if max_runtime_sec is None:
        max_wall = float(getattr(sim, "max_walltime_sec", 0.0) or 0.0)
    else:
        max_wall = float(max_runtime_sec)
    min_adv = float(getattr(sim, "min_t_advance", 0.0) or 0.0)

    solver_success = True
    solver_status = 0
    solver_message = "OK"

    while t0 < sim.t_max:
        t1 = min(t0 + sim.dt_chunk, sim.t_max)
        t_prev = float(t0)

        def fun(t, yy):
            if max_wall > 0.0 and (time.time() - start) > max_wall:
                raise TimeoutError(f"max_walltime_sec={max_wall} exceeded")
            if rhs_fun is None:
                return rhs_newton(t, yy, units, phys)
            return rhs_fun(t, yy, units, phys)

        events = None
        if use_stop_event:
            def ev_stop(t, yy):
                return min_pair_distance(yy) - r_stop
            ev_stop.terminal = True
            ev_stop.direction = -1
            events = [ev_stop]

            max_step = sim.max_step
            if max_step is None:
                max_step = np.inf
            else:
                max_step = float(max_step)

            if not np.isfinite(max_step) or max_step <= 0:
                max_step = np.inf

        try:
            sol = solve_ivp(
                fun, (t0, t1), y,
                method=sim.method,
                rtol=sim.rtol, atol=sim.atol,
                max_step=max_step,
                events=events,
            )
        except TimeoutError:
            solver_success = False
            solver_status = -2
            solver_message = f"Aborted: walltime exceeded ({max_wall}s)"
            timeout = True
            stop_reason = "walltime"
            break
        except Exception as e:
            solver_success = False
            solver_status = -3
            solver_message = f"Exception in solve_ivp: {type(e).__name__}: {e}"
            timeout = True
            stop_reason = "exception"
            break

        solver_success = bool(sol.success)
        solver_status = int(sol.status)
        solver_message = str(sol.message)

        # append results (skip the duplicated first point)
        if sol.t.size > 1:
            t_list.extend(sol.t[1:].tolist())
            y_chunks.append(sol.y.T[1:])

        if sol.t.size == 0:
            solver_success = False
            solver_status = -4
            solver_message = "solve_ivp returned empty time array"
            timeout = True
            stop_reason = "solver_empty"
            break

        y = sol.y[:, -1].copy()
        t0 = float(sol.t[-1])

        if not np.all(np.isfinite(y)):
            solver_success = False
            solver_status = -5
            solver_message = "Non-finite state encountered (nan/inf)"
            timeout = True
            stop_reason = "nonfinite"
            break

        if min_adv > 0.0 and (t0 - t_prev) < min_adv:
            solver_success = False
            solver_status = -6
            solver_message = "Stalled: solver did not advance time"
            timeout = True
            stop_reason = "stalled"
            break

        # If SciPy reports failure (status=-1), stop immediately.
        # Otherwise basin sweeps can hang forever.
        if solver_status == -1 or (not solver_success):
            timeout = True
            stop_reason = "solver_failed"
            break

        if rg > 0 and min_pair_distance(y) < r_warn:
            pn_warning = True

        if use_stop_event and solver_status == 1:
            merged = True
            stop_reason = "r_merge" if (np.isfinite(r_merge) and r_merge >= r_coll and r_merge > 0) else "r_coll"
            break

        if _is_hierarchical(y, sim.hier_ratio) and (_max_radius_from_com(y, phys) > sim.R_end):
            stop_reason = "hierarchical_far"
            break

        # memory guard
        npts = sum(arr.shape[0] for arr in y_chunks)
        if npts > sim.max_store_points:
            Y = np.vstack(y_chunks)
            T = np.array(t_list, dtype=float)
            idx = np.linspace(0, len(T)-1, sim.decimate_to).astype(int)
            t_list = T[idx].tolist()
            y_chunks = [Y[idx]]

    if t0 >= sim.t_max and (not merged) and (not timeout):
        timeout = True
        stop_reason = "t_max"

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
        "stop_reason": stop_reason,
        "r_stop": float(r_stop),
        "E0": float(E0),
        "E_end": float(E_end),
        "runtime_sec": float(time.time() - start),
        "solver_success": bool(solver_success),
        "solver_status": int(solver_status),
        "solver_message": str(solver_message),
    }
