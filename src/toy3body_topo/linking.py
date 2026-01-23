from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _prep_polyline(C: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (midpoints, dvec) for closed polyline C (N+1,3)."""
    d = C[1:] - C[:-1]                 # (N,3)
    mid = 0.5*(C[1:] + C[:-1])         # (N,3)
    return mid, d


def gauss_linking(midP: NDArray[np.float64], dP: NDArray[np.float64],
                  midQ: NDArray[np.float64], dQ: NDArray[np.float64]) -> float:
    """Discretized Gauss linking integral using segment midpoints (Riemann sum).

    Returns a real number which should be near an integer when curves are well separated.
    """
    # Pairwise displacement between midpoints
    R = midP[:, None, :] - midQ[None, :, :]  # (N,M,3)
    cross = np.cross(dP[:, None, :], dQ[None, :, :])  # (N,M,3)
    num = np.einsum("ijk,ijk->ij", R, cross)
    den = np.linalg.norm(R, axis=-1)**3
    with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(den > 0, num/den, 0.0).sum()
    return float(val / (4.0*np.pi))


def min_vertex_distance(C1: NDArray[np.float64], C2: NDArray[np.float64]) -> float:
    A = C1[:-1]
    B = C2[:-1]
    D = A[:, None, :] - B[None, :, :]
    d = np.linalg.norm(D, axis=-1)
    return float(np.min(d))


def safe_linking_integer(C1: NDArray[np.float64], C2: NDArray[np.float64],
                         round_tol: float = 0.1,
                         min_sep: float = 1e-3) -> tuple[bool, int, float]:
    dmin = min_vertex_distance(C1, C2)
    if dmin < min_sep:
        return (False, 0, np.nan)

    mid1, d1 = _prep_polyline(C1)
    mid2, d2 = _prep_polyline(C2)
    lk = gauss_linking(mid1, d1, mid2, d2)
    lk_int = int(np.rint(lk))
    ok = abs(lk - lk_int) < round_tol
    return (ok, lk_int, float(lk))


def linking_fingerprint(curves: list[NDArray[np.float64]],
                        round_tol: float,
                        min_sep: float) -> dict:
    """Compute linking matrix + summary stats from list of closed polylines."""
    K = len(curves)
    if K < 2:
        return {
            "K": K,
            "ok_frac": 0.0,
            "nz_frac": 0.0,
            "mean_abs": 0.0,
            "raw_mean_abs": 0.0,
            "L": None,
        }

    prepped = [_prep_polyline(C) for C in curves]

    L = np.zeros((K, K), dtype=int)
    okmask = np.zeros((K, K), dtype=bool)
    raw = np.full((K, K), np.nan, dtype=float)

    for i in range(K):
        for j in range(i+1, K):
            # quick min-sep check
            if min_vertex_distance(curves[i], curves[j]) < min_sep:
                continue
            mid1, d1 = prepped[i]
            mid2, d2 = prepped[j]
            lk = gauss_linking(mid1, d1, mid2, d2)
            raw[i, j] = raw[j, i] = lk
            lk_int = int(np.rint(lk))
            if abs(lk - lk_int) < round_tol:
                okmask[i, j] = okmask[j, i] = True
                L[i, j] = L[j, i] = lk_int

    upper = [(i, j) for i in range(K) for j in range(i+1, K)]
    oks = np.array([okmask[i, j] for i, j in upper], dtype=bool)
    ints = np.array([L[i, j] for i, j in upper], dtype=int)
    raws = np.array([raw[i, j] for i, j in upper], dtype=float)

    ok_frac = float(np.mean(oks)) if len(oks) else 0.0
    if np.any(oks):
        nz_frac = float(np.mean(np.abs(ints[oks]) > 0))
        mean_abs = float(np.mean(np.abs(ints[oks])))
    else:
        nz_frac = 0.0
        mean_abs = 0.0
    raw_mean_abs = float(np.nanmean(np.abs(raws))) if np.any(np.isfinite(raws)) else 0.0

    return {
        "K": K,
        "ok_frac": ok_frac,
        "nz_frac": nz_frac,
        "mean_abs": mean_abs,
        "raw_mean_abs": raw_mean_abs,
        "L": L,
    }


def gauss_writhe(curve: NDArray[np.float64], skip_near: int = 2) -> float:
    """Approximate writhe of a (closed) 3D curve using a Gauss integral discretization.

    Writhe is a geometric/topological proxy (real-valued) that can be computed from a single loop,
    unlike the integer Gauss linking number which needs two loops.

    Parameters
    ----------
    curve : (N+1,3) closed polyline (last == first)
    skip_near : exclude pairs of nearby segments (to avoid near-singular contributions)

    Returns
    -------
    Wr : float
    """
    curve = np.asarray(curve, dtype=float)
    if curve.ndim != 2 or curve.shape[0] < 5:
        return 0.0

    # segments + midpoints
    dr = np.diff(curve, axis=0)  # (N,3)
    mid = 0.5 * (curve[:-1] + curve[1:])  # (N,3)
    N = dr.shape[0]

    R = mid[:, None, :] - mid[None, :, :]          # (N,N,3)
    denom = np.linalg.norm(R, axis=-1) ** 3 + 1e-30

    cross = np.cross(dr[:, None, :], dr[None, :, :])  # (N,N,3)
    num = np.einsum("ijk,ijk->ij", R, cross)

    mask = np.ones((N, N), dtype=bool)
    np.fill_diagonal(mask, False)

    kmax = max(0, int(skip_near))
    for k in range(1, kmax + 1):
        # near-diagonal
        mask[np.arange(0, N - k), np.arange(k, N)] = False
        mask[np.arange(k, N), np.arange(0, N - k)] = False
        # wrap-around adjacency on a closed curve
        mask[np.arange(0, k), np.arange(N - k, N)] = False
        mask[np.arange(N - k, N), np.arange(0, k)] = False

    val = (1.0 / (4.0 * np.pi)) * float(np.sum((num / denom)[mask]))
    return float(val)
