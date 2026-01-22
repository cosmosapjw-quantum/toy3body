from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


@dataclass
class Loop:
    i0: int
    i1: int
    gap: float
    duration: float
    curve: NDArray[np.float64]  # (N+1,3) closed polyline


def _resample_polyline_arclen(P: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    # P: (M,3)
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0.0:
        return np.repeat(P[:1], n, axis=0)
    s_new = np.linspace(0.0, s[-1], n)
    out = np.empty((n, P.shape[1]), dtype=np.float64)
    for k in range(P.shape[1]):
        out[:,k] = np.interp(s_new, s, P[:,k])
    return out


def extract_loops(S: NDArray[np.float64],
                  t: NDArray[np.float64],
                  eps_close: float,
                  min_dt: float,
                  k_nn: int,
                  max_loops: int,
                  resample_n: int) -> list[Loop]:
    """Extract nearly-closed loops from close returns in S(t) (shape space).

    Optimization idea:
      - KDTree + k-nearest neighbors per point (small k), instead of radius queries per point.
      - Candidate list sorted by closure gap, then greedy non-overlap selection.
    """
    n = len(S)
    if n < 20:
        return []

    tree = cKDTree(S)

    candidates = []
    # query k_nn+1 because nearest neighbor is often itself
    kq = max(2, k_nn + 1)
    for i in range(n):
        dists, idxs = tree.query(S[i], k=kq)
        # make arrays
        if np.isscalar(dists):
            continue
        for dist, j in zip(dists[1:], idxs[1:]):  # skip itself
            if j <= i:
                continue
            if dist > eps_close:
                # since query returns sorted by distance, break early
                break
            if (t[j] - t[i]) < min_dt:
                continue
            candidates.append((float(dist), i, int(j), float(t[j]-t[i])))
            break  # keep only best match for this i

    if not candidates:
        return []

    # sort: smallest gap first, tie-breaker longer duration
    candidates.sort(key=lambda x: (x[0], -x[3]))

    used = np.zeros(n, dtype=bool)
    loops: list[Loop] = []
    for gap, i0, i1, dur in candidates:
        if used[i0] or used[i1]:
            continue
        # avoid heavy overlap: skip if too many points already used in [i0,i1]
        if np.mean(used[i0:i1+1]) > 0.2:
            continue

        seg = S[i0:i1+1]
        seg_rs = _resample_polyline_arclen(seg, resample_n)
        curve = np.vstack([seg_rs, seg_rs[0:1]])
        loops.append(Loop(i0=i0, i1=i1, gap=gap, duration=dur, curve=curve))

        used[i0:i1+1] = True
        if len(loops) >= max_loops:
            break

    return loops
