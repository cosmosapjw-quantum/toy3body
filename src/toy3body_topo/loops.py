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
    curve: NDArray[np.float64]  # (N+1,3) closed polyline (last==first)


def _resample_polyline_arclen(P: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """Resample a 3D polyline by arclength."""
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2 or n < 2:
        return np.repeat(P[:1], max(2, int(n)), axis=0)

    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if not np.isfinite(s[-1]) or s[-1] <= 0.0:
        return np.repeat(P[:1], int(n), axis=0)

    s_new = np.linspace(0.0, s[-1], int(n), dtype=np.float64)
    out = np.empty((int(n), P.shape[1]), dtype=np.float64)
    for k in range(P.shape[1]):
        out[:, k] = np.interp(s_new, s, P[:, k])
    return out


def extract_loops(S: NDArray[np.float64],
                  t: NDArray[np.float64],
                  eps_close: float,
                  min_dt: float,
                  k_nn: int,
                  max_loops: int,
                  resample_n: int,
                  max_pairs_per_i: int = 1,
                  max_overlap_frac: float = 0.2) -> list[Loop]:
    """Extract nearly-closed loops from close returns in an embedding S(t).

    Strategy
    --------
    - Build a KD-tree in the embedding.
    - For each point i, look at its k nearest neighbors (excluding itself).
    - Keep up to `max_pairs_per_i` candidate (i,j) pairs with:
        dist(S[i], S[j]) < eps_close
        t[j]-t[i] >= min_dt
    - Sort candidates by (gap, -duration) and greedily accept non-overlapping loops.

    Notes
    -----
    - The extracted curve is explicitly *closed* by appending the start point.
      This is critical for Gauss linking / writhe computations.
    - `max_overlap_frac` lets you trade off: more loops vs more redundancy.
    """
    S = np.asarray(S, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    n = len(S)
    if n < 20:
        return []

    tree = cKDTree(S)
    candidates: list[tuple[float, int, int, float]] = []

    kq = max(2, int(k_nn) + 1)  # +1 because NN often returns itself
    max_pairs_per_i = max(1, int(max_pairs_per_i))

    for i in range(n):
        dists, idxs = tree.query(S[i], k=kq)
        if np.isscalar(dists):
            continue
        kept = 0
        for dist, j in zip(dists[1:], idxs[1:]):  # skip itself
            j = int(j)
            if j <= i:
                continue
            if dist > eps_close:
                # query returns sorted by distance -> stop early
                break
            dt = float(t[j] - t[i])
            if dt < min_dt:
                continue
            candidates.append((float(dist), int(i), int(j), dt))
            kept += 1
            if kept >= max_pairs_per_i:
                break

    if not candidates:
        return []

    # sort: smallest closure gap first; tie-breaker longer duration
    candidates.sort(key=lambda x: (x[0], -x[3]))

    used = np.zeros(n, dtype=bool)
    loops: list[Loop] = []

    for gap, i0, i1, dur in candidates:
        if used[i0] or used[i1]:
            continue
        # skip if too much overlap with already-used points
        if float(np.mean(used[i0:i1+1])) > float(max_overlap_frac):
            continue

        seg = S[i0:i1+1]
        seg_rs = _resample_polyline_arclen(seg, int(resample_n))
        curve = np.vstack([seg_rs, seg_rs[0:1]])  # close

        loops.append(Loop(i0=i0, i1=i1, gap=float(gap), duration=float(dur), curve=curve))
        used[i0:i1+1] = True

        if len(loops) >= int(max_loops):
            break

    return loops
