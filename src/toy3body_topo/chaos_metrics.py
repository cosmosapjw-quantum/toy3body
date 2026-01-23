from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def survival_function(times: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (t, S(t)) empirical survival function."""
    t = np.sort(times.astype(float))
    uniq = np.unique(t)
    S = np.array([np.mean(t >= u) for u in uniq], dtype=float)
    return uniq, S


def fit_escape_rate(times: NDArray[np.float64], t_min: float = 0.0) -> dict:
    """Fit survival tail S(t) ~ exp(-kappa t) by linear regression on log S."""
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times)]
    times = times[times >= t_min]
    if len(times) < 30:
        return {"kappa": np.nan, "note": "too_few", "t_min": float(t_min), "n": int(len(times))}
    t, S = survival_function(times)
    mask = (S > 0.0) & (S < 1.0) & (t >= t_min)
    if np.sum(mask) < 10:
        return {"kappa": np.nan, "note": "bad_tail", "t_min": float(t_min), "n": int(len(times))}
    x = t[mask]
    y = np.log(S[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return {"kappa": float(-slope), "intercept": float(intercept), "t_min": float(t_min), "n": int(len(times))}


def basin_entropy_boxed(grid: NDArray[np.int64], n_outcomes: int, box: int) -> dict:
    """Compute basin entropy by boxing a label grid.

    grid: (nx, ny) integer labels in [0, n_outcomes-1]
    box: box size (pixels)

    Returns:
      Sb: average Shannon entropy over all boxes
      Sbb: average entropy over mixed boxes only (boundary basin entropy)
      frac_mixed: fraction of boxes containing >1 outcome
    """
    g = np.asarray(grid, dtype=int)
    nx, ny = g.shape
    bx = nx // box
    by = ny // box
    if bx == 0 or by == 0:
        raise ValueError("box too large for grid")
    ent = []
    ent_mixed = []
    mixed = 0
    total = 0

    for i in range(bx):
        for j in range(by):
            block = g[i*box:(i+1)*box, j*box:(j+1)*box].ravel()
            # Filter out negative values (invalid/uncomputed data)
            block = block[block >= 0]
            if len(block) == 0:
                # Skip empty blocks
                total += 1
                continue
            counts = np.bincount(block, minlength=n_outcomes).astype(float)
            p = counts / counts.sum()
            # Shannon entropy
            h = -np.sum(np.where(p > 0, p*np.log(p), 0.0))
            ent.append(h)
            total += 1
            if np.count_nonzero(counts) > 1:
                mixed += 1
                ent_mixed.append(h)

    Sb = float(np.mean(ent)) if ent else 0.0
    Sbb = float(np.mean(ent_mixed)) if ent_mixed else 0.0
    frac_mixed = mixed/total if total else 0.0
    return {"Sb": Sb, "Sbb": Sbb, "frac_mixed": float(frac_mixed), "box": int(box), "n_boxes": int(total)}
