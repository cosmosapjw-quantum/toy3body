from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .config import Units, PhysicalParams


def classify_final_pair(units: Units, phys: PhysicalParams, y_final: NDArray[np.float64]) -> dict:
    """Find most bound pair (if any) at final state and compute (a,e)."""
    G = units.G
    m = np.array([phys.m1, phys.m2, phys.m3], dtype=float)

    r = y_final[:6].reshape(3,2)
    v = y_final[6:].reshape(3,2)

    best_E = None
    best_pair = None
    for i in range(3):
        for j in range(i+1,3):
            dr = r[i]-r[j]
            dv = v[i]-v[j]
            rij = float(np.linalg.norm(dr))
            mu = m[i]*m[j]/(m[i]+m[j])
            E = 0.5*mu*float(np.dot(dv,dv)) - G*m[i]*m[j]/rij
            if best_E is None or E < best_E:
                best_E = E
                best_pair = (i,j)

    out = {"binary_pair": None, "escaper": None, "a": None, "e": None, "E_pair": None}
    if best_pair is None or best_E is None:
        return out

    if best_E < 0.0:
        i,j = best_pair
        k = 3 - i - j
        M = m[i]+m[j]

        a = -G*m[i]*m[j] / (2.0*best_E)

        dr = r[i]-r[j]
        dv = v[i]-v[j]
        h = dr[0]*dv[1] - dr[1]*dv[0]  # scalar z

        # e vector: (v x h)/GM - r/|r|
        # in-plane representation: v x h = (dv_y*h, -dv_x*h)
        evec = np.array([dv[1]*h, -dv[0]*h], dtype=float)/(G*M) - dr/np.linalg.norm(dr)
        e = float(np.linalg.norm(evec))

        out.update({"binary_pair": (i,j), "escaper": k, "a": float(a), "e": e, "E_pair": float(best_E)})
    return out


def outcome_label(binary_pair: tuple[int,int] | None, initial_binary: tuple[int,int]=(0,1)) -> str:
    if binary_pair is None:
        return "unknown"
    if binary_pair == initial_binary:
        return "escape"
    return "exchange"
