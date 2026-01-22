from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def positions_from_Y(Y: NDArray[np.float64]) -> NDArray[np.float64]:
    return Y[:, :6].reshape(-1, 3, 2)


def pairwise_distances(r: NDArray[np.float64]) -> NDArray[np.float64]:
    """r: (T,3,2) -> d: (T,3) = [r12, r23, r31]"""
    r12 = np.linalg.norm(r[:,0,:]-r[:,1,:], axis=1)
    r23 = np.linalg.norm(r[:,1,:]-r[:,2,:], axis=1)
    r31 = np.linalg.norm(r[:,2,:]-r[:,0,:], axis=1)
    return np.vstack([r12, r23, r31]).T


def shape_logdist(r: NDArray[np.float64], eps: float = 1e-15) -> NDArray[np.float64]:
    d = pairwise_distances(r)
    return np.log(d + eps)


def binary_com(r: NDArray[np.float64], m1: float, m2: float) -> NDArray[np.float64]:
    """Binary COM trajectory: (T,2)"""
    M = m1 + m2
    return (m1*r[:,0,:] + m2*r[:,1,:]) / M


def intruder_distance_to_binary_com(r: NDArray[np.float64], m1: float, m2: float) -> NDArray[np.float64]:
    com12 = binary_com(r, m1, m2)
    d = np.linalg.norm(r[:,2,:] - com12, axis=1)
    return d


def interaction_mask(r: NDArray[np.float64], m1: float, m2: float, R_int: float, pad: int = 5) -> NDArray[np.bool_]:
    """Mask indices where intruder is within R_int of binary COM (with optional padding)."""
    d = intruder_distance_to_binary_com(r, m1, m2)
    mask = d < R_int
    if not np.any(mask):
        return mask
    idx = np.where(mask)[0]
    i0, i1 = idx[0], idx[-1]
    i0 = max(i0 - pad, 0)
    i1 = min(i1 + pad, len(mask)-1)
    out = np.zeros_like(mask)
    out[i0:i1+1] = True
    return out
