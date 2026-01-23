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



def shape_uvrho(r: NDArray[np.float64], eps: float = 1e-15) -> NDArray[np.float64]:
    """Shape+scale coordinates built from log pairwise distances.

    Returns S(t) = (u, v, rho) where:
      - rho is the mean log-distance (log of geometric-mean scale)
      - (u, v) are an orthonormal projection of the *shape* part
        (log-distances with rho removed) onto a symmetric 2D basis.

    This is useful for close-return detection because:
      - recurrent *shape* can occur at different overall scales in scattering,
      - separating scale reduces spurious "no loop found" failures.

    Notes:
      - (u, v) live in the sum-zero plane of (log r12, log r23, log r31)
      - any orthonormal basis of that plane is fine; we use a standard symmetric choice.
    """
    s = shape_logdist(r, eps=eps)  # (T,3)
    rho = np.mean(s, axis=1)       # (T,)
    s0 = s - rho[:, None]          # shape part, sum ~ 0

    # Orthonormal basis for the sum-zero plane in R^3
    e1 = np.array([1.0, -1.0, 0.0], dtype=float) / np.sqrt(2.0)
    e2 = np.array([1.0, 1.0, -2.0], dtype=float) / np.sqrt(6.0)
    u = s0 @ e1
    v = s0 @ e2
    return np.stack([u, v, rho], axis=1)


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
