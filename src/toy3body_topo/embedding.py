from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def resample_uniform(T: NDArray[np.float64], X: NDArray[np.float64], n: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample a 1D time series X(T) onto a uniform grid.

    Parameters
    ----------
    T : (N,) increasing times
    X : (N,) values
    n : number of uniform samples

    Returns
    -------
    Tu : (n,) uniform times in [T[0], T[-1]]
    Xu : (n,) values linearly interpolated on Tu

    Notes
    -----
    - We deliberately keep this simple and dependency-free (np.interp).
    - For the toy model, this is good enough to build a robust delay embedding.
    """
    if n <= 2 or len(T) < 2:
        return T.copy(), X.copy()
    t0 = float(T[0]); t1 = float(T[-1])
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return T.copy(), X.copy()
    Tu = np.linspace(t0, t1, int(n), dtype=float)
    Xu = np.interp(Tu, T.astype(float), X.astype(float))
    return Tu, Xu


def delay_embed_1d(T: NDArray[np.float64],
                   X: NDArray[np.float64],
                   tau: float,
                   dim: int = 3,
                   n_uniform: int = 2048) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Takens-style delay embedding of a scalar observable into R^dim.

    Steps:
      1) Uniformly resample X(T) to make a constant dt.
      2) Convert the physical lag tau into an integer step-lag.
      3) Build the embedding (X(t), X(t+tau), ..., X(t+(dim-1)tau)).

    Returns
    -------
    Te : (M,) times aligned with the embedding start
    S  : (M, dim) embedded points

    If the embedding is impossible (too short), returns (T, X[:,None]) as a fallback.
    """
    T = np.asarray(T, dtype=float)
    X = np.asarray(X, dtype=float)
    if len(T) < 4 or dim < 2:
        return T.copy(), X.reshape(-1, 1).copy()

    Tu, Xu = resample_uniform(T, X, n=int(n_uniform))
    dt = float(Tu[1] - Tu[0])
    if dt <= 0:
        return T.copy(), X.reshape(-1, 1).copy()

    lag = int(max(1, round(float(tau) / dt)))
    max_lag = (dim - 1) * lag
    if len(Tu) <= max_lag + 2:
        # too short -> fallback
        return T.copy(), X.reshape(-1, 1).copy()

    M = len(Tu) - max_lag
    S = np.empty((M, dim), dtype=float)
    for k in range(dim):
        S[:, k] = Xu[k * lag: k * lag + M]
    Te = Tu[:M].copy()
    return Te, S
