from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .config import Units, PhysicalParams


def unpack(y: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r = y[:6].reshape(3, 2)
    v = y[6:].reshape(3, 2)
    return r, v


def rhs_newton(t: float, y: NDArray[np.float64], units: Units, phys: PhysicalParams) -> NDArray[np.float64]:
    """Newtonian 3-body RHS (planar). Designed to minimize allocations."""
    G = units.G
    m1, m2, m3 = phys.m1, phys.m2, phys.m3

    # positions/velocities
    x1, y1, x2, y2, x3, y3 = y[0], y[1], y[2], y[3], y[4], y[5]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6], y[7], y[8], y[9], y[10], y[11]

    # pair vectors
    dx12 = x2 - x1; dy12 = y2 - y1
    dx13 = x3 - x1; dy13 = y3 - y1
    dx23 = x3 - x2; dy23 = y3 - y2

    r12_2 = dx12*dx12 + dy12*dy12
    r13_2 = dx13*dx13 + dy13*dy13
    r23_2 = dx23*dx23 + dy23*dy23

    inv12 = r12_2**(-1.5)
    inv13 = r13_2**(-1.5)
    inv23 = r23_2**(-1.5)

    a1x = G*(m2*dx12*inv12 + m3*dx13*inv13)
    a1y = G*(m2*dy12*inv12 + m3*dy13*inv13)

    a2x = G*(m1*(-dx12)*inv12 + m3*dx23*inv23)
    a2y = G*(m1*(-dy12)*inv12 + m3*dy23*inv23)

    a3x = G*(m1*(-dx13)*inv13 + m2*(-dx23)*inv23)
    a3y = G*(m1*(-dy13)*inv13 + m2*(-dy23)*inv23)

    # pack dy/dt
    out = np.empty_like(y)
    out[0] = vx1; out[1] = vy1
    out[2] = vx2; out[3] = vy2
    out[4] = vx3; out[5] = vy3
    out[6] = a1x; out[7] = a1y
    out[8] = a2x; out[9] = a2y
    out[10] = a3x; out[11] = a3y
    return out


def min_pair_distance(y: NDArray[np.float64]) -> float:
    x1, y1, x2, y2, x3, y3 = y[0], y[1], y[2], y[3], y[4], y[5]
    d12 = (x2-x1)**2 + (y2-y1)**2
    d13 = (x3-x1)**2 + (y3-y1)**2
    d23 = (x3-x2)**2 + (y3-y2)**2
    return float(np.sqrt(min(d12, d13, d23)))


def energy_newton(y: NDArray[np.float64], units: Units, phys: PhysicalParams) -> float:
    G = units.G
    m1, m2, m3 = phys.m1, phys.m2, phys.m3

    x1, y1, x2, y2, x3, y3 = y[0], y[1], y[2], y[3], y[4], y[5]
    vx1, vy1, vx2, vy2, vx3, vy3 = y[6], y[7], y[8], y[9], y[10], y[11]

    KE = 0.5*(m1*(vx1*vx1+vy1*vy1) + m2*(vx2*vx2+vy2*vy2) + m3*(vx3*vx3+vy3*vy3))

    r12 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    r13 = np.sqrt((x3-x1)**2 + (y3-y1)**2)
    r23 = np.sqrt((x3-x2)**2 + (y3-y2)**2)

    PE = -G*(m1*m2/r12 + m1*m3/r13 + m2*m3/r23)
    return float(KE + PE)
