from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import replace
from .config import Units, PhysicalParams, ICParams


def rot2(theta: float) -> NDArray[np.float64]:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]], dtype=np.float64)


def make_initial_state(units: Units, phys: PhysicalParams, ic: ICParams) -> NDArray[np.float64]:
    """Planar binary-single IC in barycentric coordinates.

    State ordering:
      y = [x1,y1,x2,y2,x3,y3, vx1,vy1,vx2,vy2,vx3,vy3]
    """
    G = units.G
    m1, m2, m3 = phys.m1, phys.m2, phys.m3
    Mbin = m1 + m2

    # Binary relative vectors (toy: circular)
    r_rel = np.array([ic.a0, 0.0], dtype=np.float64)
    v_rel = np.array([0.0, np.sqrt(G * Mbin / ic.a0)], dtype=np.float64)

    # distribute about binary COM
    r1 = -(m2/Mbin)*r_rel
    r2 = +(m1/Mbin)*r_rel
    v1 = -(m2/Mbin)*v_rel
    v2 = +(m1/Mbin)*v_rel

    R = rot2(ic.phase + ic.theta)
    r1 = R @ r1; r2 = R @ r2
    v1 = R @ v1; v2 = R @ v2

    # intruder initial position, then focusing-corrected velocity
    r3 = np.array([-ic.R0, ic.b], dtype=np.float64)
    R0 = float(np.linalg.norm(r3))
    v_inf = ic.v_inf

    v0 = np.sqrt(v_inf*v_inf + 2.0*G*Mbin/R0)
    h = ic.b * v_inf
    v_t = h / R0
    v_r2 = max(v0*v0 - v_t*v_t, 0.0)
    v_r = -np.sqrt(v_r2)

    r_hat = r3 / R0
    t_hat = np.array([-r_hat[1], r_hat[0]], dtype=np.float64)
    v3 = v_r*r_hat + v_t*t_hat

    # rotate intruder by same global rotation
    r3 = R @ r3
    v3 = R @ v3

    # shift to total COM and zero total momentum
    Mtot = Mbin + m3
    r_com = (m1*r1 + m2*r2 + m3*r3) / Mtot
    v_com = (m1*v1 + m2*v2 + m3*v3) / Mtot
    r1 -= r_com; r2 -= r_com; r3 -= r_com
    v1 -= v_com; v2 -= v_com; v3 -= v_com

    y0 = np.hstack([r1, r2, r3, v1, v2, v3]).astype(np.float64)
    return y0


def sample_ic(rng: np.random.Generator,
              a0: float,
              v_inf: float,
              b_max: float,
              R0: float,
              random_phase: bool = True,
              random_theta: bool = True) -> ICParams:
    # b^2 uniform
    b = np.sqrt(rng.random()) * b_max
    phase = float(rng.uniform(0.0, 2*np.pi)) if random_phase else 0.0
    theta = float(rng.uniform(0.0, 2*np.pi)) if random_theta else 0.0
    return ICParams(a0=a0, e0=0.0, phase=phase, v_inf=v_inf, b=float(b), R0=R0, theta=theta)
