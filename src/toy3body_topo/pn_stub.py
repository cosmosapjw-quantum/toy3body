"""PN scaffolding (placeholder).

This project intentionally DOES NOT implement post-Newtonian equations of motion,
because doing it correctly (especially for 3 bodies + spins) is easy to get wrong.

However, the rest of the pipeline (shape space -> loops -> linking fingerprints)
is agnostic to the dynamics engine. You can plug a PN/GR engine by supplying
a `rhs_fun(t, y, units, phys)` callable to `integrate_scatter(...)`.

If you want a fast and reliable PN path:
- consider using a few-body integrator that already supports PN terms, or
- use REBOUNDx (for N-body with extra forces) / custom AR-CHAIN variants.

When you implement PN here, prefer:
- unit tests on known two-body limits (perihelion precession, GW inspiral rate),
- tight energy/angular-momentum budget checks,
- cross-check against another code for a subset.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .config import Units, PhysicalParams


def rhs_pn_placeholder(t: float, y: NDArray[np.float64], units: Units, phys: PhysicalParams) -> NDArray[np.float64]:
    raise NotImplementedError("PN RHS not implemented in this toy project.")
