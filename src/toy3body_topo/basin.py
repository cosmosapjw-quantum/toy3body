from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from .config import Units, PhysicalParams, SimParams, ICParams
from .ic import make_initial_state
from .integrate import integrate_scatter
from .outcomes import classify_final_pair, outcome_label


def run_outcome_only(seed: int,
                     phys: PhysicalParams,
                     sim: SimParams,
                     ic: ICParams) -> dict:
    units = Units()
    y0 = make_initial_state(units, phys, ic)
    res = integrate_scatter(units, phys, sim, y0)

    final_info = classify_final_pair(units, phys, res["Y"][-1])
    if res["merged"]:
        label = "merger_cut"
    elif res["timeout"]:
        label = "timeout"
    else:
        label = outcome_label(final_info["binary_pair"], initial_binary=(0,1))

    return {
        "seed": int(seed),
        "label": label,
        "t_end": float(res["t_end"]),
        "merged": bool(res["merged"]),
        "timeout": bool(res["timeout"]),
        "pn_warning": bool(res["pn_warning"]),
    }
