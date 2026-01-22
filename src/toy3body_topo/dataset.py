from __future__ import annotations

import os
import json
import numpy as np
from numpy.typing import NDArray
from dataclasses import asdict

from .config import Units, PhysicalParams, SimParams, ICParams, LoopParams, OutputParams
from .ic import make_initial_state
from .integrate import integrate_scatter
from .outcomes import classify_final_pair, outcome_label
from .features import extract_topo_features


def run_one(seed: int,
            phys: PhysicalParams,
            sim: SimParams,
            ic: ICParams,
            loop_params: LoopParams,
            output: OutputParams) -> dict:
    """Run a single scattering experiment and return a flat dict for tabular storage."""
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

    topo, loops, Linfo = extract_topo_features(res["T"], res["Y"], phys, sim, loop_params)

    row = {
        "seed": int(seed),
        "label": label,
        "merged": bool(res["merged"]),
        "timeout": bool(res["timeout"]),
        "pn_warning": bool(res["pn_warning"]),
        "E0": float(res["E0"]),
        "E_end": float(res["E_end"]),
        "dE": float(res["E_end"] - res["E0"]),
        "runtime_sec": float(res["runtime_sec"]),
        # final pair info
        "binary_pair": str(final_info.get("binary_pair")),
        "escaper": final_info.get("escaper"),
        "a_final": final_info.get("a"),
        "e_final": final_info.get("e"),
        "E_pair": final_info.get("E_pair"),
    }
    row.update(topo)

    # parameters (flatten)
    for k, v in asdict(phys).items():
        row[f"phys_{k}"] = v
    for k, v in asdict(sim).items():
        row[f"sim_{k}"] = v
    for k, v in asdict(ic).items():
        row[f"ic_{k}"] = v
    for k, v in asdict(loop_params).items():
        row[f"loop_{k}"] = v

    # raw storage (debugging)
    if output.store_raw and output.store_raw_every and (seed % output.store_raw_every == 0):
        raw_dir = os.path.join(output.out_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        fname = os.path.join(raw_dir, f"run_seed{seed}.npz")
        np.savez_compressed(fname, T=res["T"], Y=res["Y"])

        # store loops and linking matrix if present
        if len(loops) > 0:
            curves = np.stack([L.curve for L in loops], axis=0)  # (K,N+1,3)
            np.savez_compressed(os.path.join(raw_dir, f"loops_seed{seed}.npz"),
                                curves=curves,
                                L=Linfo["L"] if Linfo["L"] is not None else np.array([]))

    return row
