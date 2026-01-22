#!/usr/bin/env python
from __future__ import annotations

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm import tqdm

from toy3body_topo.config import PhysicalParams, SimParams, ICParams
from toy3body_topo.basin import run_outcome_only


LABEL_TO_INT = {"escape": 0, "exchange": 1, "merger_cut": 2, "timeout": 3, "unknown": 4}


def _set_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _worker(task):
    (i, j, seed, phys, sim, ic) = task
    out = run_outcome_only(seed=seed, phys=phys, sim=sim, ic=ic)
    return (i, j, LABEL_TO_INT.get(out["label"], 4), out["t_end"])


def main():
    _set_thread_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg.get("out_dir", "out_basin")
    os.makedirs(out_dir, exist_ok=True)

    # build objects
    phys = PhysicalParams(**cfg["phys"])
    sim = SimParams(**cfg["sim"])

    v_inf = float(cfg["v_inf"])
    R0 = float(cfg.get("R0", 100.0))
    theta = float(cfg.get("theta", 0.0))

    nb = int(cfg["nb"])
    nphi = int(cfg["nphi"])
    b0, b1 = cfg["b_range"]
    b_vals = np.linspace(float(b0), float(b1), nb)
    phi_vals = np.linspace(0.0, 2*np.pi, nphi, endpoint=False)

    seed0 = int(cfg.get("seed0", 20000))
    tasks = []
    seed = seed0
    for i, b in enumerate(b_vals):
        for j, phi in enumerate(phi_vals):
            ic = ICParams(a0=1.0, e0=0.0, phase=float(phi), v_inf=v_inf, b=float(b), R0=R0, theta=theta)
            tasks.append((i, j, seed, phys, sim, ic))
            seed += 1

    workers = int(cfg.get("workers", 0)) or (os.cpu_count() or 4)
    chunksize = int(cfg.get("chunksize", 8))

    grid = np.empty((nb, nphi), dtype=np.int64)
    tend = np.empty((nb, nphi), dtype=float)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = ex.map(_worker, tasks, chunksize=chunksize)
        for (i, j, lab, t_end) in tqdm(it, total=len(tasks)):
            grid[i, j] = lab
            tend[i, j] = t_end

    np.save(os.path.join(out_dir, "basin_grid.npy"), grid)
    np.save(os.path.join(out_dir, "basin_tend.npy"), tend)
    np.save(os.path.join(out_dir, "b_vals.npy"), b_vals)
    np.save(os.path.join(out_dir, "phi_vals.npy"), phi_vals)

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("Saved basin outputs to:", out_dir)


if __name__ == "__main__":
    main()
