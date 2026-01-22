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


def _worker(task):
    seed, phys, sim, ic = task
    out = run_outcome_only(seed=seed, phys=phys, sim=sim, ic=ic)
    return LABEL_TO_INT.get(out["label"], 4)


def main():
    _set_thread_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="JSON with phys/sim and base (b,phase)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    phys = PhysicalParams(**cfg["phys"])
    sim = SimParams(**cfg["sim"])

    base_b = float(cfg["base_b"])
    base_phi = float(cfg["base_phi"])
    v_inf = float(cfg.get("v_inf", 1.0))
    R0 = float(cfg.get("R0", 100.0))
    theta = float(cfg.get("theta", 0.0))

    deltas = np.array(cfg.get("deltas", [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]), dtype=float)
    n_pairs = int(cfg.get("n_pairs", 200))
    seed0 = int(cfg.get("seed0", 50000))

    workers = int(cfg.get("workers", 0)) or (os.cpu_count() or 4)
    chunksize = int(cfg.get("chunksize", 16))

    results = []
    seed = seed0
    for d in deltas:
        # build tasks for pairs
        tasks = []
        for k in range(n_pairs):
            # alternate perturb direction
            if k % 2 == 0:
                b1, phi1 = base_b, base_phi
                b2, phi2 = base_b + float(d), base_phi
            else:
                b1, phi1 = base_b, base_phi
                b2, phi2 = base_b, base_phi + float(d)

            ic1 = ICParams(a0=1.0, e0=0.0, phase=float(phi1), v_inf=v_inf, b=float(b1), R0=R0, theta=theta)
            ic2 = ICParams(a0=1.0, e0=0.0, phase=float(phi2), v_inf=v_inf, b=float(b2), R0=R0, theta=theta)
            tasks.append((seed, phys, sim, ic1)); seed += 1
            tasks.append((seed, phys, sim, ic2)); seed += 1

        with ProcessPoolExecutor(max_workers=workers) as ex:
            labs = list(tqdm(ex.map(_worker, tasks, chunksize=chunksize), total=len(tasks)))

        # compare pairs
        diff = 0
        for k in range(n_pairs):
            if labs[2*k] != labs[2*k+1]:
                diff += 1
        f_unc = diff / n_pairs
        results.append((float(d), float(f_unc)))
        print("delta", d, "uncertain fraction", f_unc)

    arr = np.array(results, dtype=float)
    out_path = cfg.get("out_path", "uncertainty.npy")
    np.save(out_path, arr)
    print("saved:", out_path)

    # fit log f ~ alpha log d where f in (0,1)
    mask = (arr[:,1] > 0) & (arr[:,1] < 1)
    if np.sum(mask) >= 3:
        x = np.log(arr[mask,0])
        y = np.log(arr[mask,1])
        A = np.vstack([x, np.ones_like(x)]).T
        alpha, c0 = np.linalg.lstsq(A, y, rcond=None)[0]
        print("uncertainty exponent alpha ~", float(alpha))


if __name__ == "__main__":
    main()
