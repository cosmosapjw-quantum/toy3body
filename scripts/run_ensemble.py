#!/usr/bin/env python
from __future__ import annotations

import os
import csv
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from toy3body_topo.config import load_ensemble_config, phys_from_mu_nu
from toy3body_topo.ic import sample_ic
from toy3body_topo.dataset import run_one


def _set_thread_env():
    # Avoid oversubscription when also using multiple processes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def build_tasks(cfg) -> list[tuple[int, Dict[str, Any]]]:
    tasks = []
    seed = int(cfg.seed0)
    for mu in cfg.mu_list:
        for nu in cfg.nu_list:
            for eps in cfg.epsilon_list:
                phys = phys_from_mu_nu(float(mu), float(nu), float(eps))
                for v_inf in cfg.v_inf_list:
                    rng = np.random.default_rng(seed + 99991)
                    for _ in range(cfg.n_per_setting):
                        ic = sample_ic(rng, a0=float(cfg.a0), v_inf=float(v_inf), b_max=float(cfg.b_max), R0=float(cfg.R0),
                                       random_phase=bool(cfg.random_phase), random_theta=bool(cfg.random_theta))
                        payload = {
                            "seed": seed,
                            "phys": phys,
                            "sim": cfg.sim,
                            "ic": ic,
                            "loop": cfg.loop,
                            "output": cfg.output,
                        }
                        tasks.append((seed, payload))
                        seed += 1
    return tasks


def _worker(task: tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
    seed, p = task
    return run_one(seed=seed, phys=p["phys"], sim=p["sim"], ic=p["ic"], loop_params=p["loop"], output=p["output"])


def main():
    _set_thread_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to ensemble JSON config")
    args = ap.parse_args()

    cfg = load_ensemble_config(args.config)
    out_dir = cfg.output.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # save config snapshot
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_raw = json.load(f)
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_raw, f, indent=2)

    tasks = build_tasks(cfg)

    workers = cfg.parallel.workers or (os.cpu_count() or 4)
    chunksize = max(1, int(cfg.parallel.chunksize))

    out_csv = os.path.join(out_dir, "runs.csv")
    tmp_csv = out_csv + ".tmp"

    with ProcessPoolExecutor(max_workers=workers) as ex:
        it = ex.map(_worker, tasks, chunksize=chunksize)

        first = True
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = None
            for row in tqdm(it, total=len(tasks)):
                if first:
                    fieldnames = list(row.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    first = False
                assert writer is not None
                writer.writerow(row)

    os.replace(tmp_csv, out_csv)
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
