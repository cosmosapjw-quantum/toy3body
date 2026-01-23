#!/usr/bin/env python
from __future__ import annotations

import os
import json
import argparse
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from toy3body_topo.config import PhysicalParams, SimParams, LoopParams, Units, phys_from_mu_nu, ICParams
from toy3body_topo.dataset import run_one
from toy3body_topo.config import OutputParams


LABEL_TO_INT = {"escape": 0, "exchange": 1, "merger_cut": 2, "timeout": 3, "unknown": 4, "failed": 5}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def one_task(task):
    rid, seed, mu, nu, eps, v_inf, b, phase, theta, R0, sim_dict, loop_dict = task
    phys = phys_from_mu_nu(mu=mu, nu=nu, epsilon=eps)
    sim = SimParams(**sim_dict)
    loop = LoopParams(**loop_dict)
    ic = ICParams(a0=1.0, e0=0.0, phase=float(phase), v_inf=float(v_inf), b=float(b), R0=float(R0), theta=float(theta))
    out = run_one(seed, phys, sim, ic, loop, OutputParams(out_dir="", store_raw=False))
    out["rid"] = rid
    out["mu"] = mu
    out["nu"] = nu
    out["epsilon"] = eps
    out["v_inf"] = v_inf
    out["b"] = b
    out["phase"] = phase
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_csv = cfg.get("out_csv", "scan_regimes.csv")

    sim_dict = cfg["sim"]
    loop_dict = cfg.get("loop", {})

    v_list = list(cfg.get("v_inf_list", [0.5]))
    nu_list = list(cfg.get("nu_list", [1.0]))
    eps_list = list(cfg.get("epsilon_list", [0.0]))
    b0, b1 = cfg.get("b_range", [0.0, 6.0])
    theta = float(cfg.get("theta", 0.0))
    R0 = float(cfg.get("R0", 100.0))
    n_samples = int(cfg.get("n_samples", 200))
    seed0 = int(cfg.get("seed0", 50000))

    workers = int(cfg.get("workers", 0)) or (os.cpu_count() or 4)

    tasks = []
    rid = 0
    seed = seed0
    rng = np.random.default_rng(seed0)

    for v_inf in v_list:
        for nu in nu_list:
            for eps in eps_list:
                mu = 1.0
                for _ in range(n_samples):
                    b = float(rng.uniform(b0, b1))
                    phase = float(rng.uniform(0.0, 2*np.pi))
                    tasks.append((rid, seed, mu, float(nu), float(eps), float(v_inf), b, phase, theta, R0, sim_dict, loop_dict))
                    seed += 1
                rid += 1

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_task, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            try:
                rows.append(fut.result())
            except Exception as e:
                # Keep going: regime scans should not die from a single bad IC.
                rows.append({"label": "failed", "error": repr(e)})

    df = pd.DataFrame(rows)
    df["label_int"] = df["label"].map(LABEL_TO_INT).fillna(LABEL_TO_INT["failed"]).astype(int)

    # aggregate per regime id
    def agg(group: pd.DataFrame) -> dict:
        out = {}
        out["n"] = int(len(group))
        out["escape_frac"] = float(np.mean(group["label_int"] == 0))
        out["exchange_frac"] = float(np.mean(group["label_int"] == 1))
        out["merger_frac"] = float(np.mean(group["label_int"] == 2))
        out["timeout_frac"] = float(np.mean(group["label_int"] == 3))
        out["t_end_median"] = float(np.nanmedian(group.get("t_end", np.nan)))
        out["t_int_median"] = float(np.nanmedian(group.get("t_int", np.nan)))
        out["n_loops_mean"] = float(np.nanmean(group.get("n_loops", np.nan)))
        out["wind_abs_mean"] = float(np.nanmean(group.get("wind_abs", np.nan)))
        out["link_ok_frac_mean"] = float(np.nanmean(group.get("link_ok_frac", np.nan)))
        return out

    g = df.groupby("rid")
    agg_rows = []
    for rid, group in g:
        # recover regime params from the first row
        first = group.iloc[0]
        rec = {
            "rid": int(rid),
            "mu": float(first.get("mu", 1.0)),
            "nu": float(first.get("nu", np.nan)),
            "epsilon": float(first.get("epsilon", np.nan)),
            "v_inf": float(first.get("v_inf", np.nan)),
        }
        rec.update(agg(group))
        agg_rows.append(rec)

    out_df = pd.DataFrame(agg_rows).sort_values(["exchange_frac","timeout_frac","n_loops_mean"], ascending=False)
    out_df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(out_df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
