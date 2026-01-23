#!/usr/bin/env python
from __future__ import annotations

import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

from toy3body_topo.config import PhysicalParams, SimParams, ICParams
from toy3body_topo.basin import run_outcome_only

# 4 outcomes를 쓰고 싶으면 unknown/failed를 timeout(3)로 합칠 수도 있는데,
# 디버깅/품질관리 위해 일단 5번(unknown)으로 분리해두는 걸 추천.
LABEL_TO_INT = {
    "escape": 0,
    "exchange": 1,
    "merger_cut": 2,
    "timeout": 3,
    "unknown": 4,
    "failed": 4,
}

def _set_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _chunked(seq, n):
    for k in range(0, len(seq), n):
        yield seq[k:k+n]

def _worker_one(task):
    """
    Return: (i, j, lab_int, t_end, lab_str, err_str)
    """

    if len(task) != 7:
        raise ValueError(f"Bad task tuple length: expected 7, got {len(task)}; task={task[:3]}...")

    i, j, seed, phys, sim, ic, max_runtime_sec = task

    out = run_outcome_only(seed=seed, phys=phys, sim=sim, ic=ic, max_runtime_sec=max_runtime_sec)
    lab_str = out.get("label", "unknown")
    lab_int = LABEL_TO_INT.get(lab_str, LABEL_TO_INT["unknown"])
    t_end = float(out.get("t_end", np.nan))
    return (i, j, lab_int, t_end, lab_str, "")

def _worker_chunk(chunk):
    """
    Run a chunk in one process.
    Any numerical/physics exception is captured and returned as failed cell(s).
    Programming bugs (like bad task tuple length) will still raise (fail-fast).
    """
    out = []
    for task in chunk:
        if len(task) != 7:
            # 버그는 즉시 올려서 전체가 멈추게 하는 게 맞음
            raise ValueError(f"Bad task tuple length in chunk: expected 7, got {len(task)}")

        try:
            out.append(_worker_one(task))
        except Exception as e:
            i, j = int(task[0]), int(task[1])
            err = f"{type(e).__name__}: {e}"
            # 너무 긴 에러는 잘라서 IPC 부담 줄이기
            if len(err) > 300:
                err = err[:300] + "..."
            out.append((i, j, LABEL_TO_INT["failed"], float("nan"), "failed", err))
    return out

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
    max_runtime_sec = float(cfg.get("max_runtime_sec", 0.0) or 0.0)

    tasks = []
    seed = seed0
    for i, b in enumerate(b_vals):
        for j, phi in enumerate(phi_vals):
            ic = ICParams(a0=1.0, e0=0.0, phase=float(phi),
                          v_inf=v_inf, b=float(b), R0=R0, theta=theta)
            tasks.append((i, j, seed, phys, sim, ic, max_runtime_sec))
            seed += 1

    workers = int(cfg.get("workers", 0)) or (os.cpu_count() or 4)
    chunksize = int(cfg.get("chunksize", 8))

    grid = np.full((nb, nphi), LABEL_TO_INT["unknown"], dtype=np.int64)
    tend = np.full((nb, nphi), np.nan, dtype=float)

    task_chunks = list(_chunked(tasks, chunksize))

    err_path = os.path.join(out_dir, "errors.jsonl")
    n_failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_worker_chunk, ch) for ch in task_chunks]

        with open(err_path, "w", encoding="utf-8") as ef, tqdm(total=len(tasks)) as pbar:
            for fut in as_completed(futures):
                results = fut.result()  # list[(i,j,lab_int,t_end,lab_str,err_str)]
                for (i, j, lab_int, t_end, lab_str, err_str) in results:
                    grid[i, j] = int(lab_int)
                    tend[i, j] = float(t_end)

                    if err_str:
                        n_failed += 1
                        ef.write(json.dumps({
                            "i": int(i), "j": int(j),
                            "b": float(b_vals[int(i)]),
                            "phi": float(phi_vals[int(j)]),
                            "seed": int(seed0 + int(i)*nphi + int(j)),
                            "label": str(lab_str),
                            "error": str(err_str),
                        }) + "\n")

                pbar.update(len(results))

    np.save(os.path.join(out_dir, "basin_grid.npy"), grid)
    np.save(os.path.join(out_dir, "basin_tend.npy"), tend)
    np.save(os.path.join(out_dir, "b_vals.npy"), b_vals)
    np.save(os.path.join(out_dir, "phi_vals.npy"), phi_vals)

    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # quick sanity report
    u, c = np.unique(grid, return_counts=True)
    print("Saved basin outputs to:", out_dir)
    print("Label counts:", {int(uu): int(cc) for uu, cc in zip(u, c)})
    print("Failed cells logged:", n_failed, "->", err_path)

if __name__ == "__main__":
    main()
