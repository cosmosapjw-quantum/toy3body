#!/usr/bin/env python
from __future__ import annotations

import os
import json
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu, rankdata

import matplotlib.pyplot as plt

from toy3body_topo.config import PhysicalParams, SimParams, ICParams, LoopParams, OutputParams
from toy3body_topo.dataset import run_one


# Must match scripts/run_basin_map.py
LABEL_TO_INT = {"escape": 0, "exchange": 1, "merger_cut": 2, "timeout": 3, "unknown": 4}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


def _set_thread_env():
    # Avoid oversubscription when using multiple processes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_basin_arrays(basin_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.load(os.path.join(basin_dir, "basin_grid.npy"))
    b_vals = np.load(os.path.join(basin_dir, "b_vals.npy"))
    phi_vals = np.load(os.path.join(basin_dir, "phi_vals.npy"))
    return grid, b_vals, phi_vals


def boundary_mask(grid: np.ndarray, allowed_mask: np.ndarray, mode: str = "allowed_change") -> np.ndarray:
    """Boundary mask based on neighbor label changes.
    mode:
      - 'any_change'     : mark boundary if label changes across neighbors
      - 'allowed_change' : mark boundary only if BOTH cells are allowed AND labels differ
    """
    g = grid
    a = allowed_mask.astype(bool)
    bd = np.zeros_like(a, dtype=bool)

    # phi periodic neighbors
    g_p = np.roll(g, 1, axis=1)
    a_p = np.roll(a, 1, axis=1)
    diff = (g != g_p)
    if mode == "allowed_change":
        diff = diff & a & a_p
    bd |= diff

    g_m = np.roll(g, -1, axis=1)
    a_m = np.roll(a, -1, axis=1)
    diff = (g != g_m)
    if mode == "allowed_change":
        diff = diff & a & a_m
    bd |= diff

    # b axis neighbors (non-periodic)
    diff_b = (g[1:, :] != g[:-1, :])
    if mode == "allowed_change":
        diff_b = diff_b & a[1:, :] & a[:-1, :]
    bd[1:, :] |= diff_b
    bd[:-1, :] |= diff_b

    return bd


def dist_to_boundary_periodic_phi(bd: np.ndarray) -> np.ndarray:
    """Compute distance-to-boundary in pixel units, treating phi as periodic by tiling 3x."""
    nb, nphi = bd.shape
    bd_tile = np.tile(bd, (1, 3))
    # distance_transform_edt: distance from non-zero (foreground) to nearest zero (background). :contentReference[oaicite:2]{index=2}
    dist_tile = distance_transform_edt(~bd_tile)
    return dist_tile[:, nphi:2 * nphi]


def sample_pixels(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("No pixels available to sample from mask.")
    replace = n > len(idx)
    pick = rng.choice(len(idx), size=n, replace=replace)
    return idx[pick]


def jitter_b_phi(
    b0: float,
    phi0: float,
    db_cell: float,
    dphi_cell: float,
    jitter_frac: float,
    rng: np.random.Generator,
    b_min: float,
    b_max: float,
) -> Tuple[float, float, float, float]:
    db = float(rng.uniform(-0.5, 0.5) * db_cell * jitter_frac)
    dphi = float(rng.uniform(-0.5, 0.5) * dphi_cell * jitter_frac)
    b = float(np.clip(b0 + db, b_min, b_max))
    phi = float((phi0 + dphi) % (2 * np.pi))
    return b, phi, db, dphi


def _worker(task):
    seed, payload = task
    row = run_one(
        seed=seed,
        phys=payload["phys"],
        sim=payload["sim"],
        ic=payload["ic"],
        loop_params=payload["loop"],
        output=payload["output"],
    )
    row.update(payload["extra"])
    row["label_int"] = int(LABEL_TO_INT.get(row.get("label", "unknown"), 4))
    row["changed_from_base"] = int(row["label_int"] != row["base_label_int"])
    return row


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan")
    combined = np.concatenate([x, y])
    r = rankdata(combined)  # average ranks for ties
    rx = float(np.sum(r[:nx]))
    U = rx - nx * (nx + 1) / 2.0
    delta = (2.0 * U) / (nx * ny) - 1.0
    return float(delta)


def compare_metric(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) < 15 or len(y) < 15:
        return {"p": float("nan"), "u": float("nan"), "note": "too_few", "nx": int(len(x)), "ny": int(len(y))}
    u, p = mannwhitneyu(x, y, alternative="two-sided")
    return {"p": float(p), "u": float(u), "nx": int(len(x)), "ny": int(len(y)), "cliffs_delta": cliffs_delta(x, y)}


def save_boxplot(x: np.ndarray, y: np.ndarray, labels: Tuple[str, str], title: str, ylabel: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.boxplot([x, y], tick_labels=list(labels), showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_scatter(df: pd.DataFrame, xcol: str, ycol: str, huecol: str, title: str, path: str) -> None:
    if xcol not in df.columns or ycol not in df.columns or huecol not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for key, sub in df.groupby(huecol):
        x = sub[xcol].to_numpy(dtype=float)
        y = sub[ycol].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if np.sum(m) == 0:
            continue
        ax.scatter(x[m], y[m], s=8, alpha=0.35, label=str(key))
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)

def save_hist(df: pd.DataFrame, col: str, huecol: str, title: str, path: str, bins: int = 40) -> None:
    if col not in df.columns or huecol not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for key, sub in df.groupby(huecol):
        x = sub[col].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        ax.hist(x, bins=bins, histtype='step', density=True, label=str(key))
    ax.set_xlabel(col)
    ax.set_ylabel('density')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)



def analyze(csv_path: str, out_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "subset" not in df.columns:
        raise ValueError("CSV missing 'subset' column; did you run boundary_focus_study correctly?")

    analysis_cfg = cfg.get("analysis", {})
    metrics = analysis_cfg.get("metrics", ["t_int","n_loops","link_ok_frac","link_mean_abs","loop_writhe_abs_mean","wind_abs","wind_total","r_min"])
    min_loops = int(analysis_cfg.get("min_loops_for_link_stats", 2))
    min_ok = float(analysis_cfg.get("min_ok_frac_for_link_stats", 0.0))

    def subset_stats(d: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out["n"] = int(len(d))
        if len(d) == 0:
            return out
        out["outcome_frac"] = (d["label"].value_counts(normalize=True).to_dict() if "label" in d else {})
        out["mean_t_end"] = float(d["t_end"].mean()) if "t_end" in d else float("nan")
        out["frac_changed_from_base"] = float(d["changed_from_base"].mean()) if "changed_from_base" in d else float("nan")
        out["frac_with_loops"] = float((d["n_loops"] > 0).mean()) if "n_loops" in d else float("nan")
        # "valid link stats" filter
        if "n_loops" in d and "link_ok_frac" in d:
            dv = d[(d["n_loops"] >= min_loops) & (d["link_ok_frac"] >= min_ok)]
        else:
            dv = d.iloc[0:0]
        out["n_valid_link"] = int(len(dv))
        for m in metrics:
            if m in d.columns:
                out[f"{m}_median"] = float(d[m].median())
                out[f"{m}_mean"] = float(d[m].mean())
            if m in dv.columns:
                out[f"{m}_valid_median"] = float(dv[m].median()) if len(dv) else float("nan")
                out[f"{m}_valid_mean"] = float(dv[m].mean()) if len(dv) else float("nan")
        return out

    db = df[df["subset"] == "boundary"]
    di = df[df["subset"] == "interior"]

    summary: Dict[str, Any] = {
        "paths": {"csv": csv_path},
        "settings": {"min_loops_for_link_stats": min_loops, "min_ok_frac_for_link_stats": min_ok},
        "boundary": subset_stats(db),
        "interior": subset_stats(di),
        "comparisons": {}
    }

    # comparisons on raw metrics and on "valid link" filtered metrics
    for m in metrics:
        if m not in df.columns:
            continue

        xb = db[m].to_numpy()
        xi = di[m].to_numpy()
        summary["comparisons"][m] = {
            "all": compare_metric(xb, xi)
        }

        if ("n_loops" in df.columns) and ("link_ok_frac" in df.columns):
            dbv = db[(db["n_loops"] >= min_loops) & (db["link_ok_frac"] >= min_ok)]
            div = di[(di["n_loops"] >= min_loops) & (di["link_ok_frac"] >= min_ok)]
            summary["comparisons"][m]["valid_link_only"] = compare_metric(dbv[m].to_numpy(), div[m].to_numpy())

    # per base-label stratified comparisons (optional but usually important)
    if "base_label_int" in df.columns:
        allowed = sorted(df["base_label_int"].dropna().unique().astype(int).tolist())
        per_label = {}
        for lab in allowed:
            dfl = df[df["base_label_int"] == lab]
            dbl = dfl[dfl["subset"] == "boundary"]
            dil = dfl[dfl["subset"] == "interior"]
            if len(dbl) < 20 or len(dil) < 20:
                continue
            key = INT_TO_LABEL.get(lab, str(lab))
            per_label[key] = {
                "boundary_n": int(len(dbl)),
                "interior_n": int(len(dil)),
                "comparisons": {}
            }
            for m in metrics:
                if m in dfl.columns:
                    per_label[key]["comparisons"][m] = compare_metric(dbl[m].to_numpy(), dil[m].to_numpy())
        summary["per_base_label"] = per_label

    # figures
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    for m in metrics:
        if m not in df.columns:
            continue
        xb = db[m].to_numpy()
        xi = di[m].to_numpy()
        # If metric is linking-related, also plot only valid-link subset
        if m.startswith("link_") or m in ("n_loops", "loop_mean_gap", "loop_max_dur"):
            if ("n_loops" in df.columns) and ("link_ok_frac" in df.columns):
                dbv = db[(db["n_loops"] >= min_loops) & (db["link_ok_frac"] >= min_ok)]
                div = di[(di["n_loops"] >= min_loops) & (di["link_ok_frac"] >= min_ok)]
                save_boxplot(
                    dbv[m].to_numpy(), div[m].to_numpy(),
                    ("boundary", "interior"),
                    f"{m} (valid link only)",
                    m,
                    os.path.join(fig_dir, f"box_{m}_valid.pdf"),
                )
        save_boxplot(
            xb, xi,
            ("boundary", "interior"),
            f"{m} (all)",
            m,
            os.path.join(fig_dir, f"box_{m}_all.pdf"),
        )

    # extra paper-friendly plots (optional)
    scatter_pairs = analysis_cfg.get("scatter_pairs", [("t_int","wind_abs"), ("t_int","loop_writhe_abs_mean"), ("t_int","n_loops")])
    for (xcol, ycol) in scatter_pairs:
        save_scatter(
            df,
            xcol=xcol,
            ycol=ycol,
            huecol="subset",
            title=f"{ycol} vs {xcol}",
            path=os.path.join(fig_dir, f"scatter_{ycol}_vs_{xcol}.pdf"),
        )

    hist_metrics = analysis_cfg.get("hist_metrics", ["wind_abs", "wind_total", "loop_writhe_abs_mean", "n_loops", "t_int"])
    for col in hist_metrics:
        save_hist(
            df,
            col=col,
            huecol="subset",
            title=f"hist: {col}",
            path=os.path.join(fig_dir, f"hist_{col}.pdf"),
            bins=int(analysis_cfg.get("hist_bins", 40)),
        )

    save_json(summary, os.path.join(out_dir, "summary.json"))
    return summary


def main():
    _set_thread_env()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["run", "analyze", "both"], default="both")
    args = ap.parse_args()

    cfg = load_json(args.config)

    basin_dir = cfg["basin_dir"]
    out_dir = cfg.get("out_dir", "out_boundary_focus")
    os.makedirs(out_dir, exist_ok=True)

    # inherit basin config for consistency
    base_cfg: Dict[str, Any] = {}
    if cfg.get("inherit_basin_config", True):
        basin_cfg_path = os.path.join(basin_dir, "config_used.json")
        if not os.path.exists(basin_cfg_path):
            raise FileNotFoundError(f"Missing basin config_used.json at: {basin_cfg_path}")
        base_cfg = load_json(basin_cfg_path)

    overrides = cfg.get("overrides", {})
    merged = _deep_update(base_cfg, overrides)

    # objects
    if "phys" not in merged:
        raise ValueError("Need 'phys' in basin config or overrides.")
    if "sim" not in merged:
        raise ValueError("Need 'sim' in basin config or overrides.")
    phys = PhysicalParams(**merged["phys"])
    sim = SimParams(**merged["sim"])

    v_inf = float(merged.get("v_inf", cfg.get("v_inf", 1.0)))
    R0 = float(merged.get("R0", cfg.get("R0", 100.0)))
    theta = float(merged.get("theta", cfg.get("theta", 0.0)))
    a0 = float(merged.get("a0", cfg.get("a0", 1.0)))

    loop = LoopParams(**merged.get("loop", cfg.get("loop", {})))

    # output params: force out_dir for raw storage
    out_cfg = merged.get("output", cfg.get("output", {}))
    output = OutputParams(
        out_dir=out_dir,
        store_raw=bool(out_cfg.get("store_raw", False)),
        store_raw_every=int(out_cfg.get("store_raw_every", 0)),
        compress_npz=bool(out_cfg.get("compress_npz", True)),
    )

    # load basin arrays
    grid, b_vals, phi_vals = load_basin_arrays(basin_dir)
    nb, nphi = grid.shape

    # pixel steps for jitter
    if len(b_vals) < 2:
        raise ValueError("b_vals too short")
    db_cell = float(np.median(np.diff(b_vals)))
    dphi_cell = float(2 * np.pi / nphi)

    sampling = cfg.get("sampling", {})
    allowed_labels = sampling.get("allowed_labels", [0, 1])  # default: escape/exchange only
    boundary_mode = sampling.get("boundary_mode", "allowed_change")
    near_radius = int(sampling.get("near_radius_pix", 2))
    interior_min = int(sampling.get("interior_min_radius_pix", 10))
    n_boundary_pixels = int(sampling.get("n_boundary_pixels", 600))
    refine_per_pixel = int(sampling.get("refine_per_pixel", 4))
    n_interior_pixels = int(sampling.get("n_interior_pixels", n_boundary_pixels))
    refine_interior_per_pixel = int(sampling.get("refine_interior_per_pixel", 1))
    jitter_frac = float(sampling.get("jitter_frac", 0.95))
    stratify_interior = bool(sampling.get("stratify_interior_by_base_label", True))

    allowed_mask = np.isin(grid, np.array(allowed_labels, dtype=int))
    bd = boundary_mask(grid, allowed_mask=allowed_mask, mode=boundary_mode)
    dist = dist_to_boundary_periodic_phi(bd)

    near_mask = (dist <= near_radius) & allowed_mask
    interior_mask = (dist >= interior_min) & allowed_mask

    # save masks for debugging/figures
    np.save(os.path.join(out_dir, "boundary_mask.npy"), bd.astype(np.uint8))
    np.save(os.path.join(out_dir, "dist_to_boundary.npy"), dist.astype(np.float32))

    rng = np.random.default_rng(int(cfg.get("seed0", 500000)))

    # pick boundary pixels
    boundary_pix = sample_pixels(near_mask, n_boundary_pixels, rng)

    # pick interior pixels (optionally stratified by base label distribution of boundary_pix)
    if stratify_interior:
        # boundary base-label histogram
        b_labs = grid[boundary_pix[:, 0], boundary_pix[:, 1]].astype(int)
        counts = {lab: int(np.sum(b_labs == lab)) for lab in np.unique(b_labs)}
        interior_list: List[np.ndarray] = []
        for lab, nlab in counts.items():
            mask_lab = interior_mask & (grid == lab)
            if np.sum(mask_lab) == 0:
                # fallback: sample from all interior pixels if none for this label
                mask_lab = interior_mask
            interior_list.append(sample_pixels(mask_lab, nlab, rng))
        interior_pix = np.vstack(interior_list) if interior_list else sample_pixels(interior_mask, n_interior_pixels, rng)
        # if user forced explicit n_interior_pixels, subsample/oversample
        if len(interior_pix) >= n_interior_pixels:
            pick = rng.choice(len(interior_pix), size=n_interior_pixels, replace=False)
            interior_pix = interior_pix[pick]
        else:
            pick = rng.choice(len(interior_pix), size=n_interior_pixels, replace=True)
            interior_pix = interior_pix[pick]
    else:
        interior_pix = sample_pixels(interior_mask, n_interior_pixels, rng)

    out_csv = os.path.join(out_dir, "boundary_runs.csv")
    tmp_csv = out_csv + ".tmp"

    # save config snapshot for reproducibility
    save_json(cfg, os.path.join(out_dir, "config_used.json"))

    if args.mode in ("run", "both"):
        tasks = []
        seed = int(cfg.get("seed0", 500000))

        b_min = float(b_vals[0])
        b_max = float(b_vals[-1])

        def add_tasks_for_subset(subset: str, pix: np.ndarray, nrep: int):
            nonlocal seed
            for (i, j) in pix:
                i = int(i); j = int(j)
                b0 = float(b_vals[i])
                phi0 = float(phi_vals[j])
                base_lab = int(grid[i, j])
                dist_ij = float(dist[i, j])
                for _ in range(nrep):
                    b, phi, db, dphi = jitter_b_phi(
                        b0=b0, phi0=phi0,
                        db_cell=db_cell, dphi_cell=dphi_cell,
                        jitter_frac=jitter_frac, rng=rng,
                        b_min=b_min, b_max=b_max
                    )
                    ic = ICParams(a0=a0, e0=0.0, phase=phi, v_inf=v_inf, b=b, R0=R0, theta=theta)
                    extra = {
                        "subset": subset,
                        "pix_i": i,
                        "pix_j": j,
                        "pix_dist": dist_ij,
                        "base_label_int": base_lab,
                        "base_label": INT_TO_LABEL.get(base_lab, "unknown"),
                        "b0": b0,
                        "phi0": phi0,
                        "db_jitter": db,
                        "dphi_jitter": dphi,
                        "b": b,
                        "phi": phi,
                    }
                    payload = {
                        "seed": seed,
                        "phys": phys,
                        "sim": sim,
                        "ic": ic,
                        "loop": loop,
                        "output": output,
                        "extra": extra,
                    }
                    tasks.append((seed, payload))
                    seed += 1

        add_tasks_for_subset("boundary", boundary_pix, refine_per_pixel)
        add_tasks_for_subset("interior", interior_pix, refine_interior_per_pixel)

        # parallel settings
        par = cfg.get("parallel", {})
        workers = int(par.get("workers", 0)) or (os.cpu_count() or 4)
        chunksize = int(par.get("chunksize", 1))

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

    if args.mode in ("analyze", "both"):
        summary = analyze(out_csv, out_dir, cfg)
        print("Analysis summary saved to:", os.path.join(out_dir, "summary.json"))
        # print a short console summary
        print("Boundary n =", summary["boundary"].get("n"), "Interior n =", summary["interior"].get("n"))
        print("Boundary frac_changed =", summary["boundary"].get("frac_changed_from_base"))
        print("Interior frac_changed =", summary["interior"].get("frac_changed_from_base"))


if __name__ == "__main__":
    main()