# toy3body_topo

A small research playground for **planar Newtonian binary–single scattering** with a focus on:

- **chaotic scattering diagnostics**: basin maps, basin entropy, uncertainty exponent
- **topological / geometric proxies**: close-return loops in an embedding, Gauss linking, **writhe**, and a very robust physical quantity: **intruder winding around the binary COM**

This is *not* a production-grade N-body package. The goal is fast iteration on “does a topological signal show up at all?” questions.

---

## What changed in this upgrade

If your old plots looked “flat” (e.g., linking values almost always 0), that was usually because:

1) **Your regime is too simple** (e.g., light intruder + high v_inf → almost always trivial escape)  
2) **You only have ~0–1 loops per trajectory** in the window you analyze, so pairwise-linking statistics are not even defined  
3) In scattering, “shape recurrence” often happens with a changing overall scale; using raw `(log r12, log r23, log r31)` can hide returns

So this upgrade adds:

- A better default embedding for loop finding: **`shape_mode="uvrho"`** (shape `(u,v)` + scale `rho`)  
- Extra loop extraction controls: `max_pairs_per_i`, `max_overlap_frac`
- A single-loop topological/geometric proxy: **Gauss writhe** (`loop_writhe_*`)
- A physically interpretable, very strong signal for chaotic/resonant scattering:  
  **intruder winding around the binary COM** (`wind_*`)
- Boundary-focus script now also emits **scatter + histogram PDFs** (not only boxplots)
- More experiment configs (equal-mass, lower v_inf, etc.) and a quick regime scanner

---

## Install

Create a venv and install editable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Dependencies are minimal: numpy, scipy, pandas, tqdm, matplotlib.

---

## Quick start: generate a basin map

Example (richer regime than the tiny demo):

```bash
python scripts/run_basin_map.py --config configs/basin_equalmass_v05.json
python scripts/compute_basin_entropy.py --grid out_basin_eq_v05/basin_grid.npy --box 8 --n_outcomes 0
python scripts/plot_basin_figures.py --basin_dir out_basin_eq_v05
```

Outputs:
- `basin_grid.npy`: integer label per (b, phase)
- `basin_tend.npy`: stopping time
- `config_used.json`: exactly what was used

---

## Boundary focus: boundary vs interior reruns (with topology proxies)

Once you have a basin directory, run the boundary-focus study:

```bash
python scripts/boundary_focus_study.py --config configs/boundary_focus_escape_exchange_eq_v05.json
```

Outputs:
- `out_boundary_eq_v05_escape_exchange/boundary_runs.csv`
- `out_boundary_eq_v05_escape_exchange/summary.json`
- `out_boundary_eq_v05_escape_exchange/figs/*.pdf` (boxplots + hist + scatter)

---

## If your basin entropy is ~0: what to change

Basin entropy needs **more than one outcome** in the grid. If everything is `escape`, entropy will be 0.

The easiest way to “make it interesting”:

- use **equal-ish masses** (exchange becomes common)
- decrease **v_inf** (more resonant interactions)
- widen **b_range**
- increase `sim.t_max` (avoid prematurely tagging long-lived resonances as timeout)
- if runtime explodes, cap with `sim.max_walltime_sec` and/or `run_basin_map.max_runtime_sec`

Recommended starting points:
- `configs/basin_equalmass_v05.json`
- `configs/basin_equalmass_v03.json` (slower, more resonant)

---

## Topology / geometry features (what they mean)

Per trajectory, we compute:

- `n_loops`  
  number of close-return loops found in an embedding

- `link_ok_frac` / `link_mean_abs` / `link_nz_frac`  
  pairwise Gauss linking stats among extracted loops (requires at least 2 loops)

- `loop_writhe_*`  
  Gauss writhe per loop (works even when there is only 1 loop).  
  Writhe tends to grow when the curve coils strongly in 3D embedding space.

- `wind_raw`, `wind_abs`, `wind_total`  
  intruder winding around the binary COM during the interaction window.  
  This is usually a *very* clean separator between “trivial flyby escape” and “resonant chaotic scattering”.

---

## Choosing the embedding for loops

In `LoopParams`:

- `shape_mode="uvrho"` (default)  
  Uses a symmetric decomposition of log-distances into:
  - (u,v): shape (scale removed)
  - rho: overall scale  
  Then optionally downweights rho in NN search via `nn_rho_weight`.

- `shape_mode="logdist"`  
  Raw (log r12, log r23, log r31). Good for debugging, but can miss returns.

- `shape_mode="delay_d3"`  
  Delay embedding (Takens) of `log d3(t)` into R^3.  
  Sometimes produces clearer “attractor-like” geometry for linking/writhe.

Tuning tips:
- Increase `eps_close`, `k_nn`, `max_pairs_per_i` to find more loops  
- Don’t set `nn_rho_weight` to 0.0 unless you *want* a nearly planar curve (linking will collapse)

---

## Quick regime scan

To find “interesting” parameters quickly:

```bash
python scripts/scan_regimes.py --config configs/scan_regimes_small.json
```

This outputs a CSV ranking regimes by exchange/timeout/loop activity.

---

## Notes / caveats (read this if results look weird)

- This is a toy solver around SciPy `solve_ivp`. Extremely close encounters can be stiff and expensive.
  Use the walltime caps (`sim.max_walltime_sec`, `max_runtime_sec`) for basin sweeps.

- “Topology of chaos” (templates + linking of UPOs) is cleanest for *bounded* strange attractors.
  Chaotic scattering is a transient-chaos situation; you should expect noisier signals.
  That’s why the code also exposes writhe and winding, which behave better in practice.

---

## References / background pointers

- Basin entropy (chaotic scattering uncertainty):
  - Daza et al., *Basin entropy: a new tool to analyze uncertainty in dynamical systems* (2016), arXiv:1601.01416
    https://arxiv.org/abs/1601.01416

- Topology of chaos + linking of periodic orbits:
  - Gilmore & Lefranc, *The Topology of Chaos* (book)
    https://www.worldscientific.com/worldscibooks/10.1142/3973

- Shape sphere / shape coordinates for planar 3-body:
  - Montgomery, *The N-body problem, the braid group, and action-minimizing periodic solutions* (survey / notes)
    https://www.ams.org/notices/200106/fea-montgomery.pdf
