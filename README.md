# toy3body-topo (shape-space topological fingerprints)

This is a **research-oriented toy pipeline** for:
- planar 3-body **binary–single scattering** (escape vs exchange vs merger-cut),
- mapping outcomes over initial-condition grids (exit basins),
- transient-chaos summaries (dwell-time tails / escape-rate fit),
- **shape space** time series: `S(t) = (log r12, log r23, log r31)`,
- extracting **nearly-closed loops** from close returns in shape space,
- computing **Gauss linking numbers** between extracted loops (as a topological fingerprint).

It is intentionally designed so that you can later swap the dynamics engine
(e.g. REBOUND/IAS15 + PN/2.5PN) while reusing the **same** shape/loop/linking pipeline.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run a small ensemble:
```bash
python scripts/run_ensemble.py --config configs/ensemble_small.json
```

Run an exit-basin grid (b vs phase):
```bash
python scripts/run_basin_map.py --config configs/basin_small.json
```

Compute basin entropy by boxing:
```bash
python scripts/compute_basin_entropy.py --grid out_basin/basin_grid.npy --box 8 --n_outcomes 4
```

## Output

Ensemble run writes:
- `out_runs/runs.csv` : one-row-per-run table with physics + topology/chaos features
- (optional) `out_runs/raw/` : selected raw trajectories for debugging

Basin run writes:
- `out_basin/basin_grid.npy`, `b_vals.npy`, `phi_vals.npy`

## Notes on "research readiness"

- Integration uses `scipy.integrate.solve_ivp` (default: `DOP853`), chunked with early stopping.
- For very strong close encounters / extreme mass ratios / PN physics, you should
  **cross-check** a subset using a few-body integrator (e.g. REBOUND IAS15 or AR-CHAIN).
- Linking numbers are computed via a discretized Gauss integral on polygonal loops,
  with quality gates (minimum separation + closeness-to-integer).
  This is a practical engineering choice for large sweeps.

## References (background)

- SciPy `solve_ivp` docs (methods, events).
- Gauss linking integral / linking number definition.
- (Future upgrade) Rein & Spiegel (2015) IAS15 integrator for close encounters and high-eccentricity orbits.
