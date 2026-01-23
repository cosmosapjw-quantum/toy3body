#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import numpy as np

from toy3body_topo.chaos_metrics import basin_entropy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Path to basin_grid.npy")
    ap.add_argument("--box", type=int, default=8, help="Box size (pixels)")
    ap.add_argument("--n_outcomes", type=int, default=0,
                    help="Number of outcomes. If 0, infer from data (max label + 1).")
    ap.add_argument("--ignore_label", type=int, default=None,
                    help="If set, treat this label as invalid (-1) before computing entropy.")
    args = ap.parse_args()

    grid = np.load(args.grid)

    if args.ignore_label is not None:
        grid = grid.copy()
        grid[grid == int(args.ignore_label)] = -1

    n_out = int(args.n_outcomes)
    if n_out <= 0:
        valid = grid[grid >= 0]
        n_out = int(valid.max() + 1) if valid.size else 1

    out = basin_entropy(grid, box=int(args.box), n_outcomes=n_out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
