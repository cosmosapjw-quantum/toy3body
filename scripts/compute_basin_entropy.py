#!/usr/bin/env python
from __future__ import annotations

import argparse
import numpy as np
from toy3body_topo.chaos_metrics import basin_entropy_boxed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, help="Path to basin_grid.npy")
    ap.add_argument("--box", type=int, default=8)
    ap.add_argument("--n_outcomes", type=int, default=4)
    args = ap.parse_args()

    grid = np.load(args.grid)
    out = basin_entropy_boxed(grid, n_outcomes=args.n_outcomes, box=args.box)
    print(out)


if __name__ == "__main__":
    main()
