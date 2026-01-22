#!/usr/bin/env python
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from toy3body_topo.chaos_metrics import fit_escape_rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", required=True)
    ap.add_argument("--tmin", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(args.runs_csv)
    print("n runs:", len(df))
    print(df["label"].value_counts())

    times = df["t_end"].to_numpy(dtype=float)
    fit = fit_escape_rate(times, t_min=args.tmin)
    print("escape-rate fit:", fit)

    # quick topology summary
    cols = ["n_loops", "link_ok_frac", "link_nz_frac", "link_mean_abs", "link_raw_mean_abs"]
    print("\nTopology summary (overall):")
    print(df[cols].describe())

    print("\nTopology by outcome:")
    for lab, g in df.groupby("label"):
        print("==", lab, "n=", len(g))
        print(g[cols].mean(numeric_only=True))


if __name__ == "__main__":
    main()
