import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    rows = []
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        cut = 0.6 * vf
        dates, M = tm.load_days(rel, panel, "train", "mainline_states_masked", order)
        months = np.array([d[:7] for d in dates])
        Q = (M[:, 0] <= cut).astype(np.float32)
        Q[np.isnan(M[:, 0])] = np.nan
        groups = [(m, Q[months == m], np.nanmean(Q[months != m], 0)) for m in sorted(set(months))]
        for split in ["validation", "private"]:
            sd, S = tm.load_days(rel, panel, split, "mainline_states_masked", order)
            q = (S[:, 0] <= cut).astype(np.float32)
            q[np.isnan(S[:, 0])] = np.nan
            groups.append((split, q, np.nanmean(Q, 0)))
        for name, q, ref in groups:
            ref = np.nan_to_num(ref)
            qq = np.nan_to_num(q)
            rare = ref[None] < 0.02
            tot = np.nansum(q)
            rows.append({"panel": panel, "month": name, "queued_cells": float(tot), "rare_queued": float((qq * rare).sum()),
                         "rare_share": float((qq * rare).sum() / max(tot, 1)), "days_with_rare_cluster": int(((qq * rare).sum(axis=(1, 2)) >= 30).sum()),
                         "n_days": len(q)})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df.groupby("month")[["queued_cells", "rare_queued", "days_with_rare_cluster", "n_days"]].sum()
    g["rare_share"] = g.rare_queued / g.queued_cells
    g["rare_cluster_days_per_30"] = g.days_with_rare_cluster / g.n_days * 30 * 8 / 8
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
