import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

PANELS = tm.PANELS + ["D12_I405_N", "D12_I405_S"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in PANELS:
        order, vf, cap, lanes = tm.network(rel, panel)
        for split in ["train", "validation", "private"]:
            dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", order)
            months = np.array([d[:7] for d in dates])
            for m in sorted(set(months)):
                X = A[months == m]
                S, Q = X[:, 0], X[:, 1] / lanes
                day = np.zeros(tm.T, bool)
                day[6 * 12:20 * 12] = True
                Sd, Qd = S[:, day], Q[:, day]
                rows.append({"panel": panel, "split": split, "month": m,
                             "q99_lane": float(np.nanmean(np.nanpercentile(Q, 99, axis=(0, 1)))),
                             "q995_lane": float(np.nanmean(np.nanpercentile(Q, 99.5, axis=(0, 1)))),
                             "qmean_day": float(np.nanmean(Qd)),
                             "queue_frac": float(np.nanmean(S <= 0.6 * vf)),
                             "v_mean_day": float(np.nanmean(Sd)),
                             "q_at_cap_ratio": float(np.nanmean(np.nanpercentile(X[:, 1], 99, axis=(0, 1)) / cap))})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scenario_shift.csv", index=False)
    g = df.groupby(["split", "month"])[["q99_lane", "q995_lane", "qmean_day", "queue_frac", "v_mean_day", "q_at_cap_ratio"]].mean().round(4)
    print(g.to_string())


if __name__ == "__main__":
    main()
