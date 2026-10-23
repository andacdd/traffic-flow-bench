import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

RESUME = 19


def iou(p, y, m):
    p, y = p[m], y[m]
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    rows = []
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        cut = 0.6 * vf
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        di = {d: i for i, d in enumerate(dates)}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + RESUME + 3 >= tm.T:
                continue
            S = F[d, 0]
            Y = S[t0 + 1:t0 + tm.H + 1]
            m = ~np.isnan(Y)
            y = Y <= cut
            a0 = S[t0]
            a1 = np.nanmean(S[t0 + RESUME:t0 + RESUME + 3], 0)
            w = np.arange(1, tm.H + 1)[:, None] / RESUME
            lin = (1 - w) * a0[None] + w * a1[None]
            pers = np.repeat((a0 <= cut)[None], tm.H, 0)
            back = np.repeat((a1 <= cut)[None], tm.H, 0)
            pred_lin = lin <= cut
            if r.condition == "queue_onset":
                pred_lin = np.zeros_like(pred_lin)
                pred_lin[-1] = (a1 <= cut) | (lin[-1] <= cut)
            rows.append({"panel": panel, "condition": r.condition, "pers": iou(pers, y, m), "future_back": iou(back, y, m), "future_interp": iou(pred_lin, y, m)})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby("condition")[["pers", "future_back", "future_interp"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
