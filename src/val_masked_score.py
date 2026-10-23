import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm


def labels(rel, split="validation"):
    out = {}
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", order)
        di = {d: i for i, d in enumerate(dates)}
        w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
        for r in w.itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            out[r.window_id] = (r.condition, order, tm.queue_matrix(A[di[r.date], 0, t0 + 1:t0 + tm.H + 1], vf))
    return out


def score(sub, lab):
    sub = sub.copy()
    sub["ts"] = pd.to_datetime(sub.timestamp, utc=True)
    res = []
    for wid, g in sub[sub.window_id.isin(lab)].groupby("window_id"):
        cond, order, Y = lab[wid]
        g = g.sort_values("ts")
        P = np.zeros_like(Y, dtype=bool)
        ts = sorted(g.ts.unique())
        idx = {l: i for i, l in enumerate(order)}
        for h, t in enumerate(ts):
            x = g[g.ts == t]
            P[h, x.link_id.map(idx).to_numpy()] = x.queue_pred.to_numpy() > 0
        m = ~np.isnan(Y)
        u = (P[m] | (Y[m] > 0)).sum()
        res.append({"window_id": wid, "cond": cond, "iou": 1.0 if u == 0 else (P[m] & (Y[m] > 0)).sum() / u, "cov": m.mean(), "ypos": np.nansum(Y)})
    return pd.DataFrame(res)


if __name__ == "__main__":
    rel = Path(sys.argv[1])
    lab = labels(rel)
    for f in sys.argv[2:]:
        r = score(pd.read_csv(f), lab)
        print(f, len(r), r.groupby("cond").iou.mean().round(4).to_dict(), "all", round(r.iou.mean(), 4), "cov", round(r["cov"].mean(), 3))
