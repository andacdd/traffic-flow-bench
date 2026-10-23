import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm


def jac(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum() / u


def main():
    rel, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = []
    for pid, panel in enumerate(tm.PANELS):
        P = td.load_panel(rel, pid, panel)
        full = np.nan_to_num(np.nanmean(P["Q"], 0))
        ords = pd.to_datetime(P["dates"]).values.astype("datetime64[D]").astype(np.int64)
        di = {d: i for i, d in enumerate(P["dates"])}
        oi = {o: i for i, o in enumerate(ords)}
        w = pd.read_csv(rel / "task2" / panel / "train" / "window_index.csv")
        for r in w[w.condition == "queue_onset"].itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            d = di[r.date]
            Q = P["Q"][d]
            y = Q[t0 + tm.H] > 0
            k = int(y.sum())
            idx = np.where(y)[0]
            S = P["F"][d, 0]
            r_t = S[t0] / P["vf"]
            f30 = full[t0 + tm.H]
            topf = np.zeros_like(y)
            topf[np.argsort(-f30)[:max(k, 1)]] = True
            topr = np.zeros_like(y)
            topr[np.argsort(np.nan_to_num(r_t, nan=9))[:max(k, 1)]] = True
            prev = oi.get(ords[d] - 1)
            yday = (np.nan_to_num(P["Q"][prev, t0 + tm.H]) > 0) if prev is not None else np.zeros_like(y)
            ywin = (np.nan_to_num(P["Q"][prev, max(0, t0 - 6):t0 + 13]) > 0).any(0) if prev is not None else np.zeros_like(y)
            pre = [int(np.nansum(Q[t0 + h])) for h in range(1, tm.H)]
            post = [int(np.nansum(Q[t0 + h])) for h in range(tm.H, min(tm.T - t0, 19))]
            res.append({"panel": panel, "window_id": r.window_id, "tod": t0, "k": k, "contig": int(k > 0 and idx.max() - idx.min() + 1 == k),
                        "pos": idx.mean() / len(y) if k else np.nan, "freq_at_y": f30[y].mean() if k else np.nan, "freq_rank": float(np.mean([np.sum(f30 > f30[i]) for i in idx])) if k else np.nan,
                        "r_t_at_y": np.nanmean(r_t[y]) if k else np.nan, "r_t_min": np.nanmin(r_t),
                        "iou_topfreq_k": jac(topf, y), "iou_lowspeed_k": jac(topr, y), "iou_yday_same": jac(yday, y), "iou_yday_win": jac(ywin, y),
                        "pre_q_cells": sum(pre), "post_q": post[:6], "hist_cov": r.history_coverage})
        del P
    r = pd.DataFrame(res)
    r.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(r.drop(columns=["post_q", "window_id"]).describe().round(3).T.to_string())
    print(r[["panel", "tod", "k", "contig", "freq_at_y", "freq_rank", "r_t_at_y", "iou_topfreq_k", "iou_lowspeed_k", "iou_yday_win", "post_q"]].round(2).to_string())


if __name__ == "__main__":
    main()
