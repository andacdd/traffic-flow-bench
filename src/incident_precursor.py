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
    ap.add_argument("--windows", required=True)
    ap.add_argument("--rarity", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    rr = pd.read_csv(a.rarity).set_index("window_id")
    D = D[D.condition == "queue_onset"].join(rr[["rare_share"]], on="window_id")
    rows = []
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        months = np.array([d[:7] for d in dates])
        S, Qf = F[:, 0] / vf, F[:, 1]
        di = {d: i for i, d in enumerate(dates)}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + tm.H >= tm.T:
                continue
            other = months != r.month
            prof_v = np.nanmedian(S[other, t0], 0)
            prof_q = np.nanmedian(Qf[other, t0], 0)
            y = (F[d, 0, t0 + tm.H] <= 0.6 * vf)
            if not y.any():
                continue
            heads = np.where(y)[0]
            near = np.unique(np.clip(np.concatenate([heads + k for k in range(-2, 3)]), 0, len(order) - 1))
            far = np.setdiff1d(np.arange(len(order)), np.unique(np.clip(np.concatenate([heads + k for k in range(-10, 11)]), 0, len(order) - 1)))
            for t_rel in [0, -3, -6]:
                v = S[d, t0 + t_rel]
                q = Qf[d, t0 + t_rel]
                dv = v - np.nanmedian(S[other, t0 + t_rel], 0)
                dq = q / np.maximum(np.nanmedian(Qf[other, t0 + t_rel], 0), 1) - 1
                rows.append({"window_id": r.window_id, "panel": panel, "rare": r.rare_share > 0.5, "t_rel": t_rel,
                             "dv_near": np.nanmean(dv[near]), "dv_far": np.nanmean(dv[far]) if len(far) else np.nan,
                             "min_v_near": np.nanmin(v[near]), "min_v_far": np.nanmin(v[far]) if len(far) else np.nan,
                             "dq_near": np.nanmean(dq[near]), "dq_far": np.nanmean(dq[far]) if len(far) else np.nan,
                             "rank_minv_near": float(np.mean(np.nanmin(v[near]) <= v[~np.isnan(v)]))})
        print(panel, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["rare", "t_rel"])[["dv_near", "dv_far", "min_v_near", "min_v_far", "dq_near", "dq_far", "rank_minv_near"]].mean().round(3).to_string())
    print(df.groupby("rare").window_id.nunique().to_dict())


if __name__ == "__main__":
    main()
