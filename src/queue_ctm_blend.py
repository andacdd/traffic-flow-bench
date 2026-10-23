import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_ctm as qc
import task2_model as tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cfg", default="zsh6")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(a.windows).set_index("window_id")
    rows = []
    for panel in tm.PANELS:
        z = np.load(Path(a.preds) / f"v3_{panel}.npz")
        ids = z["window_id"]
        w = W.loc[ids]
        keep = (w.condition == "queue_ongoing").to_numpy()
        ids, prob = ids[keep], z["prob"][keep]
        w = w[keep]
        order, vf, _, _ = tm.network(rel, panel)
        p = qc.params(rel, panel, order)
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        prof = qc.profile(F)
        di = {d: i for i, d in enumerate(dates)}
        t0s = np.array([o.hour * 12 + o.minute // 5 for o in pd.to_datetime(w.origin)])
        ds = np.array([di[str(x)[:10]] for x in w.date])
        Sv, Sq = F[ds, 0], F[ds, 1]
        cut = 0.6 * vf
        V = qc.forecast(Sv, Sq, t0s, p, prof, dict(qc.CONFIGS[a.cfg], mask=None))
        ctm = (V <= cut).astype(float)
        margin = np.clip((cut - V) / 10.0, -1, 1)
        Y = np.stack([Sv[b, t + 1:t + tm.H + 1] for b, t in enumerate(t0s)])
        Yt = np.where(np.isnan(Y), np.nan, (Y <= cut).astype(float))
        vT = np.stack([Sv[b, t] for b, t in enumerate(t0s)])
        pers = np.repeat((vT <= cut)[:, None], tm.H, 1)
        for b in range(len(ids)):
            r = {"panel": panel, "window_id": ids[b], "persistence": qc.iou(pers[b], Yt[b]), "ctm": qc.iou(ctm[b] > 0.5, Yt[b])}
            for thr in [0.35, 0.45]:
                r[f"lgb_{thr}"] = qc.iou(prob[b] >= thr, Yt[b])
            for al in [0.3, 0.5, 0.7]:
                s = al * prob[b] + (1 - al) * ctm[b]
                for thr in [0.35, 0.5, 0.65]:
                    r[f"bl{al}_{thr}"] = qc.iou(s >= thr, Yt[b])
                s2 = al * prob[b] + (1 - al) * (0.5 + 0.5 * margin[b])
                r[f"bm{al}"] = qc.iou(s2 >= 0.45, Yt[b])
            rows.append(r)
        df = pd.DataFrame([x for x in rows if x["panel"] == panel])
        print(panel, len(df), df[["persistence", "ctm", "lgb_0.35", "bl0.5_0.5", "bm0.5"]].mean().round(3).to_dict(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"blend_{a.cfg}.csv", index=False)
    cols = [c for c in df.columns if c not in ("panel", "window_id")]
    pm = df.groupby("panel")[cols].mean()
    print(pm.mean().sort_values(ascending=False).round(4).head(25).to_string())


if __name__ == "__main__":
    main()
