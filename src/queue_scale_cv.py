import argparse
import gc
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm
from task2_predict_sel import eiou_select, iou

NEAR = 10


def rows_for(P, pid, d, t0, freq, rng, keep_far):
    X = tm.features(td.hybrid(P["F"][d], P["M"][d], t0), t0, P["vf"], P["cap"], P["lanes"], freq, pd.Timestamp(P["dates"][d]).weekday(), pid)
    L = len(X)
    w = np.ones(L, np.float32)
    if keep_far < 1.0:
        q = np.nan_to_num(P["Q"][d, t0 - 2:t0 + 1]).max(0) > 0
        near = np.zeros(L, bool)
        for i in np.where(q)[0]:
            near[max(0, i - NEAR):i + NEAR + 1] = True
        far_keep = rng.random(L) < keep_far
        sel = near | far_keep
        w = np.where(near, 1.0, 1.0 / keep_far).astype(np.float32)[sel]
        X = X[sel].reset_index(drop=True)
        idx = np.where(sel)[0]
    else:
        idx = np.arange(L)
    out = []
    for h in range(1, tm.H + 1):
        out.append(X.assign(h=h, y=P["Q"][d, t0 + h][idx], w=w))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--rarity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-day", type=int, nargs="+", default=[2, 8])
    ap.add_argument("--keep-far", type=float, default=0.3)
    ap.add_argument("--rounds", type=int, nargs="+", default=[400])
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    D = D[D.condition == "queue_ongoing"]
    rr = pd.read_csv(a.rarity).set_index("window_id").rare_share
    train = {n: [] for n in a.per_day}
    test = []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        _, by_m = td.month_freqs(P, days, True)
        for n in a.per_day:
            rng = np.random.default_rng(100 + n)
            for d in days:
                cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(P["Q"][d, t0]) > 0]
                if not cand:
                    continue
                m = P["dates"][d][:7]
                for t0 in rng.choice(cand, min(n, len(cand)), replace=False):
                    for x in rows_for(P, pid, d, int(t0), by_m[m], rng, a.keep_far if n > 2 else 1.0):
                        train[n].append(x.assign(month=m))
        di = {d: i for i, d in enumerate(P["dates"])}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + tm.H >= tm.T:
                continue
            for x in rows_for(P, pid, d, t0, by_m[r.month], None, 1.0):
                test.append(x.assign(month=r.month, win=r.window_id))
        print(panel, f"{time.time()-s:.0f}s", {n: len(v) for n, v in train.items()}, flush=True)
        del P
        gc.collect()
    test = pd.concat(test, ignore_index=True)
    drop = ["y", "w", "month", "win"] + td.FREQ_COLS
    months = sorted(test.month.unique())
    folds = [months[i::3] for i in range(3)]
    res = []
    for n in a.per_day:
        tr_all = pd.concat(train[n], ignore_index=True)
        train[n] = None
        gc.collect()
        print(f"per_day {n}: {len(tr_all):,} rows", flush=True)
        for k, fm in enumerate(folds):
            tr = tr_all[~tr_all.month.isin(fm) & tr_all.y.notna()]
            X = tr.drop(columns=[c for c in drop if c in tr.columns])
            cols = list(X.columns)
            ds = lgb.Dataset(X, tr.y, weight=tr.w, free_raw_data=True)
            for R in a.rounds:
                s = time.time()
                m = lgb.train(tm.PARAMS, ds, R)
                te = test[test.month.isin(fm)]
                p = m.predict(te[cols])
                te = te.assign(p=p)
                for win, g in te.groupby("win"):
                    g = g.sort_values("h", kind="stable")
                    Pm = g.p.to_numpy().reshape(tm.H, -1)
                    Y = g.y.to_numpy().reshape(tm.H, -1)
                    for rule in ["thr0.35", "eiou"]:
                        pred = (Pm >= 0.35) if rule == "thr0.35" else eiou_select(Pm.ravel()).reshape(Pm.shape)
                        res.append({"per_day": n, "rounds": R, "fold": k, "rule": rule, "win": win, "iou": iou(pred, Y)})
                print(f"  n{n} r{R} fold {k} {time.time()-s:.0f}s", flush=True)
            del ds, X, tr
            gc.collect()
        del tr_all
        gc.collect()
    r = pd.DataFrame(res)
    r["rare"] = r.win.map(rr) > 0.5
    r.to_csv(out / "scale_cv.csv", index=False)
    g = r.groupby(["per_day", "rounds", "rule", "rare"]).iou.mean().unstack()
    g["rho0.28"] = 0.72 * g[False] + 0.28 * g[True]
    g["fold_min"] = r.groupby(["per_day", "rounds", "rule", "fold"]).iou.mean().groupby(["per_day", "rounds", "rule"]).min()
    print(g.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
