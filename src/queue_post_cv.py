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
from queue_scale_cv import rows_for
from task2_predict_sel import eiou_select, iou


def smooth(P, a):
    if a <= 0:
        return P
    S = P.copy()
    S[:, 1:] += a * P[:, :-1]
    S[:, :-1] += a * P[:, 1:]
    S[1:] += a * P[:-1]
    S[:-1] += a * P[1:]
    n = np.ones_like(P)
    n[:, 1:] += a
    n[:, :-1] += a
    n[1:] += a
    n[:-1] += a
    return S / n


def build(rel, D, per_day):
    train, test = [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        _, by_m = td.month_freqs(P, days, True)
        rng = np.random.default_rng(100 + per_day)
        for d in days:
            cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(P["Q"][d, t0]) > 0]
            if not cand:
                continue
            m = P["dates"][d][:7]
            for t0 in rng.choice(cand, min(per_day, len(cand)), replace=False):
                for x in rows_for(P, pid, d, int(t0), by_m[m], rng, 1.0):
                    train.append(x.assign(month=m))
        di = {d: i for i, d in enumerate(P["dates"])}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + tm.H >= tm.T:
                continue
            for x in rows_for(P, pid, d, t0, by_m[r.month], None, 1.0):
                test.append(x.assign(month=r.month, win=r.window_id))
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P
        gc.collect()
    return pd.concat(train, ignore_index=True), pd.concat(test, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--rarity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    D = D[D.condition == "queue_ongoing"]
    rare = pd.read_csv(a.rarity).set_index("window_id").rare_share > 0.5
    tr_all, test = build(rel, D, 2)
    drop = ["y", "w", "month", "win"] + td.FREQ_COLS
    months = sorted(test.month.unique())
    folds = [months[i::3] for i in range(3)]
    test["p"] = np.nan
    probs = np.zeros((a.seeds, len(test)), np.float32)
    for k, fm in enumerate(folds):
        tr = tr_all[~tr_all.month.isin(fm) & tr_all.y.notna()]
        X = tr.drop(columns=[c for c in drop if c in tr.columns])
        cols = list(X.columns)
        te_idx = np.where(test.month.isin(fm))[0]
        for sd in range(a.seeds):
            s = time.time()
            p = dict(tm.PARAMS, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            m = lgb.train(p, lgb.Dataset(X, tr.y), a.rounds)
            probs[sd, te_idx] = m.predict(test.iloc[te_idx][cols])
            print(f"fold {k} seed {sd} {time.time()-s:.0f}s", flush=True)
        del X, tr
        gc.collect()
    np.save(out / "probs.npy", probs)
    test[["win", "h", "y"]].to_parquet(out / "test_index.parquet")
    variants = {"s0_thr0.35": (False, 1.0, 0.0, "thr"), "s0_eiou": (False, 1.0, 0.0, "eiou"), "avg_thr0.35": (True, 1.0, 0.0, "thr"), "avg_eiou": (True, 1.0, 0.0, "eiou")}
    for g in [0.7, 0.85, 1.2, 1.5]:
        variants[f"avg_eiou_g{g}"] = (True, g, 0.0, "eiou")
    for sm in [0.1, 0.25, 0.5]:
        variants[f"avg_eiou_sm{sm}"] = (True, 1.0, sm, "eiou")
    res = []
    for win, idx in test.groupby("win").indices.items():
        g = test.iloc[idx]
        o = np.argsort(g.h.to_numpy(), kind="stable")
        idx = idx[o]
        Y = test.y.to_numpy()[idx].reshape(tm.H, -1)
        for name, (avg, gam, sm, rule) in variants.items():
            pr = probs[:, idx].mean(0) if avg else probs[0, idx]
            Pm = smooth(np.clip(pr, 0, 1).reshape(tm.H, -1) ** gam, sm)
            pred = (Pm >= 0.35) if rule == "thr" else eiou_select(Pm.ravel()).reshape(Pm.shape)
            res.append({"win": win, "variant": name, "iou": iou(pred, Y)})
    r = pd.DataFrame(res)
    r["rare"] = r.win.map(rare)
    r.to_csv(out / "post_cv.csv", index=False)
    t = r.groupby(["variant", "rare"]).iou.mean().unstack()
    t.columns = ["normal", "rare"]
    t["rho0.28"] = 0.72 * t.normal + 0.28 * t.rare
    t["rho0.4"] = 0.6 * t.normal + 0.4 * t.rare
    print(t.sort_values("rho0.28", ascending=False).round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
