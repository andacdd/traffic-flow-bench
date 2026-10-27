import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm

TW, LW = 9, 10
PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 127, "min_data_in_leaf": 100, "feature_fraction": 0.5,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0, "verbose": -1, "num_threads": 8}


def patch(day, t0, vf, cap, links):
    S = day[0, t0 - TW + 1:t0 + 1] / vf[None]
    F = day[1, t0 - TW + 1:t0 + 1] / cap[None]
    S = np.pad(S, ((0, 0), (LW, LW)), constant_values=np.nan)
    F = np.pad(F, ((0, 0), (LW, LW)), constant_values=np.nan)
    idx = links[:, None] + np.arange(2 * LW + 1)[None]
    return np.concatenate([S[:, idx].transpose(1, 0, 2).reshape(len(links), -1),
                           F[:, idx].transpose(1, 0, 2).reshape(len(links), -1)], 1).astype(np.float32)


def rows(day, t0, vf, cap, links, h):
    X = patch(day, t0, vf, cap, links)
    L = len(vf)
    return np.column_stack([X, h.astype(np.float32), np.full(len(links), t0, np.float32), (links / max(L - 1, 1)).astype(np.float32)])


def build_train(rel, per_day, months, rng):
    Xs, ys = [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        L = len(P["order"])
        for d, date in enumerate(P["dates"]):
            if months is not None and date[:7] not in months:
                continue
            Q = P["Q"][d]
            cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(Q[t0]) > 0]
            for t0 in (rng.choice(cand, min(per_day, len(cand)), replace=False) if cand else []):
                day = td.hybrid(P["F"][d], P["M"][d], t0)
                links = np.arange(L)
                h = rng.integers(1, tm.H + 1, L)
                y = Q[t0 + h, links]
                ok = ~np.isnan(y)
                Xs.append(rows(day, t0, P["vf"], P["cap"], links[ok], h[ok]))
                ys.append(y[ok].astype(np.float32))
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P
        gc.collect()
    return np.concatenate(Xs), np.concatenate(ys)


def predict_split(rel, m, split):
    out = []
    for pid, panel in enumerate(tm.PANELS):
        order, vf, cap, _ = tm.network(rel, panel)
        sd, SA = tm.load_days(rel, panel, split, "mainline_states_masked", order)
        di = {d: i for i, d in enumerate(sd)}
        w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
        L = len(order)
        for r in w[w.condition == "queue_ongoing"].itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            day = SA[di[r.date]]
            links = np.tile(np.arange(L), tm.H)
            h = np.repeat(np.arange(1, tm.H + 1), L)
            p = m.predict(rows(day, t0, vf, cap, links, h))
            out.append(pd.DataFrame({"window_id": r.window_id, "h": h, "link_id": np.array(order)[links], "p_patch": p}))
    return pd.concat(out, ignore_index=True)


def holdout_eval(rel, m, months):
    from task2_predict_sel import eiou_select
    res = []
    for pid, panel in enumerate(tm.PANELS):
        P = td.load_panel(rel, pid, panel)
        di = {d: i for i, d in enumerate(P["dates"])}
        L = len(P["order"])
        w = pd.read_csv(rel / "task2" / panel / "train" / "window_index.csv")
        for r in w[(w.condition == "queue_ongoing") & w.date.str[:7].isin(months)].itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            d = di[r.date]
            Y = P["Q"][d, t0 + 1:t0 + tm.H + 1]
            links = np.tile(np.arange(L), tm.H)
            h = np.repeat(np.arange(1, tm.H + 1), L)
            p = m.predict(rows(P["M"][d], t0, P["vf"], P["cap"], links, h)).reshape(tm.H, L)
            mk = ~np.isnan(Y)
            yb = Y[mk].astype(bool)
            iou = lambda pr: float((pr[mk] & yb).sum() / max((pr[mk] | yb).sum(), 1))
            S = P["M"][d][0, t0]
            pers = np.repeat((S <= 0.6 * P["vf"])[None], tm.H, 0)
            res.append({"panel": panel, "window_id": r.window_id, "iou_eiou": iou(eiou_select(p.ravel()).reshape(p.shape)), "iou_045": iou(p >= 0.45), "iou_pers": iou(pers)})
        del P
        gc.collect()
    r = pd.DataFrame(res)
    return {"n": int(len(r)), **r[["iou_eiou", "iou_045", "iou_pers"]].mean().round(4).to_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--out", default="submissions/queue_variants/patch")
    ap.add_argument("--per-day", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    allm = sorted({f"2030-{m:02d}" for m in range(6, 13)} | {"2031-01", "2031-02"})
    X, y = build_train(rel, a.per_day, [m for m in allm if m not in a.holdout] if a.holdout else None, rng)
    info = {"rows": int(len(y)), "feats": int(X.shape[1]), "pos_rate": float(y.mean())}
    print(info, flush=True)
    s = time.time()
    m = lgb.train(PARAMS, lgb.Dataset(X, y, free_raw_data=True), a.rounds)
    info["train_s"] = round(time.time() - s)
    m.save_model(str(out / "patch_lgb.txt"))
    del X, y
    gc.collect()
    if a.holdout:
        info["holdout"] = holdout_eval(rel, m, a.holdout)
        (out / "info_holdout.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)
        return
    for split in ("validation", "private"):
        pr = predict_split(rel, m, split)
        pr.to_parquet(out / f"probs_{split}.parquet", index=False)
        info[f"{split}_rows"] = int(len(pr))
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
