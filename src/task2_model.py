import argparse
import glob
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T = 288
H = 6
HIST = 12
PANELS = ["D12_I5_N", "D12_I5_S", "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S"]
COLS = ["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy"]
NB = [-3, -2, -1, 1, 2, 3]


def network(rel, panel):
    topo = pd.read_csv(rel / "corridors" / panel / "network" / "lwr_mainline_topology.csv")
    order = topo.sort_values("order_index").mainline_link_id.tolist()
    links = pd.read_csv(rel / "corridors" / panel / "network" / "links.csv").set_index("link_id").loc[order]
    return order, links.free_speed_kmh.fillna(105.0).to_numpy(np.float32), links.capacity_vph.to_numpy(np.float32), links.lanes.to_numpy(np.float32)


def load_days(rel, panel, split, sub, order):
    idx = {l: i for i, l in enumerate(order)}
    files = sorted(glob.glob(str(rel / "corridors" / panel / split / sub / "**" / "*.parquet"), recursive=True))
    dates, arrs = [], []
    for f in files:
        d = pq.ParquetFile(f).read(columns=COLS).to_pandas()
        li = d.link_id.map(idx)
        d = d[li.notna()]
        li = li[li.notna()].astype(int).to_numpy()
        ts = pd.to_datetime(d.timestamp, utc=True)
        tod = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        a = np.full((3, T, len(order)), np.nan, np.float32)
        for k, c in enumerate(["speed_kmh", "flow_vph", "occupancy"]):
            a[k, tod, li] = d[c].to_numpy(np.float32)
        dates.append(str(d.date.iloc[0]))
        arrs.append(a)
    return dates, np.stack(arrs)


def last_valid(x):
    out = np.full(x.shape[1:], np.nan, np.float32)
    for k in range(x.shape[0]):
        v = x[k]
        out = np.where(np.isnan(v), out, v)
    return out


def shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def features(day, t0, vf, cap, lanes, freq, dow, pid, drop_t=0.0, rng=None):
    S, F, O = day[0], day[1], day[2]
    L = S.shape[1]
    hs, hf, ho = S[t0 - HIST:t0], F[t0 - HIST:t0], O[t0 - HIST:t0]
    sT, fT, oT = S[t0].copy(), F[t0].copy(), O[t0].copy()
    if drop_t > 0:
        m = rng.random(L) < drop_t
        sT[m] = np.nan
        fT[m] = np.nan
        oT[m] = np.nan
    s_last = np.where(np.isnan(sT), last_valid(hs), sT)
    f_last = np.where(np.isnan(fT), last_valid(hf), fT)
    o_last = np.where(np.isnan(oT), last_valid(ho), oT)
    with np.errstate(all="ignore"):
        r_last = s_last / vf
        r_mean = np.nanmean(hs, 0) / vf
        r_min = np.nanmin(hs, 0) / vf
        r_slope = (np.nanmean(hs[-3:], 0) - np.nanmean(hs[:3], 0)) / vf
        fc_last = f_last / cap
        fc_mean = np.nanmean(hf, 0) / cap
        fc_max = np.nanmax(hf, 0) / cap
        fc_slope = (np.nanmean(hf[-3:], 0) - np.nanmean(hf[:3], 0)) / cap
        o_mean = np.nanmean(ho, 0)
        qh = (hs <= 0.6 * vf).sum(0).astype(np.float32)
        q_now = (r_last <= 0.6).astype(np.float32)
        day_min = np.nanmin(S[:t0 + 1], 0) / vf
    cols = {"r_last": r_last, "r_mean": r_mean, "r_min": r_min, "r_slope": r_slope, "fc_last": fc_last, "fc_mean": fc_mean,
            "fc_max": fc_max, "fc_slope": fc_slope, "o_last": o_last, "o_mean": o_mean, "q_hist": qh, "q_now": q_now,
            "day_min_r": day_min, "sT_missing": np.isnan(sT).astype(np.float32)}
    for k in NB:
        cols[f"r_last_{k}"] = shift(r_last, k)
        cols[f"fc_last_{k}"] = shift(fc_last, k)
        cols[f"q_now_{k}"] = shift(q_now, k)
    dn = np.stack([shift(r_last, k) for k in range(1, 6)])
    up = np.stack([shift(r_last, -k) for k in range(1, 6)])
    with np.errstate(all="ignore"):
        cols["r_min_dn5"] = np.nanmin(dn, 0)
        cols["r_min_up5"] = np.nanmin(up, 0)
    cols["corr_q_frac"] = np.full(L, np.nanmean(q_now), np.float32)
    cols["corr_fc_mean"] = np.full(L, np.nanmean(fc_last), np.float32)
    cols["corr_r_mean"] = np.full(L, np.nanmean(r_last), np.float32)
    cols["freq_t0"] = freq[t0]
    cols["freq_max_h"] = freq[t0 + 1:t0 + H + 1].max(0)
    cols["tod"] = np.full(L, t0, np.float32)
    cols["dow"] = np.full(L, dow, np.float32)
    cols["lanes"] = lanes
    cols["cap"] = cap
    cols["pos"] = np.arange(L, dtype=np.float32) / L
    cols["panel"] = np.full(L, pid, np.float32)
    return pd.DataFrame(cols)


def queue_matrix(S, vf):
    q = (S <= 0.6 * vf).astype(np.float32)
    q[np.isnan(S)] = np.nan
    return q


def onset_origins(Q):
    out = []
    t = HIST + 6
    while t < T:
        quiet = np.nansum(Q[t - HIST - 6:t]) == 0
        if quiet and np.nansum(Q[t]) > 0:
            out.append(t - 6)
            t += HIST + 6
        else:
            t += 1
    return out


def iou(pred, true):
    k = ~np.isnan(true)
    p, y = pred[k].astype(bool), true[k].astype(bool)
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def build_train(rel, rng, ongoing_per_day):
    on_rows, og_rows, meta = [], [], {}
    for pid, panel in enumerate(PANELS):
        s = time.time()
        order, vf, cap, lanes = network(rel, panel)
        dates, A = load_days(rel, panel, "train", "mainline_states", order)
        Q = np.stack([queue_matrix(A[d, 0], vf) for d in range(len(dates))])
        freq = np.nan_to_num(np.nanmean(Q, 0)).astype(np.float32)
        meta[panel] = {"order": order, "vf": vf, "cap": cap, "lanes": lanes, "freq": freq}
        for d, date in enumerate(dates):
            dow = pd.Timestamp(date).weekday()
            for t0 in onset_origins(Q[d]):
                if t0 + H >= T:
                    continue
                X = features(A[d], t0, vf, cap, lanes, freq, dow, pid, 0.35, rng)
                X["y"] = Q[d, t0 + H]
                X["win"] = f"{panel}|{date}|{t0}"
                X["day"] = d
                on_rows.append(X)
            cand = [t0 for t0 in range(HIST, T - H) if np.nansum(Q[d, t0]) > 0]
            for t0 in rng.choice(cand, min(ongoing_per_day, len(cand)), replace=False) if cand else []:
                X = features(A[d], t0, vf, cap, lanes, freq, dow, pid, 0.35, rng)
                for h in range(1, H + 1):
                    Xh = X.copy()
                    Xh["h"] = h
                    Xh["freq_h"] = freq[t0 + h]
                    Xh["y"] = Q[d, t0 + h]
                    Xh["win"] = f"{panel}|{date}|{t0}"
                    Xh["day"] = d
                    og_rows.append(Xh)
        print(f"{panel}: {len(dates)} gün, {time.time()-s:.0f}s", flush=True)
    return pd.concat(on_rows, ignore_index=True), pd.concat(og_rows, ignore_index=True), meta


PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 4}


def fit(df, rounds):
    d = df[df.y.notna()]
    X = d.drop(columns=["y", "win", "day"])
    return lgb.train(PARAMS, lgb.Dataset(X, d.y), rounds), list(X.columns)


def win_iou(df, prob, thr, top1):
    out = []
    for w, g in df.assign(p=prob).groupby("win"):
        pred = (g.p >= thr).to_numpy()
        if top1 and not pred.any():
            pred[np.argmax(g.p.to_numpy())] = True
        out.append(iou(pred, g.y.to_numpy()))
    return float(np.mean(out))


def persistence_iou(df):
    return float(np.mean([iou(g.q_now.to_numpy() == 1, g.y.to_numpy()) for _, g in df.groupby("win")]))


def evaluate(on, og, cut, rounds):
    res = {}
    for name, df, top1 in [("onset", on, True), ("ongoing", og, False)]:
        tr, te = df[df.day < cut], df[df.day >= cut]
        m, cols = fit(tr, rounds)
        p = m.predict(te[cols])
        grid = {thr: win_iou(te, p, thr, top1) for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]}
        best = max(grid, key=grid.get)
        res[name] = {"persistence": persistence_iou(te), "grid": grid, "best_thr": best, "best_iou": grid[best], "n_win": te.win.nunique()}
        imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False).head(12)
        res[name]["top_features"] = (imp / imp.sum()).round(3).to_dict()
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "top_features"}, default=float), flush=True)
    return res


def official_windows(rel, split, meta, sub):
    rows = []
    for pid, panel in enumerate(PANELS):
        wi = rel / "task2" / panel / split / "window_index.csv"
        w = pd.read_csv(wi)
        m = meta[panel]
        dates, A = load_days(rel, panel, split, sub, m["order"])
        di = {d: i for i, d in enumerate(dates)}
        for r in w.itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            day = A[di[r.date]]
            X = features(day, t0, m["vf"], m["cap"], m["lanes"], m["freq"], T0.weekday(), pid)
            X["link_id"] = m["order"]
            X["window_id"] = r.window_id
            X["condition"] = r.condition
            X["t0"] = t0
            X["T0"] = T0
            ytrue = queue_matrix(day[0], m["vf"])[t0 + 1:t0 + H + 1] if split == "train" else None
            rows.append((X, ytrue))
    return rows


def predict_windows(rows, m_on, c_on, m_og, c_og, thr_on, thr_og):
    out, scores = [], []
    for X, ytrue in rows:
        cond = X.condition.iloc[0]
        L = len(X)
        P = np.zeros((H, L), bool)
        if cond == "queue_onset":
            p = m_on.predict(X[c_on])
            sel = p >= thr_on
            if not sel.any():
                sel[np.argmax(p)] = True
            P[H - 1] = sel
        else:
            t0 = int(X.t0.iloc[0])
            freq = FREQ[int(X.panel.iloc[0])]
            for h in range(1, H + 1):
                Xh = X.assign(h=h, freq_h=freq[min(t0 + h, T - 1)])
                P[h - 1] = m_og.predict(Xh[c_og]) >= thr_og
        if ytrue is not None:
            scores.append({"window_id": X.window_id.iloc[0], "condition": cond, "iou": iou(P.ravel(), ytrue.ravel()),
                           "persistence": iou(np.repeat(X.q_now.to_numpy()[None] == 1, H, 0).ravel(), ytrue.ravel())})
        T0 = X.T0.iloc[0]
        for h in range(1, H + 1):
            ts = (T0 + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append(pd.DataFrame({"window_id": X.window_id.iloc[0], "timestamp": ts, "link_id": X.link_id, "queue_pred": P[h - 1].astype(int)}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(scores)


FREQ = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ongoing-per-day", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    on, og, meta = build_train(rel, rng, a.ongoing_per_day)
    for pid, p in enumerate(PANELS):
        FREQ[pid] = meta[p]["freq"]
    print(f"onset satır {len(on):,} ({on.win.nunique()} pencere), ongoing satır {len(og):,} ({og.win.nunique()} pencere)", flush=True)
    res = evaluate(on, og, 218, a.rounds)
    m_on, c_on = fit(on, a.rounds)
    m_og, c_og = fit(og, a.rounds)
    thr_on, thr_og = res["onset"]["best_thr"], res["ongoing"]["best_thr"]
    _, sc = predict_windows(official_windows(rel, "train", meta, "mainline_states"), m_on, c_on, m_og, c_og, thr_on, thr_og)
    res["official_train"] = sc.groupby("condition")[["iou", "persistence"]].mean().round(4).to_dict()
    res["official_train_all"] = {"iou": float(sc.iou.mean()), "persistence": float(sc.persistence.mean())}
    print("resmi train pencereleri (in-sample model):", json.dumps(res["official_train"]), res["official_train_all"], flush=True)
    for split in ["validation", "private"]:
        sub, _ = predict_windows(official_windows(rel, split, meta, "mainline_states_masked"), m_on, c_on, m_og, c_og, thr_on, thr_og)
        sub.to_csv(out / f"queue_{split}.csv", index=False)
        print(split, len(sub), sub.queue_pred.mean().round(4), flush=True)
    (out / "task2_results.json").write_text(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main()
