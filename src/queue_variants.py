import argparse
import json
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import queue_exp3c as qx
import task1_resid_predict as trp
import task2_diag as td
import task2_model as tm

SPLITS = ["validation", "private"]
KEY = ["window_id", "timestamp", "link_id"]
LAGS = [1, 2, 3, 6, 9, 12]
NB2 = [-2, -1, 1, 2]
THR_GRID = np.round(np.arange(0.1, 0.851, 0.05), 2)
NONFEAT = {"y", "yf", "win", "day", "_f", "_m", "_w", "_g", "_L"} | set(td.FREQ_COLS)
PARAMS_M2 = {"objective": "binary", "learning_rate": 0.045, "num_leaves": 31, "max_depth": 8, "min_data_in_leaf": 250, "feature_fraction": 0.85,
             "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 8.0, "max_bin": 127, "verbose": -1, "num_threads": 4, "seed": 0}
M2_MAX, M2_ES = 700, 60
THR_V4 = (0.3, 0.35)


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def mileposts(rel, panel, order):
    ln = pd.read_csv(rel / "corridors" / panel / "network" / "links.csv").set_index("link_id").loc[order].length_km.to_numpy(np.float32)
    return ln, (np.cumsum(ln) - ln / 2).astype(np.float32)


def features_q4(day, t0, vf, cap, lanes, ln, mp, dow, pid):
    S, F = day[0], day[1]
    L = S.shape[1]
    hs, hf = S[t0 - tm.HIST:t0] / vf, F[t0 - tm.HIST:t0] / cap
    valid = ~np.isnan(hs)
    cols = {}
    for k in LAGS:
        cols[f"r_lag{k}"] = hs[-k]
        cols[f"fc_lag{k}"] = hf[-k]
    r_last, fc_last = tm.last_valid(hs), tm.last_valid(hf)
    last_i = np.where(valid.any(0), tm.HIST - 1 - np.argmax(valid[::-1], 0), -1)
    q_last = (r_last <= 0.6).astype(np.float32)
    with np.errstate(all="ignore"):
        cols.update({"r_last": r_last, "fc_last": fc_last, "age": (tm.HIST - last_i).astype(np.float32), "r_mean": np.nanmean(hs, 0),
                     "r_min": np.nanmin(hs, 0), "r_max": np.nanmax(hs, 0), "r_min3": np.nanmin(hs[-3:], 0), "r_min6": np.nanmin(hs[-6:], 0),
                     "r_d3": np.nanmean(hs[-3:], 0) - np.nanmean(hs[-6:-3], 0), "fc_mean": np.nanmean(hf, 0), "fc_max": np.nanmax(hf, 0),
                     "fc_d3": np.nanmean(hf[-3:], 0) - np.nanmean(hf[-6:-3], 0), "valid_frac": valid.mean(0).astype(np.float32),
                     "q_hist": (hs <= 0.6).sum(0).astype(np.float32), "q_last": q_last})
        r3 = np.nanmean(hs[-3:], 0)
        f3 = np.nanmean(hf[-3:], 0)
    for k in NB2:
        cols[f"d_mp_{k}"] = tm.shift(mp, k) - mp
        for j in [1, 2, 3]:
            cols[f"r_lag{j}_{k}"] = tm.shift(hs[-j], k)
        cols[f"fc_lag1_{k}"] = tm.shift(hf[-1], k)
        cols[f"r_last_{k}"] = tm.shift(r_last, k)
        cols[f"r_recent_{k}"] = tm.shift(r3, k)
        cols[f"fc_recent_{k}"] = tm.shift(f3, k)
        cols[f"q_last_{k}"] = tm.shift(q_last, k)
    with np.errstate(all="ignore"):
        cols["win_valid"] = np.full(L, valid.mean(), np.float32)
        cols["win_rmin"] = np.full(L, np.nanmin(r_last), np.float32)
        cols["win_qfrac"] = np.full(L, np.nanmean(q_last), np.float32)
    cols.update({"link_idx": np.arange(L, dtype=np.float32), "milepost": mp, "length": ln, "vf": vf, "cap": cap, "lanes": lanes,
                 "tod": np.full(L, t0, np.float32), "dow": np.full(L, dow, np.float32), "panel": np.full(L, pid, np.float32)})
    return pd.DataFrame(cols)


def tag(chunks, fold_day, wbase):
    out = []
    for X in chunks:
        date = X.win.iloc[0].split("|")[1]
        X = X.copy()
        X["_f"] = np.int8(fold_day[date])
        X["_m"] = np.int32(date[:4] + date[5:7])
        out.append(X)
    df = pd.concat(out, ignore_index=True)
    codes, _ = pd.factorize(df.win)
    df["win"] = codes.astype(np.int64) + wbase
    return df


def add_weights(df):
    n = df.y.notna().groupby(df.win).transform("sum")
    df["_w"] = (1000.0 / n.clip(lower=1)).astype(np.float32)
    return df


def feat_cols(df):
    return [c for c in df.columns if c not in NONFEAT]


def fit(df, rounds, weighted=False, params=None, fold=None):
    k = df.y.notna().to_numpy()
    if fold is not None:
        k &= (df._f != fold).to_numpy()
    cols = feat_cols(df)
    w = df._w[k] if weighted else None
    return lgb.train(params or tm.PARAMS, lgb.Dataset(df.loc[k, cols], df.y[k], weight=w), rounds), cols


def fit_m2(df, fold=None):
    k = df.y.notna().to_numpy()
    if fold is not None:
        k &= (df._f != fold).to_numpy()
    ho = int(df._m[k].max())
    cols = feat_cols(df)
    tr, va = k & (df._m != ho).to_numpy(), k & (df._m == ho).to_numpy()
    m = lgb.train(PARAMS_M2, lgb.Dataset(df.loc[tr, cols], df.y[tr]), M2_MAX, valid_sets=[lgb.Dataset(df.loc[va, cols], df.y[va])],
                  callbacks=[lgb.early_stopping(M2_ES, verbose=False)])
    it = int(m.best_iteration or M2_MAX)
    return lgb.train(PARAMS_M2, lgb.Dataset(df.loc[k, cols], df.y[k]), it), cols, ho, it


def q4_chunks(P, pid, ln, mp, days):
    out = []
    for d in days:
        date = P["dates"][d]
        dow = pd.Timestamp(date).weekday()
        for t0 in [t for t in tm.onset_origins(P["Q"][d]) if t + tm.H < tm.T]:
            X = features_q4(P["F"][d], t0, P["vf"], P["cap"], P["lanes"], ln, mp, dow, pid)
            X["y"] = P["Q"][d, t0 + tm.H]
            X["win"] = f"{pid}|{date}|{t0}"
            X["day"] = d
            out.append(X)
    return out


def og_rows(X):
    return pd.concat([X.assign(h=h) for h in range(1, tm.H + 1)], ignore_index=True)


def predict_og(m, cols, Xg):
    return m.predict(Xg[cols], num_threads=4)


def split_by(pr, wi, hh=None):
    out = {}
    if hh is None:
        for w, g in pd.Series(pr).groupby(wi):
            out[w] = g.to_numpy()
        return out
    for (w, h), g in pd.Series(pr).groupby([wi, hh]):
        out.setdefault(w, {})[h] = g.to_numpy()
    return out


def avg_probs(a, b):
    return {w: {h: 0.5 * (a[w][h] + b[w][h]) for h in a[w]} for w in a}


def variants(thr_q3):
    return {"v4": ("v4", THR_V4[0], "v4", THR_V4[1]),
            "q1_final_og": ("v4", THR_V4[0], "final", 0.40),
            "q2a_og040": ("v4", THR_V4[0], "v4", 0.40),
            "q2b_og045": ("v4", THR_V4[0], "v4", 0.45),
            "q3_elig_devthr": ("q3", thr_q3[0], "q3", thr_q3[1]),
            "q3_elig_v4thr": ("q3", THR_V4[0], "q3", THR_V4[1]),
            "q3on_elig_v4thr": ("q3", THR_V4[0], "v4", THR_V4[1]),
            "q3og_elig_v4thr": ("v4", THR_V4[0], "q3", THR_V4[1]),
            "q4_onset055": ("q4", 0.55, "v4", THR_V4[1]),
            "q5_avg035": ("v4", THR_V4[0], "avg", 0.35),
            "q5_avg040": ("v4", THR_V4[0], "avg", 0.40),
            "q5_avg045": ("v4", THR_V4[0], "avg", 0.45)}


def mats(probs, wins, spec):
    on, t_on, og, t_og = spec
    out = []
    for wi, w in enumerate(wins):
        src = on if w["condition"] == "queue_onset" else og
        out.append(qx.pred_mat(probs[src][wi], w, t_on, t_og))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--subs", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--panels", default=",".join(tm.PANELS))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t_start = time.time()
    rel, out, dest, subs = Path(a.release_root), Path(a.out), Path(a.dest), Path(a.subs)
    dest.mkdir(parents=True, exist_ok=True)
    panels = a.panels.split(",")
    full_run = panels == tm.PANELS
    timing = {}
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dev_m = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["fold"] = dev.month.map(qc.fold_of)
    rng_v4, rng_q3, rng_e1 = np.random.default_rng(0), np.random.default_rng(0), np.random.default_rng(1)
    syn = {k: [] for k in ["on", "og", "on3", "og3", "on4", "og_e1"]}
    FQ, FM = [[] for _ in qc.FOLDS], [[] for _ in qc.FOLDS]
    DX, DX4, DW, meta = [], [], [], {}
    off, wbase = 0, 0
    for panel in panels:
        pid = tm.PANELS.index(panel)
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        D = qd.load_panel(rel, panel)
        assert D["dates"] == P["dates"]
        L = len(P["order"])
        ln, mp = mileposts(rel, panel, P["order"])
        meta[panel] = {"order": P["order"], "vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "ln": ln, "mp": mp}
        months = np.array([d[:7] for d in P["dates"]])
        fold_day = {d: qc.fold_of(d[:7]) for d in P["dates"]}
        di = {d: i for i, d in enumerate(P["dates"])}
        days = list(range(len(P["dates"])))
        fq = []
        for k, f in enumerate(qc.FOLDS):
            tr = ~np.isin(months, f)
            q = np.nan_to_num(np.nanmean(P["Q"][tr], 0)).astype(np.float32)
            m6 = np.zeros_like(q)
            for t in range(tm.T - tm.H):
                m6[t] = q[t + 1:t + tm.H + 1].max(0)
            FQ[k].append(q.ravel())
            FM[k].append(m6.ravel())
            fq.append(q)
        zero = np.zeros((tm.T, L), np.float32)
        o, g = td.gen(P, pid, "hybrid", rng_v4, 2, days, lambda dt: zero)
        n_og = len(g) // tm.H
        Qe = np.where(D["E"] & ~np.isnan(D["S"]), D["Qo"], np.nan).astype(np.float32)
        o3, g3 = td.gen(dict(P, Q=Qe), pid, "hybrid", rng_q3, 2, days, lambda dt: zero)
        o4 = q4_chunks(P, pid, ln, mp, days)
        g1, c1 = qx.selector_chunks(P, D, pid, rng_e1, "og", n_og)
        for nm, ch in [("on", o), ("og", g), ("on3", o3), ("og3", g3), ("on4", o4)]:
            df = tag(ch, fold_day, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        df = qc.tag(qx.add_fill(g1, D["U"], di), L, off, fold_day, wbase)
        wbase = int(df.win.max()) + 1
        syn["og_e1"].append(df)
        del o, g, o3, g3, o4, g1
        for w in dev[dev.panel == panel].itertuples():
            d = di[w.date]
            t = int(dev_m.loc[w.window_id, "tod"])
            T0 = pd.Timestamp(w.origin)
            inp = td.hybrid(P["F"][d], P["M"][d], t)
            X = tm.features(inp, t, P["vf"], P["cap"], P["lanes"], fq[w.fold], T0.weekday(), pid)
            DX.append(X.assign(_wi=len(DW), _g=off + t * L + np.arange(L), _L=L))
            if w.condition == "queue_onset":
                DX4.append(features_q4(inp, t, P["vf"], P["cap"], P["lanes"], ln, mp, T0.weekday(), pid).assign(_wi=len(DW)))
            yf = D["U"][d][t + 1:t + tm.H + 1]
            mk = ~np.isnan(D["S"][d][t + 1:t + tm.H + 1])
            me = D["E"][d][t + 1:t + tm.H + 1] & mk
            ye = D["Qo"][d][t + 1:t + tm.H + 1] & me
            pers = np.repeat(D["Qv"][d][t - 1][None], tm.H, 0)
            DW.append({"window_id": w.window_id, "panel": panel, "month": w.month, "sample_id": w.sample_id, "condition": w.condition,
                       "fold": w.fold, "t0": t, "L": L, "yf": yf, "ye": ye, "me": me, "pers_fill": qd.iou(pers, yf), "pers_elig": qd.iou(pers, ye, me)})
        off += tm.T * L
        print(panel, {nm: int(syn[nm][-1].win.nunique()) for nm in syn}, "e1_cand", c1, f"{time.time()-s:.0f}s", f"peak {peak_gb():.1f}GB", flush=True)
        del P, D, Qe
    FQ = [np.concatenate(x) for x in FQ]
    FM = [np.concatenate(x) for x in FM]
    for nm in syn:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
    for nm in ["on3", "og3"]:
        syn[nm] = add_weights(syn[nm])
    counts = {nm: {"rows": len(df), "windows": int(df.win.nunique()), "labeled": int(df[("yf" if nm == "og_e1" else "y")].notna().sum())} for nm, df in syn.items()}
    print(counts, flush=True)
    DXa = pd.concat(DX, ignore_index=True)
    DX4a = pd.concat(DX4, ignore_index=True)
    del DX, DX4
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("yf", "ye", "me")} for w in DW])
    timing["build_s"] = round(time.time() - t_start, 1)

    probs = {k: {} for k in ["v4", "q3", "q4", "m2", "final"]}
    m2_info = []
    for k in range(len(qc.FOLDS)):
        s = time.time()
        qc.set_freq(syn["og_e1"], FQ[k], FM[k])
        on_w = W.index[(W.fold == k) & (W.condition == "queue_onset")]
        og_w = W.index[(W.fold == k) & (W.condition == "queue_ongoing")]
        xo = DXa[DXa._wi.isin(on_w)]
        xo4 = DX4a[DX4a._wi.isin(on_w)]
        xg = DXa[DXa._wi.isin(og_w)]
        Xg = pd.concat([xg.assign(h=h, freq_h=FQ[k][xg._g.to_numpy() + h * xg._L.to_numpy()]) for h in range(1, tm.H + 1)], ignore_index=True)
        wi_o, wi_o4, wi_g, hh = xo._wi.to_numpy(), xo4._wi.to_numpy(), Xg._wi.to_numpy(), Xg.h.to_numpy()
        for key, nm, weighted in [("v4", "on", False), ("q3", "on3", True)]:
            m, cols = fit(syn[nm], a.rounds, weighted, fold=k)
            probs[key].update(split_by(m.predict(xo[cols], num_threads=4), wi_o))
        m, cols = fit(syn["on4"], a.rounds, fold=k)
        probs["q4"].update(split_by(m.predict(xo4[cols], num_threads=4), wi_o4))
        for key, nm, weighted in [("v4", "og", False), ("q3", "og3", True)]:
            m, cols = fit(syn[nm], a.rounds, weighted, fold=k)
            for w, v in split_by(predict_og(m, cols, Xg), wi_g, hh).items():
                probs[key][w] = v
        m, cols, ho, it = fit_m2(syn["og"], fold=k)
        m2_info.append({"fold": k, "holdout": ho, "best_iter": it})
        for w, v in split_by(predict_og(m, cols, Xg), wi_g, hh).items():
            probs["m2"][w] = v
        m, cols, n = qx.fit(syn["og_e1"][syn["og_e1"]._f != k], "yf", a.rounds)
        for w, v in split_by(m.predict(Xg[cols], num_threads=4), wi_g, hh).items():
            probs["final"][w] = v
        print("fold", k, m2_info[-1], f"{time.time()-s:.0f}s", f"peak {peak_gb():.1f}GB", flush=True)
    probs["avg"] = avg_probs({w: probs["v4"][w] for w in probs["m2"]}, probs["m2"])
    for w in probs["v4"]:
        if w not in probs["avg"]:
            probs["avg"][w] = probs["v4"][w]
    for w in probs["v4"]:
        for key in ["final", "m2"]:
            probs[key].setdefault(w, probs["v4"][w])
    for w in probs["v4"]:
        probs["q4"].setdefault(w, probs["v4"][w])
    timing["cv_s"] = round(time.time() - t_start - timing["build_s"], 1)

    curves = []
    for src in ["v4", "q3", "q4", "avg", "m2", "final"]:
        for cond in ["queue_onset", "queue_ongoing"]:
            if (src in ["q4"] and cond == "queue_ongoing") or (src in ["avg", "m2", "final"] and cond == "queue_onset"):
                continue
            idx = [i for i, w in enumerate(DW) if w["condition"] == cond]
            for thr in THR_GRID:
                f = np.array([qd.iou(qx.pred_mat(probs[src][i], DW[i], thr, thr), DW[i]["yf"]) for i in idx])
                e = np.array([qd.iou(qx.pred_mat(probs[src][i], DW[i], thr, thr), DW[i]["ye"], DW[i]["me"]) for i in idx])
                curves.append({"model": src, "condition": cond, "thr": float(thr), "n": len(idx), "iou_fill": f.mean(), "se_fill": f.std(ddof=1) / np.sqrt(len(f)),
                               "iou_elig": e.mean(), "se_elig": e.std(ddof=1) / np.sqrt(len(e))})
    C = pd.DataFrame(curves)
    C.to_csv(out / f"variants_threshold_curves{a.tag}.csv", index=False)
    q3c = C[C.model == "q3"]
    thr_q3 = tuple(float(q3c[q3c.condition == c].sort_values(["iou_elig", "thr"], ascending=[False, True]).thr.iloc[0]) for c in ["queue_onset", "queue_ongoing"])
    V = variants(thr_q3)
    for name, spec in V.items():
        M = mats(probs, DW, spec)
        W[f"{name}_fill"] = [qd.iou(m, w["yf"]) for m, w in zip(M, DW)]
        W[f"{name}_elig"] = [qd.iou(m, w["ye"], w["me"]) for m, w in zip(M, DW)]
    W.to_csv(out / f"variants_cv_results{a.tag}.csv", index=False)
    checks = {}
    ref = pd.read_csv(out / "cv_results.csv").set_index("window_id")
    Wi = W.set_index("window_id")
    checks["v4_fill_vs_cv_results_maxabs"] = float((Wi.v4_fill - ref.v4_fill.reindex(Wi.index)).abs().max())
    fr = pd.read_csv(out / "final_cv_results.csv").set_index("window_id")
    og = Wi.condition == "queue_ongoing"
    checks["q1_ongoing_fill_vs_final_cv_maxabs"] = float((Wi.q1_final_og_fill - fr.final_fill.reindex(Wi.index))[og].abs().max())
    print("checks", checks, flush=True)

    rows = []
    rng = np.random.default_rng(0)
    for t in ["fill", "elig"]:
        for name in V:
            W[f"d_{name}_{t}"] = W[f"{name}_{t}"] - W[f"v4_{t}"]
    dcols = [c for c in W.columns if c.startswith("d_")]
    bs = qc.boot(W, dcols, a.boot, rng)[0] if full_run and a.boot > 0 else None
    for name, spec in V.items():
        r = {"variant": name, "onset_src": spec[0], "thr_on": spec[1], "ongoing_src": spec[2], "thr_og": spec[3]}
        for t in ["fill", "elig"]:
            c = f"{name}_{t}"
            r[f"dev_{t}_family"] = qd.aggregate(W, c) if full_run else W[c].mean()
            r[f"dev_{t}_mean"] = W[c].mean()
            for cond, x in W.groupby("condition"):
                r[f"dev_{t}_{cond.split('_')[1]}"] = x[c].mean()
            dc = f"d_{name}_{t}"
            r[f"diff_{t}_family"] = qd.aggregate(W, dc) if full_run else W[dc].mean()
            r[f"diff_{t}_se_window"] = W[dc].std() / np.sqrt(len(W))
            r[f"diff_{t}_se_boot80"] = float(bs[:, dcols.index(dc)].std()) if bs is not None else np.nan
            for cond, x in W.groupby("condition"):
                cn = cond.split("_")[1]
                r[f"diff_{t}_{cn}"] = x[dc].mean()
                r[f"diff_{t}_{cn}_se"] = x[dc].std() / np.sqrt(len(x))
            r[f"diff_{t}_folds_pos"] = int(sum(qd.aggregate(x, dc) > 0 for _, x in W.groupby("fold"))) if full_run else np.nan
        rows.append(r)
    pers = {t: (qd.aggregate(W, f"pers_{t}") if full_run else W[f"pers_{t}"].mean()) for t in ["fill", "elig"]}
    S = pd.DataFrame(rows)
    timing["dev_eval_s"] = round(time.time() - t_start - timing["build_s"] - timing["cv_s"], 1)
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False), flush=True)

    s = time.time()
    del syn["og_e1"], DXa, DX4a
    models = {"v4_on": fit(syn["on"], a.rounds), "q3_on": fit(syn["on3"], a.rounds, True), "q4_on": fit(syn["on4"], a.rounds),
              "v4_og": fit(syn["og"], a.rounds), "q3_og": fit(syn["og3"], a.rounds, True)}
    m, cols, ho, it = fit_m2(syn["og"])
    models["m2_og"] = (m, cols)
    m2_info.append({"fold": "production", "holdout": ho, "best_iter": it})
    del syn
    timing["prod_train_s"] = round(time.time() - s, 1)
    s = time.time()
    PW, PP = [], {k: {} for k in ["v4", "q3", "q4", "m2"]}
    frames_keys = []
    for split in SPLITS:
        for panel in panels:
            pid = tm.PANELS.index(panel)
            mt = meta[panel]
            wix = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
            dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", mt["order"])
            di = {d: i for i, d in enumerate(dates)}
            zero = np.zeros((tm.T, len(mt["order"])), np.float32)
            for r in wix.itertuples():
                T0 = pd.Timestamp(r.forecast_origin)
                t0 = T0.hour * 12 + T0.minute // 5
                day = A[di[r.date]]
                X = tm.features(day, t0, mt["vf"], mt["cap"], mt["lanes"], zero, T0.weekday(), pid)
                wi = len(PW)
                PW.append({"window_id": r.window_id, "condition": r.condition, "L": len(X), "split": split, "panel": panel})
                if r.condition == "queue_onset":
                    X4 = features_q4(day, t0, mt["vf"], mt["cap"], mt["lanes"], mt["ln"], mt["mp"], T0.weekday(), pid)
                    for key, mk, xx in [("v4", "v4_on", X), ("q3", "q3_on", X), ("q4", "q4_on", X4)]:
                        PP[key][wi] = models[mk][0].predict(xx[models[mk][1]], num_threads=4)
                else:
                    Xg = og_rows(X)
                    for key, mk in [("v4", "v4_og"), ("q3", "q3_og"), ("m2", "m2_og")]:
                        pr = models[mk][0].predict(Xg[models[mk][1]], num_threads=4)
                        PP[key][wi] = {h: pr[Xg.h.to_numpy() == h] for h in range(1, tm.H + 1)}
                for h in range(1, tm.H + 1):
                    frames_keys.append(pd.DataFrame({"window_id": r.window_id, "timestamp": (T0 + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                     "link_id": mt["order"]}))
            del A
    keys = pd.concat(frames_keys, ignore_index=True)
    PP["avg"] = {w: (avg_probs({w: PP["v4"][w]}, {w: PP["m2"][w]})[w] if w in PP["m2"] else PP["v4"][w]) for w in PP["v4"]}
    for w in PP["v4"]:
        PP["q4"].setdefault(w, PP["v4"][w])
        PP["m2"].setdefault(w, PP["v4"][w])
    v4_file = pd.read_csv(subs / "v4_queue_nofreq" / "queue_vp.csv")
    fin_file = pd.read_csv(subs / "queue_final" / "queue_vp.csv")
    wset = set(w["window_id"] for w in PW)
    v4_file = v4_file[v4_file.window_id.isin(wset)].reset_index(drop=True)
    fin_file = fin_file[fin_file.window_id.isin(wset)].reset_index(drop=True)
    tpl = pd.concat([pd.read_csv(rel / "task2" / p / sp / "sample_submission_queue.csv", usecols=KEY) for sp in SPLITS for p in panels], ignore_index=True)
    assert (keys[KEY].to_numpy() == v4_file[KEY].to_numpy()).all()
    assert (keys[KEY].to_numpy() == fin_file[KEY].to_numpy()).all()
    cond_rows = keys.window_id.map({w["window_id"]: w["condition"] for w in PW}).to_numpy()
    prod = []
    for name, spec in V.items():
        if name == "q1_final_og":
            pred = np.where(cond_rows == "queue_onset", v4_file.queue_pred.to_numpy(), fin_file.queue_pred.to_numpy())
        else:
            src = {"final": "v4"}.get(spec[2], spec[2])
            pred = np.concatenate([mats({"x": {0: PP[spec[0] if w["condition"] == "queue_onset" else src][wi]}}, [w], ("x", spec[1], "x", spec[3]))[0].ravel()
                                   for wi, w in enumerate(PW)]).astype(int)
        sub = keys.assign(queue_pred=pred)
        if name == "v4":
            checks["v4_repro_agree_vs_submitted"] = float((pred == v4_file.queue_pred.to_numpy()).mean())
            checks["v4_repro_cells_diff"] = int((pred != v4_file.queue_pred.to_numpy()).sum())
        d = dest / name
        d.mkdir(parents=True, exist_ok=True)
        sub.to_csv(d / "queue_vp.csv", index=False)
        (d / "versions.json").write_text(json.dumps(trp.versions(), indent=2))
        rd = pd.read_csv(d / "queue_vp.csv")
        ok = bool(len(rd) == len(tpl) and (rd[KEY].to_numpy() == keys[KEY].to_numpy()).all() and len(tpl.merge(rd[KEY], on=KEY)) == len(tpl)
                  and rd.duplicated(KEY).sum() == 0 and set(rd.queue_pred.unique()) <= {0, 1} and rd.timestamp.str.match(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$").all())
        r = {"variant": name, "rows": len(rd), "template_rows": len(tpl), "keys_ok": ok}
        base = v4_file.queue_pred.to_numpy()
        for cond in ["queue_onset", "queue_ongoing"]:
            k = cond_rows == cond
            cn = cond.split("_")[1]
            r[f"pos_{cn}"] = int(pred[k].sum())
            r[f"pos_{cn}_v4"] = int(base[k].sum())
            r[f"agree_{cn}"] = float((pred[k] == base[k]).mean())
            r[f"diffcells_{cn}"] = int((pred[k] != base[k]).sum())
            ch = pd.Series(pred[k] != base[k]).groupby(keys.window_id.to_numpy()[k]).any()
            r[f"windows_changed_{cn}"] = int(ch.sum())
        r["changes"] = "none" if r["diffcells_onset"] + r["diffcells_ongoing"] == 0 else ("onset_only" if r["diffcells_ongoing"] == 0 else ("ongoing_only" if r["diffcells_onset"] == 0 else "both"))
        cfg = {"variant": name, "onset_src": spec[0], "thr_onset": spec[1], "ongoing_src": spec[2], "thr_ongoing": spec[3], "rounds": a.rounds,
               "m2": m2_info[-1] if spec[2] == "avg" else None, "q1_sources": [str(subs / "v4_queue_nofreq" / "queue_vp.csv"), str(subs / "queue_final" / "queue_vp.csv")] if name == "q1_final_og" else None}
        (d / "config.json").write_text(json.dumps({**cfg, "check": r}, indent=2, default=float))
        prod.append(r)
        print(r, flush=True)
    timing["prod_predict_s"] = round(time.time() - s, 1)
    PR = pd.DataFrame(prod)
    S = S.merge(PR, on="variant")
    S["pers_fill_family"] = pers["fill"]
    S["pers_elig_family"] = pers["elig"]
    S.to_csv(out / f"variants_summary{a.tag}.csv", index=False)
    timing["total_s"] = round(time.time() - t_start, 1)
    res = {"timing": timing, "peak_rss_gb": round(peak_gb(), 2), "checks": checks, "thr_q3_dev": thr_q3, "m2": m2_info, "counts": counts,
           "panels": panels, "rounds": a.rounds, "versions": trp.versions()}
    (out / f"variants_run{a.tag}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"variants_versions{a.tag}.json").write_text(json.dumps(trp.versions(), indent=2))
    print(PR.to_string(index=False), flush=True)
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
