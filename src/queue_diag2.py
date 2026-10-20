import argparse
import glob
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
from scipy.stats import rankdata, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import task2_diag as td
import task2_model as tm

T, H, HIST, BUF = tm.T, tm.H, tm.HIST, 18
CH = ["speed_kmh", "flow_vph", "occupancy"]
RATE = {"R1": 0.2, "R2": 0.3, "R3": 0.5}
MONTHS = ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02"]
NONFEAT = {"y", "yf", "win", "day", "_g", "_f", "_L", "_m"}
FREQ = ["freq_t0", "freq_max_h", "freq_h"]
THR_ON, THR_OG = 0.35, 0.5
TRUTHS = ["fill", "obs", "elig"]
LB = {"pers": 0.3017, "v3": 0.7187}
AVP = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 0.8,
       "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 4, "seed": 0}


def tod_of(ts):
    return (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()


def read_day(f, idx, L, cols):
    d = pq.ParquetFile(f).read(columns=cols).to_pandas()
    li = d.link_id.map(idx)
    k = li.notna().to_numpy()
    d, li = d[k], li[k].astype(int).to_numpy()
    t = tod_of(pd.to_datetime(d.timestamp, utc=True))
    a = np.full((3, T, L), np.nan, np.float32)
    for j, c in enumerate(CH):
        a[j, t, li] = d[c].to_numpy(np.float32)
    el = np.zeros((T, L), bool)
    el[t, li] = d.is_score_eligible.to_numpy() == 1
    pct = None
    if "pct_observed" in d.columns:
        pct = np.zeros((T, L), np.int16)
        pct[t, li] = d.pct_observed.to_numpy()
    return str(d.date.iloc[0]), a, el, pct


def split_files(rel, panel, split, sub):
    return sorted(glob.glob(str(rel / "corridors" / panel / split / sub / "**" / "*.parquet"), recursive=True))


def load_masked(rel, panel, split, order):
    idx = {l: i for i, l in enumerate(order)}
    out = {}
    for f in split_files(rel, panel, split, "mainline_states_masked"):
        date, a, el, pct = read_day(f, idx, len(order), ["date", "timestamp", "link_id"] + CH + ["pct_observed", "is_score_eligible"])
        out[date] = {"A": a, "el": el, "pct": pct, "reg": f.split("mask_regime=")[1].split("/")[0]}
    return out


def load_train(rel, pid, panel):
    order, vf, cap, lanes = tm.network(rel, panel)
    idx = {l: i for i, l in enumerate(order)}
    dates, F, E = [], [], []
    for f in split_files(rel, panel, "train", "mainline_states"):
        date, a, el, _ = read_day(f, idx, len(order), ["date", "timestamp", "link_id"] + CH + ["is_score_eligible"])
        dates.append(date)
        F.append(a)
        E.append(el)
    F, E = np.stack(F), np.stack(E)
    mk = load_masked(rel, panel, "train", order)
    M = np.stack([mk[d]["A"] for d in dates])
    PCT = np.stack([mk[d]["pct"] for d in dates])
    REG = [mk[d]["reg"] for d in dates]
    del mk
    vc = (0.6 * vf).astype(np.float32)
    S = F[:, 0]
    Q = np.stack([tm.queue_matrix(S[d], vf) for d in range(len(dates))])
    U = np.stack([pd.DataFrame(s).ffill().bfill().to_numpy() <= vc for s in S])
    Qo = S <= vc
    return {"order": order, "vf": vf, "cap": cap, "lanes": lanes, "dates": dates, "F": F, "M": M, "Q": Q, "E": E, "PCT": PCT, "REG": REG,
            "vc": vc, "U": U, "Qo": Qo, "Qv": Qo & E, "pid": pid, "panel": panel}


def load_history(rel, panel, split, order):
    idx = {l: i for i, l in enumerate(order)}
    h = pd.read_parquet(rel / "task2" / panel / split / "window_history.parquet")
    h = h[h.link_id.isin(idx)]
    out = {}
    for wid, g in h.groupby("window_id"):
        ts = pd.to_datetime(g.timestamp, utc=True)
        t = tod_of(ts)
        li = g.link_id.map(idx).to_numpy()
        a = np.full((3, T, len(order)), np.nan, np.float32)
        for j, c in enumerate(CH):
            a[j, t, li] = g[c].to_numpy(np.float32)
        pct = np.full((T, len(order)), -1, np.int16)
        pct[t, li] = g.pct_observed.to_numpy()
        el = np.zeros((T, len(order)), bool)
        el[t, li] = g.is_score_eligible.to_numpy() == 1
        out[wid] = {"A": a, "pct": pct, "el": el, "rows": np.unique(t)}
    return out


def official_index(rel, split):
    w = qd.official_index(rel, split)
    w["T0"] = pd.to_datetime(w.forecast_origin, utc=True)
    return w


def runs(B):
    pad = np.zeros((1, B.shape[1]), bool)
    x = np.concatenate([pad, B, pad]).astype(np.int8)
    d = np.diff(x, axis=0)
    s = np.argwhere(d.T == 1)
    e = np.argwhere(d.T == -1)
    return e[:, 1] - s[:, 1]


def win_feats(inp, pct, elh, t0, vf, cap, vc, dow, pid, og, rate):
    S, F, O = inp
    hs, hf = S[t0 - HIST:t0], F[t0 - HIST:t0]
    qh = (hs <= vc) & elh
    early = S[:t0 - HIST]
    pe = pct[:t0 - HIST] >= 75
    be = np.isnan(early) & pe
    rl = runs(be) if be.any() else np.array([0])
    sT = S[t0]
    with np.errstate(all="ignore"):
        r5, fc5 = S[t0 - 1] / vf, F[t0 - 1] / cap
        dmin = np.nanmin(S[:t0 + 1], 0) / vf
        f = {"tod": t0, "dow": dow, "panel": pid, "og": og, "regime_rate": rate,
             "hist_cov": elh.mean(), "hist_nan": np.isnan(hs).mean(), "hist_q_cells": qh.sum(), "hist_q_links": qh.any(0).sum(),
             "hist_q_maxlink": qh.sum(0).max(), "q_links_T5": qh[-1].sum(),
             "r_T5_mean": np.nanmean(r5), "r_T5_min": np.nanmin(r5), "r_T5_std": np.nanstd(r5), "r_hist_mean": np.nanmean(hs / vf),
             "r_hist_minlink_mean": np.nanmean(np.nanmin(hs, 0) / vf), "fc_T5_mean": np.nanmean(fc5), "fc_T5_max": np.nanmax(fc5),
             "fc_T5_std": np.nanstd(fc5), "fc_hist_mean": np.nanmean(hf / cap), "o_T5_mean": np.nanmean(O[t0 - 1]),
             "nan_T": np.isnan(sT).mean(), "nan_T_elig": np.isnan(sT)[pct[t0] >= 75].mean() if (pct[t0] >= 75).any() else np.nan,
             "r_T_mean": np.nanmean(sT / vf), "nan_early": np.isnan(early).mean() if t0 > HIST else np.nan,
             "blank_early_per_link": be.sum() / S.shape[1], "gap_mean": rl.mean(), "gap_max": rl.max(), "gap_n_per_link": (rl > 0).sum() / S.shape[1],
             "day_min_r_mean": np.nanmean(dmin), "day_min_q_share": np.nanmean(dmin <= 0.6), "mid_pct_hist_present": np.nan}
    mid = (pct[t0 - HIST:t0] > 0) & (pct[t0 - HIST:t0] < 75)
    if mid.any():
        f["mid_pct_hist_present"] = (~np.isnan(hs[mid])).mean()
    return f


def cell_feats(inp, t0, P, dow):
    X = tm.features(inp, t0, P["vf"], P["cap"], P["lanes"], np.zeros((T, len(P["order"])), np.float32), dow, P["pid"])
    return X.drop(columns=[c for c in FREQ if c in X.columns])


def degrade(full, el, t0, rate, others, rng):
    x = full.copy()
    m = el & (rng.random(el.shape) < rate)
    for t1 in others:
        m[t1 + 1:t1 + 1 + BUF] = True
    m[t0 - HIST:t0] = False
    x[:, m] = np.nan
    return x


def auc(y, p):
    r = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hanley_se(a, n1, n0):
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))


def adv_cv(X, y, groups, rounds, k=5):
    ug = np.unique(groups)
    fold = dict(zip(ug, np.random.default_rng(0).integers(0, k, len(ug))))
    fk = np.array([fold[g] for g in groups])
    oof = np.zeros(len(y))
    gain = np.zeros(X.shape[1])
    for i in range(k):
        tr, te = fk != i, fk == i
        m = lgb.train(AVP, lgb.Dataset(X[tr], y[tr], feature_name=list(map(str, range(X.shape[1])))), rounds)
        oof[te] = m.predict(X[te], num_threads=4)
        gain += m.feature_importance("gain")
    return oof, gain / gain.sum()


def iou(p, y, m=None):
    return qd.iou(p, y, m)


def versions(out, name):
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
         "lightgbm": lgb.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    (out / name).write_text(json.dumps(v, indent=2))


def part_checks(rel, out):
    rows, gaps, regs = [], [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        order, vf, _, _ = tm.network(rel, panel)
        L = len(order)
        for split in ["validation", "private", "train"]:
            w = official_index(rel, split)
            w = w[w.panel == panel]
            mk = load_masked(rel, panel, split, order)
            hist = load_history(rel, panel, split, order)
            full = None
            if split == "train":
                idx = {l: i for i, l in enumerate(order)}
                full = {}
                for f in split_files(rel, panel, "train", "mainline_states"):
                    if any(f.endswith(d.replace("-", "_") + ".parquet") for d in w.date):
                        dt, a, el, _ = read_day(f, idx, L, ["date", "timestamp", "link_id"] + CH + ["is_score_eligible"])
                        full[dt] = a
            for r in pd.Series([v["reg"] for v in mk.values()]).value_counts().items():
                regs.append({"panel": panel, "split": split, "regime": r[0], "days": int(r[1])})
            excl = {}
            for r in w.itertuples():
                excl.setdefault(r.date, []).append(r.t)
            for r in w.itertuples():
                t0, day, hw = r.t, mk[r.date], hist[r.window_id]
                hrows = np.arange(t0 - HIST, t0)
                A, pct = day["A"], day["pct"]
                hA = hw["A"][:, hrows]
                mA = A[:, hrows]
                same = np.isclose(hA, mA, equal_nan=True, rtol=0, atol=0)
                hp = hw["pct"][hrows]
                hnan = np.isnan(hA[0])
                others = [t for t in excl[r.date] if t != t0]
                ov = [t for t in others if (t + 1 <= t0 - 1) and (t + BUF >= t0 - HIST)]
                el = pct >= 75
                rec = {"panel": panel, "split": split, "window_id": r.window_id, "condition": r.condition, "t0": t0, "regime": day["reg"],
                       "hist_rows_ok": bool(np.array_equal(hw["rows"], hrows)), "hist_cells_differ_vs_masked": int((~same).sum()),
                       "hist_pct_hist_vs_masked_differ": int((hp != pct[hrows]).sum()),
                       "hist_nan_pct0": int((hnan & (hp == 0)).sum()), "hist_nan_pct1_74": int((hnan & (hp > 0) & (hp < 75)).sum()),
                       "hist_nan_pct75": int((hnan & (hp >= 75)).sum()), "hist_cells_pct1_74": int(((hp > 0) & (hp < 75)).sum()),
                       "rowT_nan_share_elig": float(np.isnan(A[0, t0])[el[t0]].mean()) if el[t0].any() else np.nan,
                       "rowT_nan_share_pct1_74": float(np.isnan(A[0, t0])[(pct[t0] > 0) & (pct[t0] < 75)].mean()) if ((pct[t0] > 0) & (pct[t0] < 75)).any() else np.nan,
                       "horizon_buffer_all_nan_share": float(np.isnan(A[0, t0 + 1:t0 + 1 + BUF]).mean()) if t0 + BUF < T else np.nan,
                       "after_buffer_nan_share_elig": float(np.isnan(A[0, t0 + 1 + BUF:t0 + 1 + BUF + 6])[el[t0 + 1 + BUF:t0 + 1 + BUF + 6]].mean()) if t0 + BUF + 7 < T else np.nan,
                       "n_other_windows_same_day": len(others), "n_other_before": sum(t < t0 for t in others),
                       "history_overlaps_other_horizon_buffer": len(ov), "min_abs_origin_gap_steps": min([abs(t - t0) for t in others], default=-1)}
                if full is not None:
                    fa = full[r.date]
                    rec["train_hist_vs_full_differ"] = int((~np.isclose(hA, fa[:, hrows], equal_nan=True, rtol=0, atol=0)).sum())
                    rec["train_horizon_nan_share_masked"] = float(np.isnan(A[0, t0 + 1:t0 + 1 + BUF]).mean())
                    rec["train_horizon_nan_share_full"] = float(np.isnan(fa[0, t0 + 1:t0 + 1 + BUF]).mean())
                rows.append(rec)
            days = list(mk.keys()) if split != "train" else list(mk.keys())[::3]
            for d in days:
                day = mk[d]
                ex = np.zeros(T, bool)
                for t in excl.get(d, []):
                    ex[t - HIST:min(T, t + 1 + BUF)] = True
                el = day["pct"] >= 75
                B = np.isnan(day["A"][0]) & el
                B[ex] = False
                rl = runs(B)
                n_el = (el & ~ex[:, None]).sum()
                hist_rl = np.bincount(np.minimum(rl, 10), minlength=11)[1:]
                g = {"panel": panel, "split": split, "date": d, "regime": day["reg"], "elig_cells": int(n_el), "blank_rate": float(B.sum() / max(n_el, 1)),
                     "n_runs": int(len(rl)), "mean_run": float(rl.mean()) if len(rl) else np.nan,
                     "mid_pct_present": float((~np.isnan(day["A"][0]))[(day["pct"] > 0) & (day["pct"] < 75) & ~ex[:, None]].mean())}
                g.update({f"run_{i + 1}{'+' if i == 9 else ''}": int(c) for i, c in enumerate(hist_rl)})
                gaps.append(g)
            del mk, hist, full
        print("checks", panel, f"{time.time()-s:.0f}s", flush=True)
    C = pd.DataFrame(rows)
    G = pd.DataFrame(gaps)
    R = pd.DataFrame(regs)
    C.to_csv(out / "maskmimic_checks_windows.csv", index=False)
    G.to_csv(out / "maskmimic_gaps_days.csv", index=False)
    R.to_csv(out / "maskmimic_regimes.csv", index=False)
    rc = [c for c in G.columns if c.startswith("run_")]
    GS = G.groupby(["split", "regime"]).agg(days=("date", "size"), blank_rate=("blank_rate", "mean"), mean_run=("mean_run", "mean"),
                                            mid_pct_present=("mid_pct_present", "mean"), **{c: (c, "sum") for c in rc}).reset_index()
    tot = GS[rc].sum(1)
    for c in rc:
        GS[c] = GS[c] / tot
    GS["geom_run1_expected"] = 1 - GS.blank_rate
    GS.to_csv(out / "maskmimic_gaps.csv", index=False)
    num = [c for c in C.columns if C[c].dtype != object and c not in ("t0",)]
    CS = C.groupby("split")[num].agg(["mean", "max"])
    CS.columns = [f"{a}_{b}" for a, b in CS.columns]
    CS.T.to_csv(out / "maskmimic_checks.csv")
    RS = R.groupby(["split", "regime"]).days.sum().unstack()
    print(CS.T.round(4).to_string(), flush=True)
    print(GS.round(4).to_string(index=False), flush=True)
    print(RS.to_string(), flush=True)


def dev_frame(out):
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dm = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["t0"] = dm.loc[dev.window_id, "tod"].to_numpy()
    dev["official"] = dm.loc[dev.window_id, "official"].to_numpy()
    dev["fold"] = dev.month.map(qc.fold_of)
    dev["m"] = dev.month.map(MONTHS.index)
    return dev, dm


def val_regime_shares(rel):
    R = []
    for panel in tm.PANELS:
        fs = split_files(rel, panel, "validation", "mainline_states_masked")
        R.append(pd.Series([f.split("mask_regime=")[1].split("/")[0] for f in fs]).value_counts(normalize=True).rename(panel))
    return pd.concat(R, axis=1).fillna(0)


def part_adv(rel, out, rounds):
    dev, _ = dev_frame(out)
    W, C = [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = load_train(rel, pid, panel)
        di = {d: i for i, d in enumerate(P["dates"])}
        for w in dev[dev.panel == panel].itertuples():
            d, t0 = di[w.date], int(w.t0)
            inp = td.hybrid(P["F"][d], P["M"][d], t0)
            og = int(w.condition == "queue_ongoing")
            dow = pd.Timestamp(w.date).weekday()
            elh = P["E"][d][t0 - HIST:t0]
            src = "dev"
            W.append({"src": src, "window_id": w.window_id, "official": bool(w.official), "month": w.month,
                      **win_feats(inp, P["PCT"][d], elh, t0, P["vf"], P["cap"], P["vc"], dow, pid, og, RATE[P["REG"][d]])})
            X = cell_feats(inp, t0, P, dow)
            X["og"] = np.float32(og)
            X["_w"] = len(W) - 1
            C.append(X)
        del P
        order, vf, cap, lanes = tm.network(rel, panel)
        Pm = {"order": order, "vf": vf, "cap": cap, "lanes": lanes, "pid": pid, "vc": (0.6 * vf).astype(np.float32)}
        for split in ["train", "validation", "private"]:
            wi = official_index(rel, split)
            wi = wi[wi.panel == panel]
            mk = load_masked(rel, panel, split, order)
            hist = load_history(rel, panel, split, order)
            for r in wi.itertuples():
                t0, day, hw = r.t, mk[r.date], hist[r.window_id]
                inp = day["A"].copy()
                inp[:, t0 - HIST:t0] = hw["A"][:, t0 - HIST:t0]
                og = int(r.condition == "queue_ongoing")
                dow = r.T0.weekday()
                elh = hw["el"][t0 - HIST:t0]
                W.append({"src": {"train": "official_train", "validation": "val", "private": "private"}[split], "window_id": r.window_id,
                          "official": True, "month": r.date[:7],
                          **win_feats(inp, day["pct"], elh, t0, vf, cap, Pm["vc"], dow, pid, og, RATE[day["reg"]])})
                X = cell_feats(inp, t0, Pm, dow)
                X["og"] = np.float32(og)
                X["_w"] = len(W) - 1
                C.append(X)
            del mk, hist
        print("adv", panel, f"{time.time()-s:.0f}s", flush=True)
    Wf = pd.DataFrame(W)
    Cf = pd.concat(C, ignore_index=True)
    Wf.to_parquet(out / "adv_window_features.parquet", index=False)
    wcols = [c for c in Wf.columns if c not in ("src", "window_id", "official", "month")]
    ccols = [c for c in Cf.columns if c != "_w"]
    comps = [("dev_vs_valpriv", Wf.src == "dev", Wf.src.isin(["val", "private"])),
             ("dev_vs_val", Wf.src == "dev", Wf.src == "val"),
             ("dev_vs_private", Wf.src == "dev", Wf.src == "private"),
             ("val_vs_private", Wf.src == "val", Wf.src == "private"),
             ("devNonOfficial_vs_officialTrain", (Wf.src == "dev") & ~Wf.official, Wf.src == "official_train"),
             ("devJune_vs_officialTrain", (Wf.src == "dev") & (Wf.month == "2030-06") & ~Wf.official, Wf.src == "official_train"),
             ("officialTrain_vs_valpriv", Wf.src == "official_train", Wf.src.isin(["val", "private"]))]
    res, imp, oofs = [], [], {}
    for name, a0, a1 in comps:
        for level in ["window", "cell"]:
            sel = (a0 | a1).to_numpy()
            yw = a1.to_numpy().astype(np.int8)
            if level == "window":
                X = Wf.loc[sel, wcols].to_numpy(np.float32)
                y = yw[sel]
                g = np.flatnonzero(sel)
                cols = wcols
            else:
                k = sel[Cf._w.to_numpy()]
                X = Cf.loc[k, ccols].to_numpy(np.float32)
                g = Cf._w.to_numpy()[k]
                y = yw[g]
                cols = ccols
            oof, gain = adv_cv(X, y, g, rounds)
            r = {"comparison": name, "level": level, "n_rows": len(y), "n_pos_windows": int(a1.sum()), "n_neg_windows": int(a0.sum()), "auc": auc(y, oof),
                 "auc_se_hanley_windows": hanley_se(auc(y, oof), int(a1.sum()), int(a0.sum()))}
            ww = pd.DataFrame({"g": g, "y": y, "p": oof}).groupby("g").agg(y=("y", "first"), p=("p", "mean"))
            r["auc_cell_meaned_to_window"] = auc(ww.y.to_numpy(), ww.p.to_numpy()) if level == "cell" else np.nan
            og = Wf.og.to_numpy()[ww.index.to_numpy()]
            for cv, nm in [(0, "onset"), (1, "ongoing")]:
                mm = og == cv
                r[f"auc_window_{nm}"] = auc(ww.y.to_numpy()[mm], ww.p.to_numpy()[mm]) if ww.y[mm].nunique() == 2 else np.nan
            res.append(r)
            for f, v in sorted(zip(cols, gain), key=lambda x: -x[1])[:15]:
                imp.append({"comparison": name, "level": level, "feature": f, "gain_share": float(v)})
            if name == "dev_vs_valpriv":
                oofs[level] = (cols, gain)
            print(json.dumps(r), flush=True)
    A = pd.DataFrame(res)
    I = pd.DataFrame(imp)
    A.to_csv(out / "adv_auc.csv", index=False)
    I.to_csv(out / "adv_importance.csv", index=False)
    stats = []
    for level, df, cols in [("window", Wf, wcols), ("cell", Cf.assign(src=Wf.src.to_numpy()[Cf._w.to_numpy()]), ccols)]:
        for c in cols:
            for src, g in df.groupby("src"):
                x = g[c].astype(float)
                stats.append({"level": level, "feature": c, "src": src, "mean": x.mean(), "std": x.std(), "q10": x.quantile(0.1), "q50": x.median(),
                              "q90": x.quantile(0.9), "nan_share": x.isna().mean()})
    pd.DataFrame(stats).to_csv(out / "adv_feature_stats.csv", index=False)
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for i, (level, df) in enumerate([("window", Wf), ("cell", Cf.assign(src=Wf.src.to_numpy()[Cf._w.to_numpy()]))]):
        top = I[(I.comparison == "dev_vs_valpriv") & (I.level == level)].feature.head(3).tolist()
        for j, c in enumerate(top):
            x = df[c].astype(float)
            lo, hi = np.nanquantile(x, 0.005), np.nanquantile(x, 0.995)
            bins = np.linspace(lo, hi, 31) if hi > lo else 10
            for src in ["dev", "official_train", "val", "private"]:
                v = x[df.src == src].dropna()
                ax[i, j].hist(v, bins=bins, alpha=0.45 if src in ("dev", "val") else 0.35, density=True, label=src, histtype="stepfilled" if src == "dev" else "step", lw=1.5)
            ax[i, j].set_title(f"{level}: {c}")
    ax[0, 0].legend()
    fig.tight_layout()
    fig.savefig(out / "adv_top3.png", dpi=110)
    pd.set_option("display.width", 250)
    print(A.round(4).to_string(index=False), flush=True)
    print(I[I.comparison.isin(["dev_vs_valpriv", "devNonOfficial_vs_officialTrain"])].round(4).to_string(index=False), flush=True)


def part_advx(rel, out, rounds):
    Wf = pd.read_parquet(out / "adv_window_features.parquet")
    allc = [c for c in Wf.columns if c not in ("src", "window_id", "official", "month")]
    mask_c = [c for c in allc if c.startswith(("gap_", "nan_", "blank_")) or c in ("hist_nan", "regime_rate", "mid_pct_hist_present", "hist_cov")]
    groups = {"all": allc, "no_mask": [c for c in allc if c not in mask_c], "mask_only": mask_c + ["og", "tod"],
              "state_only": [c for c in allc if c not in mask_c and c not in ("tod", "dow", "panel")]}
    comps = [("dev_vs_valpriv", (Wf.src == "dev"), Wf.src.isin(["val", "private"])),
             ("devNonOfficial_vs_officialTrain", (Wf.src == "dev") & ~Wf.official, Wf.src == "official_train"),
             ("officialTrain_vs_valpriv", Wf.src == "official_train", Wf.src.isin(["val", "private"]))]
    res, imp = [], []
    for name, a0, a1 in comps:
        for cond in ["all", "onset", "ongoing"]:
            cm = np.ones(len(Wf), bool) if cond == "all" else (Wf.og.to_numpy() == (cond == "ongoing"))
            for gname, cols in groups.items():
                sel = ((a0 | a1).to_numpy()) & cm
                X = Wf.loc[sel, cols].to_numpy(np.float32)
                y = a1.to_numpy()[sel].astype(np.int8)
                oof, gain = adv_cv(X, y, np.flatnonzero(sel), rounds)
                a = auc(y, oof)
                res.append({"comparison": name, "condition": cond, "features": gname, "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum()), "auc": a,
                            "auc_se_hanley": hanley_se(a, int(y.sum()), int(len(y) - y.sum()))})
                for f, v in sorted(zip(cols, gain), key=lambda x: -x[1])[:8]:
                    imp.append({"comparison": name, "condition": cond, "features": gname, "feature": f, "gain_share": float(v)})
    A = pd.DataFrame(res)
    I = pd.DataFrame(imp)
    A.to_csv(out / "adv_auc_by_condition_group.csv", index=False)
    I.to_csv(out / "adv_importance_by_condition_group.csv", index=False)
    pd.set_option("display.width", 250)
    print(A.round(3).to_string(index=False), flush=True)
    print(I[(I.comparison == "dev_vs_valpriv") & (I.features.isin(["no_mask", "state_only"]))].round(3).to_string(index=False), flush=True)
    st = Wf.assign(cond=np.where(Wf.og == 1, "ongoing", "onset")).groupby(["cond", "src"])[
        ["hist_q_cells", "hist_q_links", "q_links_T5", "r_T5_mean", "fc_T5_mean", "day_min_r_mean", "gap_max", "tod"]].mean()
    st.to_csv(out / "adv_condition_means.csv")
    print(st.round(3).to_string(), flush=True)


def wagg(df, col, w):
    x = df.assign(_v=df[col] * w, _w=w)
    g = x.groupby(["panel", "condition"])[["_v", "_w"]].sum()
    pc = (g._v / g._w).groupby("panel").mean()
    return float(pc.groupby(pc.index.map(qd.FAMILY)).mean().mean())


def part_weight(rel, out, rounds):
    Wf = pd.read_parquet(out / "adv_window_features.parquet")
    allc = [c for c in Wf.columns if c not in ("src", "window_id", "official", "month")]
    mask_c = [c for c in allc if c.startswith(("gap_", "nan_", "blank_")) or c in ("hist_nan", "regime_rate", "mid_pct_hist_present", "hist_cov")]
    sets = {"state_only": [c for c in allc if c not in mask_c and c not in ("tod", "dow", "panel")], "no_mask": [c for c in allc if c not in mask_c]}
    R = pd.read_csv(out / "maskmimic_train_results.csv")
    F = pd.read_csv(out / "final_cv_results.csv")[["window_id", "final_fill", "final_obs"]]
    C = pd.read_csv(out / "cv_results.csv")[["window_id", "v4_fill", "v4_obs"]]
    R = R.merge(F, on="window_id", how="left").merge(C, on="window_id", how="left")
    cols = [f"{n}_{t}" for n in ["pers", "v3_base"] for t in TRUTHS] + ["final_fill", "final_obs", "v4_fill", "v4_obs", "v3_mimic_elig", "v3deg_mimic_elig"]
    rows = []
    wts = {}
    for sname, fc in sets.items():
        w = pd.Series(np.nan, index=Wf.index)
        for cv in [0, 1]:
            sel = ((Wf.src.isin(["dev", "val", "private"])) & (Wf.og == cv)).to_numpy()
            X = Wf.loc[sel, fc].to_numpy(np.float32)
            y = Wf.src[sel].isin(["val", "private"]).to_numpy().astype(np.int8)
            ps = []
            for seed in range(5):
                ug = np.flatnonzero(sel)
                fk = np.random.default_rng(seed).integers(0, 5, len(ug))
                oof = np.zeros(len(ug))
                for i in range(5):
                    prm = {**AVP, "seed": seed, "num_leaves": 7, "min_data_in_leaf": 40}
                    m = lgb.train(prm, lgb.Dataset(X[fk != i], y[fk != i]), rounds)
                    oof[fk == i] = m.predict(X[fk == i], num_threads=4)
                ps.append(oof)
            p = np.clip(np.mean(ps, 0), 1e-4, 1 - 1e-4)
            r = p / (1 - p)
            dv = y == 0
            r = np.minimum(r, np.quantile(r[dv], 0.99))
            w.loc[np.flatnonzero(sel)[dv]] = r[dv] / r[dv].mean()
            rows.append({"weights": sname, "condition": "ongoing" if cv else "onset", "metric": "auc_oof_5seed", "value": auc(y, p)})
            rows.append({"weights": sname, "condition": "ongoing" if cv else "onset", "metric": "ess_share", "value": float(r[dv].sum() ** 2 / (r[dv] ** 2).sum() / dv.sum())})
        wd = Wf[Wf.src == "dev"].assign(w=w[Wf.src == "dev"])
        wts[sname] = wd.set_index("window_id").w
        RW = R.merge(wd[["window_id", "w"]], on="window_id")
        for c in cols:
            if c in RW and RW[c].notna().all():
                rows.append({"weights": sname, "condition": "all", "metric": c, "value": wagg(RW, c, RW.w), "unweighted": qd.aggregate(RW, c)})
                for cn, g in RW.groupby("condition"):
                    rows.append({"weights": sname, "condition": cn, "metric": c, "value": float(np.average(g[c], weights=g.w)), "unweighted": float(g[c].mean())})
        for c in ["hist_q_cells", "q_links_T5", "r_T5_mean", "day_min_r_mean"]:
            for cv in [0, 1]:
                g = wd[wd.og == cv]
                v = Wf[(Wf.src.isin(["val", "private"])) & (Wf.og == cv)][c]
                rows.append({"weights": sname, "condition": "ongoing" if cv else "onset", "metric": f"balance_{c}", "value": float(np.average(g[c].fillna(g[c].mean()), weights=g.w)),
                             "unweighted": float(g[c].mean()), "target_valpriv": float(v.mean())})
    S = pd.DataFrame(rows)
    S.to_csv(out / "adv_weighted_scores.csv", index=False)
    pd.DataFrame(wts).to_csv(out / "adv_dev_weights.csv")
    pd.set_option("display.width", 250)
    print(S.round(4).to_string(index=False), flush=True)


def gen_windows(P, rng, days):
    zero = np.zeros((T, len(P["order"])), np.float32)
    o, g = td.gen(P, P["pid"], "full", rng, 2, days, lambda dt: zero)
    return o, g


def gen_degraded(P, base_on, base_og, rng, shares):
    zero = np.zeros((T, len(P["order"])), np.float32)
    di = {d: i for i, d in enumerate(P["dates"])}
    regs = shares.index.to_numpy()
    pr = shares.to_numpy()
    on, og = [], []
    for kind, chunks, dst in [("on", base_on, on), ("og", base_og, og)]:
        for X0 in chunks:
            _, date, t0 = X0.win.iloc[0].split("|")
            if kind == "og" and int(X0.h.iloc[0]) != 1:
                continue
            d, t0 = di[date], int(t0)
            rate = RATE[regs[rng.choice(len(regs), p=pr)]]
            inp = degrade(P["F"][d], P["E"][d], t0, rate, [], rng)
            X = tm.features(inp, t0, P["vf"], P["cap"], P["lanes"], zero, pd.Timestamp(date).weekday(), P["pid"])
            if kind == "on":
                dst.append(X.assign(y=P["Q"][d, t0 + H], win=X0.win.iloc[0], day=d))
            else:
                for h in range(1, H + 1):
                    dst.append(X.assign(h=h, freq_h=zero[t0 + h], y=P["Q"][d, t0 + h], win=X0.win.iloc[0], day=d))
    return on, og


def tag(chunks, L, off, P, wbase):
    mi = {d: MONTHS.index(d[:7]) for d in P["dates"]}
    out = []
    for X in chunks:
        _, date, t0 = X.win.iloc[0].split("|")
        X = X.copy()
        X["_g"] = (off + int(t0) * L + np.arange(L)).astype(np.int64)
        X["_m"] = np.int8(mi[date])
        X["_L"] = np.int64(L)
        out.append(X)
    df = pd.concat(out, ignore_index=True)
    codes, _ = pd.factorize(df.win)
    df["win"] = codes.astype(np.int64) + wbase
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def freq_arrays(Qs, months_of, excl):
    fq, fm = [], []
    for Q, mo in zip(Qs, months_of):
        tr = ~np.isin(mo, excl)
        q = np.nan_to_num(np.nanmean(Q[tr], 0)).astype(np.float32)
        m6 = np.zeros_like(q)
        for t in range(T - H):
            m6[t] = q[t + 1:t + H + 1].max(0)
        fq.append(q.ravel())
        fm.append(m6.ravel())
    return np.concatenate(fq), np.concatenate(fm)


def fit(df, rowmask, rounds):
    cols = [c for c in df.columns if c not in NONFEAT]
    k = df.y.notna().to_numpy() & rowmask
    X = np.empty((int(k.sum()), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = df[c].to_numpy()[k]
    m = lgb.train(tm.PARAMS, lgb.Dataset(X, df.y.to_numpy()[k], feature_name=cols, free_raw_data=True), rounds)
    del X
    return m, cols


def to_mat(df, cols):
    X = np.empty((len(df), len(cols)), np.float32)
    for j, c in enumerate(cols):
        X[:, j] = df[c].to_numpy()
    return X


def build_common(rel, out, variants, shares, with_degraded_train):
    dev, dm = dev_frame(out)
    rng_v3 = np.random.default_rng(0)
    rng_mim = np.random.default_rng(11)
    rng_tr = np.random.default_rng(12)
    syn = {"on": [], "og": []}
    if with_degraded_train:
        syn.update({"on_deg": [], "og_deg": []})
    DX = {v: [] for v in variants}
    DW, Qs, MO = [], [], []
    off, wbase = 0, 0
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = load_train(rel, pid, panel)
        L = len(P["order"])
        di = {d: i for i, d in enumerate(P["dates"])}
        Qs.append(P["Q"])
        MO.append(np.array([d[:7] for d in P["dates"]]))
        o, g = gen_windows(P, rng_v3, list(range(len(P["dates"]))))
        parts = [("on", o), ("og", g)]
        if with_degraded_train:
            od, gd = gen_degraded(P, o, g, rng_tr, shares[panel])
            parts += [("on_deg", od), ("og_deg", gd)]
        for nm, ch in parts:
            df = tag(ch, L, off, P, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        del o, g, parts
        dp = dev[dev.panel == panel]
        same = dp.groupby(["month", "sample_id", "date"]).t0.apply(list).to_dict()
        for w in dp.itertuples():
            d, t = di[w.date], int(w.t0)
            dow = pd.Timestamp(w.date).weekday()
            for v in variants:
                if v == "base":
                    inp = td.hybrid(P["F"][d], P["M"][d], t)
                else:
                    reg = shares[panel].index.to_numpy()
                    rate = RATE[reg[rng_mim.choice(len(reg), p=shares[panel].to_numpy())]]
                    others = [x for x in same[(w.month, w.sample_id, w.date)] if x < t]
                    inp = degrade(P["F"][d], P["E"][d], t, rate, others, rng_mim)
                X = tm.features(inp, t, P["vf"], P["cap"], P["lanes"], np.zeros((T, L), np.float32), dow, pid)
                DX[v].append(X.assign(_wi=len(DW), _g=off + t * L + np.arange(L), _L=L))
            yf = P["U"][d][t + 1:t + H + 1]
            mk = ~np.isnan(P["F"][d, 0, t + 1:t + H + 1])
            yo = P["Qo"][d][t + 1:t + H + 1] & mk
            me = P["E"][d][t + 1:t + H + 1]
            ye = P["Qo"][d][t + 1:t + H + 1] & me
            pers = np.repeat(P["Qv"][d][t - 1][None], H, 0)
            DW.append({"window_id": w.window_id, "panel": panel, "month": w.month, "m": w.m, "sample_id": w.sample_id, "condition": w.condition,
                       "fold": w.fold, "t0": t, "L": L, "yf": yf, "yo": yo, "mk": mk, "ye": ye, "me": me,
                       "pers_fill": iou(pers, yf), "pers_obs": iou(pers, yo, mk), "pers_elig": iou(pers, ye, me)})
        off += T * L
        print(panel, {nm: int(syn[nm][-1].win.nunique()) for nm in syn}, f"{time.time()-s:.0f}s", flush=True)
        del P
    for nm in syn:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
    DX = {v: pd.concat(x, ignore_index=True) for v, x in DX.items()}
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("yf", "yo", "mk", "ye", "me")} for w in DW])
    chk = W.set_index("window_id").join(dm[["persistence_fill", "persistence_obs"]])
    pc = {"pers_fill_maxdiff": float((chk.pers_fill - chk.persistence_fill).abs().max()), "pers_obs_maxdiff": float((chk.pers_obs - chk.persistence_obs).abs().max())}
    print("persistence check", pc, {nm: (len(df), int(df.win.nunique())) for nm, df in syn.items()}, flush=True)
    return syn, DX, DW, W, Qs, MO, pc


def predict_dev(models, DXv, W, sel_wi, fq):
    probs = {}
    m_on, c_on = models["on"]
    m_og, c_og = models["og"]
    xs = DXv[DXv._wi.isin(sel_wi)]
    g = xs._g.to_numpy()
    xs = xs.assign(freq_t0=fq[0][g], freq_max_h=fq[1][g])
    on = xs._wi.isin(W.index[W.condition == "queue_onset"]).to_numpy()
    xo = xs[on]
    po = m_on.predict(to_mat(xo, c_on), num_threads=4)
    for wi, gg in pd.Series(po).groupby(xo._wi.to_numpy()):
        probs[wi] = gg.to_numpy()
    xg = xs[~on]
    for h in range(1, H + 1):
        Xh = xg.assign(h=h, freq_h=fq[0][xg._g.to_numpy() + h * xg._L.to_numpy()])
        pg = m_og.predict(to_mat(Xh, c_og), num_threads=4)
        for wi, gg in pd.Series(pg).groupby(Xh._wi.to_numpy()):
            probs.setdefault(wi, {})[h] = gg.to_numpy()
    return probs


def pred_mat(pr, w):
    P = np.zeros((H, w["L"]), bool)
    if w["condition"] == "queue_onset":
        sel = pr >= THR_ON
        if not sel.any():
            sel[np.argmax(pr)] = True
        P[-1] = sel
    else:
        for h in range(1, H + 1):
            P[h - 1] = pr[h] >= THR_OG
    return P


def score(probs, DW, W, name):
    for t in TRUTHS:
        if f"{name}_{t}" not in W:
            W[f"{name}_{t}"] = np.nan
    for wi, pr in probs.items():
        w = DW[wi]
        M = pred_mat(pr, w)
        W.loc[wi, f"{name}_fill"] = iou(M, w["yf"])
        W.loc[wi, f"{name}_obs"] = iou(M, w["yo"], w["mk"])
        W.loc[wi, f"{name}_elig"] = iou(M, w["ye"], w["me"])


def summarize(W, names, extra_levels):
    rows = []
    for n in names:
        for t in TRUTHS:
            c = f"{n}_{t}"
            if c not in W or W[c].isna().all():
                continue
            x = W[W[c].notna()]
            rows.append({"model": n, "truth": t, "level": "overall", "group": "all", "n": len(x), "mean": x[c].mean(), "family_agg": qd.aggregate(x, c),
                         "se_window": x[c].std() / np.sqrt(len(x))})
            for lv in ["condition"] + extra_levels:
                for g, y in x.groupby(lv):
                    rows.append({"model": n, "truth": t, "level": lv, "group": g, "n": len(y), "mean": y[c].mean(),
                                 "family_agg": qd.aggregate(y, c) if lv in ("month", "fold") and y.panel.nunique() == len(tm.PANELS) else np.nan,
                                 "se_window": y[c].std() / np.sqrt(len(y))})
    return pd.DataFrame(rows)


def part_mimic(rel, out, rounds, train_degraded):
    t_start = time.time()
    shares = val_regime_shares(rel)
    variants = ["base", "mimic"]
    syn, DX, DW, W, Qs, MO, pc = build_common(rel, out, variants, shares, train_degraded)
    srcs = [("v3", "on", "og")] + ([("v3deg", "on_deg", "og_deg")] if train_degraded else [])
    for k in range(3):
        s = time.time()
        fq = freq_arrays(Qs, MO, qc.FOLDS[k])
        for nm in syn:
            qc.set_freq(syn[nm], fq[0], fq[1])
        sel = W.index[W.fold == k]
        for rname, on, og in srcs:
            models = {"on": fit(syn[on], (syn[on]._m.to_numpy() // 3) != k, rounds), "og": fit(syn[og], (syn[og]._m.to_numpy() // 3) != k, rounds)}
            for v in variants:
                score(predict_dev(models, DX[v], W, sel, fq), DW, W, f"{rname}_{v}")
            print("fold", k, rname, f"{time.time()-s:.0f}s", flush=True)
    for t in TRUTHS:
        W[f"pers_base_{t}"] = W[f"pers_{t}"]
        W[f"pers_mimic_{t}"] = W[f"pers_{t}"]
    ref = pd.read_csv(out / "cv_results.csv").set_index("window_id")
    chk = {**pc, "v3_base_fill_vs_cv_results_maxdiff": float((W.set_index("window_id").v3_base_fill - ref.v3_fill.reindex(W.window_id)).abs().max()),
           "v3_base_obs_vs_cv_results_maxdiff": float((W.set_index("window_id").v3_base_obs - ref.v3_obs.reindex(W.window_id)).abs().max())}
    print("checks", chk, flush=True)
    tag_ = "maskmimic_train" if train_degraded else "maskmimic"
    W.to_csv(out / f"{tag_}_results.csv", index=False)
    names = ["pers_base", "pers_mimic"] + [f"{r}_{v}" for r, _, _ in srcs for v in variants]
    S = summarize(W, names, ["fold"])
    for r, _, _ in srcs:
        for t in TRUTHS:
            d = W[f"{r}_mimic_{t}"] - W[f"{r}_base_{t}"]
            S.loc[len(S)] = {"model": f"delta_{r}_mimic_minus_base", "truth": t, "level": "overall", "group": "all", "n": len(d), "mean": d.mean(),
                             "family_agg": qd.aggregate(W.assign(_d=d), "_d"), "se_window": d.std() / np.sqrt(len(d))}
    if train_degraded:
        for v in variants:
            for t in TRUTHS:
                d = W[f"v3deg_{v}_{t}"] - W[f"v3_{v}_{t}"]
                S.loc[len(S)] = {"model": f"delta_v3deg_minus_v3_on_{v}", "truth": t, "level": "overall", "group": "all", "n": len(d), "mean": d.mean(),
                                 "family_agg": qd.aggregate(W.assign(_d=d), "_d"), "se_window": d.std() / np.sqrt(len(d))}
    S["runtime_s"] = round(time.time() - t_start, 1)
    S.to_csv(out / f"{tag_}_summary.csv", index=False)
    (out / f"{tag_}_checks.json").write_text(json.dumps({**chk, "val_regime_shares": shares.to_dict()}, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(S[S.level.isin(["overall", "condition"])].round(4).to_string(index=False), flush=True)


def part_lomo(rel, out, rounds):
    t_start = time.time()
    syn, DX, DW, W, Qs, MO, pc = build_common(rel, out, ["base"], None, False)
    plans = [("lomo", MONTHS[i], [MONTHS[i]], lambda m, i=i: m != i) for i in range(9)]
    plans += [("forward", MONTHS[i], MONTHS[i:], lambda m, i=i: m < i) for i in range(3, 9)]
    plans += [("fold3", f"fold{k}", qc.FOLDS[k], lambda m, k=k: (m // 3) != k) for k in range(3)]
    for scheme, key, excl, trf in plans:
        s = time.time()
        fq = freq_arrays(Qs, MO, excl)
        for nm in syn:
            qc.set_freq(syn[nm], fq[0], fq[1])
        tm_on = trf(syn["on"]._m.to_numpy())
        tm_og = trf(syn["og"]._m.to_numpy())
        models = {"on": fit(syn["on"], tm_on, rounds), "og": fit(syn["og"], tm_og, rounds)}
        sel = W.index[W.month.isin([key])] if scheme != "fold3" else W.index[W.fold == int(key[-1])]
        score(predict_dev(models, DX["base"], W, sel, fq), DW, W, f"v3{scheme}")
        print(scheme, key, "train windows", int(syn["on"].win[tm_on].nunique()), int(syn["og"].win[tm_og].nunique()),
              "dev", len(sel), f"{time.time()-s:.0f}s", flush=True)
    ref = pd.read_csv(out / "cv_results.csv").set_index("window_id")
    chk = {**pc, "v3fold3_fill_vs_cv_results_maxdiff": float((W.set_index("window_id").v3fold3_fill - ref.v3_fill.reindex(W.window_id)).abs().max())}
    print("checks", chk, flush=True)
    W.to_csv(out / "lomo_results.csv", index=False)
    S = summarize(W, ["pers", "v3fold3", "v3lomo", "v3forward"], ["month", "fold"])
    for t in TRUTHS:
        for a_, b_ in [("v3lomo", "v3fold3"), ("v3forward", "v3lomo")]:
            x = W[W[f"{a_}_{t}"].notna() & W[f"{b_}_{t}"].notna()]
            d = x[f"{a_}_{t}"] - x[f"{b_}_{t}"]
            S.loc[len(S)] = {"model": f"delta_{a_}_minus_{b_}", "truth": t, "level": "overall", "group": "all", "n": len(d), "mean": d.mean(),
                             "family_agg": qd.aggregate(x.assign(_d=d), "_d") if x.panel.nunique() == len(tm.PANELS) else np.nan, "se_window": d.std() / np.sqrt(len(d))}
            for m, y in x.assign(_d=d).groupby("month"):
                S.loc[len(S)] = {"model": f"delta_{a_}_minus_{b_}", "truth": t, "level": "month", "group": m, "n": len(y), "mean": y._d.mean(),
                                 "family_agg": np.nan, "se_window": y._d.std() / np.sqrt(len(y))}
    S["runtime_s"] = round(time.time() - t_start, 1)
    S.to_csv(out / "lomo_summary.csv", index=False)
    (out / "lomo_checks.json").write_text(json.dumps(chk, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(S[S.level.isin(["overall", "condition"])].round(4).to_string(index=False), flush=True)
    piv = S[(S.level == "month") & (S.truth == "fill") & S.model.isin(["pers", "v3fold3", "v3lomo", "v3forward"])].pivot(index="group", columns="model", values="mean")
    print(piv.round(4).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", required=True, choices=["checks", "adv", "advx", "weight", "mimic", "mimic_train", "lomo"])
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--adv-rounds", type=int, default=300)
    ap.add_argument("--panels", default=None)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.panels:
        tm.PANELS[:] = a.panels.split(",")
    np.seterr(all="ignore")
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    if a.part == "checks":
        part_checks(rel, out)
    elif a.part == "adv":
        part_adv(rel, out, a.adv_rounds)
    elif a.part == "advx":
        part_advx(rel, out, a.adv_rounds)
    elif a.part == "weight":
        part_weight(rel, out, 150)
    elif a.part == "mimic":
        part_mimic(rel, out, a.rounds, False)
    elif a.part == "mimic_train":
        part_mimic(rel, out, a.rounds, True)
    else:
        part_lomo(rel, out, a.rounds)
    versions(out, "diag2_versions.json")
    print("runtime", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
