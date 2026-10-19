import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import queue_error_analysis as qe
import task2_diag as td
import task2_model as tm

THR_ON, THR_OG = 0.35, 0.5
NAMES = ["v3", "e1", "e2", "e3"]
SRC = {"v3": ("on|y", "og|y"), "e1": ("on|y", "og_e1|y"), "e2": ("on_e2|y", "og|y"), "e3": ("on|yf", "og|yf")}
NONFEAT = {"y", "yf", "win", "day"} | set(qc.AUX)


def add_fill(chunks, U, di):
    for X in chunks:
        _, date, t0 = X.win.iloc[0].split("|")
        h = int(X.h.iloc[0]) if "h" in X.columns else tm.H
        X["yf"] = U[di[date]][int(t0) + h].astype(np.float32)
    return chunks


def selector_chunks(P, D, pid, rng, kind, n_og):
    c = qd.panel_candidates(D, "T")
    zero = np.zeros((tm.T, len(P["order"])), np.float32)
    if kind == "og":
        x = c[c.og].reset_index(drop=True)
        n_cand = len(x)
        x = x.iloc[np.sort(rng.choice(len(x), min(n_og, len(x)), replace=False))]
    else:
        x = c[~c.og]
        U = D["U"]
        ok = np.array([(not U[d][t + 1:t + tm.H].any()) and U[d][t + tm.H].any() for d, t in zip(x.day, x.t)], bool)
        n_cand = len(x)
        x = x[ok]
    out = []
    for d, t in zip(x.day.to_numpy(), x.t.to_numpy()):
        d, t = int(d), int(t)
        date = P["dates"][d]
        X = tm.features(P["F"][d], t, P["vf"], P["cap"], P["lanes"], zero, pd.Timestamp(date).weekday(), pid, 0.35, rng)
        if kind == "og":
            for h in range(1, tm.H + 1):
                out.append(X.assign(h=h, freq_h=zero[t + h], y=P["Q"][d, t + h], win=f"{pid}|{date}|{t}", day=d))
        else:
            X["y"] = P["Q"][d, t + tm.H]
            X["win"] = f"{pid}|{date}|{t}"
            X["day"] = d
            out.append(X)
    return out, n_cand


def fit(df, ycol, rounds):
    y = df[ycol]
    k = y.notna().to_numpy()
    cols = [c for c in df.columns if c not in NONFEAT]
    return lgb.train(tm.PARAMS, lgb.Dataset(df.loc[k, cols], y[k]), rounds), cols, int(k.sum())


def pred_mat(pr, w, thr_on, thr_og):
    P = np.zeros((tm.H, w["L"]), bool)
    if w["condition"] == "queue_onset":
        sel = pr >= thr_on
        if not sel.any():
            sel[np.argmax(pr)] = True
        P[-1] = sel
    else:
        for h in range(1, tm.H + 1):
            P[h - 1] = pr[h] >= thr_og
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--panels", default=",".join(tm.PANELS))
    ap.add_argument("--folds", default="0,1,2")
    ap.add_argument("--errcat", default="auto", choices=["auto", "skip"])
    a = ap.parse_args()
    t_start = time.time()
    rel, out = Path(a.release_root), Path(a.out)
    dest = Path(a.dest) if a.dest else out
    dest.mkdir(parents=True, exist_ok=True)
    cache = Path(a.cache)
    panels = a.panels.split(",")
    folds = [int(x) for x in a.folds.split(",")]
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dev_m = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["fold"] = dev.month.map(qc.fold_of)
    dev = dev[dev.fold.isin(folds)]
    rng_v3, rng_e1, rng_e2 = np.random.default_rng(0), np.random.default_rng(1), np.random.default_rng(2)
    syn = {k: [] for k in ["on", "og", "og_e1", "on_e2"]}
    counts = []
    FQ, FM = [[] for _ in qc.FOLDS], [[] for _ in qc.FOLDS]
    DX, DW, store = [], [], {}
    off, wbase = 0, 0
    for panel in panels:
        pid = tm.PANELS.index(panel)
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        D = qd.load_panel(rel, panel)
        assert D["dates"] == P["dates"]
        L = len(P["order"])
        months = np.array([d[:7] for d in P["dates"]])
        fold_day = {d: qc.fold_of(d[:7]) for d in P["dates"]}
        di = {d: i for i, d in enumerate(P["dates"])}
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
        o, g = td.gen(P, pid, "full", rng_v3, 2, list(range(len(P["dates"]))), lambda dt: zero)
        n_og = len(g) // tm.H
        g1, c1 = selector_chunks(P, D, pid, rng_e1, "og", n_og)
        o2, c2 = selector_chunks(P, D, pid, rng_e2, "on", 0)
        counts.append({"panel": panel, "v3_on_win": len(o), "v3_og_win": n_og, "e1_og_cand": c1, "e1_og_win": len(g1) // tm.H,
                       "e2_on_cand_noog": c2, "e2_on_win": len(o2)})
        for nm, ch in [("on", add_fill(o, D["U"], di)), ("og", add_fill(g, D["U"], di)), ("og_e1", g1), ("on_e2", o2)]:
            df = qc.tag(ch, L, off, fold_day, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        lf = np.nan_to_num(np.nanmean(P["Q"], (0, 1))).astype(np.float32)
        ids, Sp, Yh, UT = [], [], [], []
        for w in dev[dev.panel == panel].itertuples():
            d = di[w.date]
            t = int(dev_m.loc[w.window_id, "tod"])
            T0 = pd.Timestamp(w.origin)
            assert T0.hour * 12 + T0.minute // 5 == t
            X = tm.features(td.hybrid(P["F"][d], P["M"][d], t), t, P["vf"], P["cap"], P["lanes"], fq[w.fold], T0.weekday(), pid)
            yf = D["U"][d][t + 1:t + tm.H + 1]
            mk = ~np.isnan(D["S"][d][t + 1:t + tm.H + 1])
            yo = D["Qo"][d][t + 1:t + tm.H + 1] & mk
            pers = np.repeat(D["Qv"][d][t - 1][None], tm.H, 0)
            DX.append(X.assign(_wi=len(DW), _g=off + t * L + np.arange(L), _L=L))
            DW.append({"window_id": w.window_id, "panel": panel, "month": w.month, "sample_id": w.sample_id, "condition": w.condition,
                       "fold": w.fold, "t0": t, "L": L, "yf": yf, "yo": yo, "mk": mk,
                       "pers_fill": qd.iou(pers, yf), "pers_obs": qd.iou(pers, yo, mk)})
            ids.append(w.window_id)
            Sp.append(D["S"][d][t - tm.HIST:t + tm.H + 1])
            Yh.append(yf)
            UT.append(D["U"][d][t - tm.HIST:t + 1])
        store[panel] = {"window_id": np.array(ids), "speed": np.stack(Sp).astype(np.float32), "yf": np.stack(Yh), "u_hist": np.stack(UT),
                        "vc": D["vc"], "vf": P["vf"], "link_qfreq": lf}
        off += tm.T * L
        print(panel, counts[-1], f"{time.time()-s:.0f}s", flush=True)
        del P, D
    FQ = [np.concatenate(x) for x in FQ]
    FM = [np.concatenate(x) for x in FM]
    for nm in syn:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
    Cn = pd.DataFrame(counts)
    print(Cn.to_string(index=False), flush=True)
    print({nm: (len(df), df.win.nunique(), int(df.y.notna().sum())) for nm, df in syn.items()}, flush=True)
    DXa = pd.concat(DX, ignore_index=True)
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("yf", "yo", "mk")} for w in DW])
    chk = W.set_index("window_id").join(dev_m[["persistence_fill", "persistence_obs"]])
    print("persistence check", float((chk.pers_fill - chk.persistence_fill).abs().max()), float((chk.pers_obs - chk.persistence_obs).abs().max()), flush=True)

    models = [("on", "y"), ("og", "y"), ("og_e1", "y"), ("on_e2", "y"), ("on", "yf"), ("og", "yf")]
    probs = {f"{s}|{y}": {} for s, y in models}
    ntrain = []
    for k in folds:
        s = time.time()
        for nm in syn:
            qc.set_freq(syn[nm], FQ[k], FM[k])
        dk = DXa[DXa._wi.isin(W.index[W.fold == k])]
        xo = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_onset")])]
        xg = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_ongoing")])]
        Xg = pd.concat([xg.assign(h=h, freq_h=FQ[k][xg._g.to_numpy() + h * xg._L.to_numpy()]) for h in range(1, tm.H + 1)], ignore_index=True)
        for src, ycol in models:
            key = f"{src}|{ycol}"
            tr = syn[src][syn[src]._f != k]
            m, cols, n = fit(tr, ycol, a.rounds)
            ntrain.append({"fold": k, "model": key, "rows": n, "windows": int(tr.win.nunique())})
            del tr
            if src.startswith("on"):
                pr = m.predict(xo[cols], num_threads=4)
                for wi, g in pd.Series(pr).groupby(xo._wi.to_numpy()):
                    probs[key][wi] = g.to_numpy()
            else:
                pr = m.predict(Xg[cols], num_threads=4)
                for (wi, h), g in pd.Series(pr).groupby([Xg._wi.to_numpy(), Xg.h.to_numpy()]):
                    probs[key].setdefault(wi, {})[h] = g.to_numpy()
            print("fold", k, key, n, f"{time.time()-s:.0f}s", flush=True)
    NT = pd.DataFrame(ntrain)
    print(NT.pivot(index="model", columns="fold", values="rows").to_string(), flush=True)

    def recipe_prob(r, wi):
        on, og = SRC[r]
        return probs[on if DW[wi]["condition"] == "queue_onset" else og][wi]

    for r in NAMES:
        fi, oi = [], []
        for wi, w in enumerate(DW):
            P = pred_mat(recipe_prob(r, wi), w, THR_ON, THR_OG)
            fi.append(qd.iou(P, w["yf"]))
            oi.append(qd.iou(P, w["yo"], w["mk"]))
        W[f"{r}_fill"] = fi
        W[f"{r}_obs"] = oi
    W.to_csv(dest / "exp3c_results.csv", index=False)
    ref = pd.read_csv(out / "cv_results.csv").set_index("window_id")
    v3chk = float((W.set_index("window_id").v3_fill - ref.v3_fill.reindex(W.window_id)).abs().max())
    print("v3 vs cv_results max abs diff", v3chk, flush=True)

    names = ["pers"] + NAMES
    cols = [f"{n}_{t}" for n in names for t in ["fill", "obs"]]
    Wd = W.assign(**{f"d_{x}_v3_{t}": W[f"{x}_{t}"] - W[f"v3_{t}"] for x in NAMES[1:] for t in ["fill", "obs"]})
    dcols = [c for c in Wd.columns if c.startswith("d_")]
    full = a.boot > 0 and len(panels) == len(tm.PANELS)
    if full:
        bs, rel_sc = qc.boot(Wd, cols + dcols, a.boot, np.random.default_rng(0))
    summ = []
    for i, c in enumerate(cols + dcols):
        base = {"metric": c, "level": "overall", "group": "all", "n": len(Wd), "mean": Wd[c].mean(), "se_window": Wd[c].std() / np.sqrt(len(Wd)),
                "family_agg": qd.aggregate(Wd, c)}
        if full:
            base.update({"se_boot80": bs[:, i].std(), "pseudo_release_mean": rel_sc[:, i].mean(), "pseudo_release_std": rel_sc[:, i].std(ddof=1)})
        summ.append(base)
        for lv in ["condition", "panel", "fold", "month"]:
            for g, x in Wd.groupby(lv):
                summ.append({"metric": c, "level": lv, "group": g, "n": len(x), "mean": x[c].mean(), "se_window": x[c].std() / np.sqrt(len(x))})
        for (f, cn), x in Wd.groupby(["fold", "condition"]):
            summ.append({"metric": c, "level": "fold_condition", "group": f"{f}|{cn}", "n": len(x), "mean": x[c].mean(), "se_window": x[c].std() / np.sqrt(len(x))})
    for r in NAMES:
        for cond, grid in [("queue_onset", qc.THR_ON), ("queue_ongoing", qc.THR_OG)]:
            idx = [i for i, w in enumerate(DW) if w["condition"] == cond]
            for thr in grid:
                f, o = [], []
                for wi in idx:
                    P = pred_mat(recipe_prob(r, wi), DW[wi], thr, thr)
                    f.append(qd.iou(P, DW[wi]["yf"]))
                    o.append(qd.iou(P, DW[wi]["yo"], DW[wi]["mk"]))
                summ.append({"metric": f"{r}_thr", "level": "threshold_curve", "group": f"{cond}|{thr:.2f}", "n": len(idx), "mean": float(np.mean(f)),
                             "se_window": float(np.std(f, ddof=1) / np.sqrt(len(f))), "obs_mean": float(np.mean(o))})
    for _, rr in Cn.iterrows():
        for c in Cn.columns[1:]:
            summ.append({"metric": f"count_{c}", "level": "panel", "group": rr.panel, "n": int(rr[c])})
    for _, rr in NT.iterrows():
        summ.append({"metric": f"train_rows_{rr.model}", "level": "fold", "group": rr.fold, "n": int(rr.rows), "mean": float(rr.windows)})
    S = pd.DataFrame(summ)
    S["runtime_s"] = round(time.time() - t_start, 1)
    S.to_csv(dest / "exp3c_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.level.isin(["overall", "condition", "fold"]) & ~S.metric.str.startswith(("d_", "count", "train"))].round(4).to_string(index=False), flush=True)
    print(S[S.metric.str.startswith("d_") & S.level.isin(["overall", "condition"])].round(4).to_string(index=False), flush=True)
    print(S[(S.level == "panel") & S.metric.str.endswith("_fill") & ~S.metric.str.startswith("d_")].pivot(index="group", columns="metric", values="mean").round(4).to_string(), flush=True)
    tcv = S[S.level == "threshold_curve"].copy()
    tcv[["cond", "thr"]] = tcv.group.str.split("|", expand=True)
    print(tcv.pivot_table(index=["cond", "thr"], columns="metric", values="mean").round(4).to_string(), flush=True)

    agg = {r: qd.aggregate(W, f"{r}_fill") for r in NAMES}
    best = max(NAMES[1:], key=lambda r: agg[r])
    res = {"runtime_s": round(time.time() - t_start, 1), "v3_check": v3chk, "family_fill": agg, "best": best}
    if a.errcat == "auto" and full and agg[best] > agg["v3"]:
        s = time.time()
        ea = cache / "ea"
        ea.mkdir(parents=True, exist_ok=True)
        W[["window_id"]].assign(v3_fill=W[f"{best}_fill"]).to_csv(ea / "cv_results.csv", index=False)
        dev_m.reset_index().to_parquet(ea / "devset_windows_metrics.parquet", index=False)
        for panel, st in store.items():
            idx = W.index[W.panel == panel]
            assert list(W.window_id[idx]) == list(st["window_id"])
            mats = []
            for wi in idx:
                pr = recipe_prob(best, wi)
                m = np.full((tm.H, DW[wi]["L"]), np.nan, np.float32)
                if DW[wi]["condition"] == "queue_onset":
                    m[-1] = pr
                else:
                    for h in range(1, tm.H + 1):
                        m[h - 1] = pr[h]
                mats.append(m)
            np.savez_compressed(ea / f"{best}_{panel}.npz", **st, prob=np.stack(mats))
        qe.R = best
        E, _ = qe.analyze(ea, ea)
        C = qe.cat_table(E)
        C.to_csv(dest / "exp3c_error_categories.csv", index=False)
        old = pd.read_csv(out / "error_categories.csv")
        cmp = C[["condition", "category", "share_of_total_loss", "mean_iou_drop"]].merge(
            old[["condition", "category", "share_of_total_loss", "mean_iou_drop"]], on=["condition", "category"], suffixes=(f"_{best}", "_v3"))
        print(cmp.round(4).to_string(index=False), flush=True)
        res["errcat"] = {"max_abs_diff_vs_exp": float((E.iou - E.iou_cv).abs().max()), "runtime_s": round(time.time() - s, 1)}
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
