import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import queue_exp3c as qx
import task2_diag as td
import task2_model as tm

RECIPES = {"v3": ("on|y", "og|y", 0.35, 0.5), "e1_040": ("on|y", "og_e1|y", 0.35, 0.40), "final": ("on|y", "og_e1|yf", 0.35, 0.40)}
MODELS = [("on", "y"), ("og", "y"), ("og_e1", "y"), ("og_e1", "yf")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t_start = time.time()
    rel, out, dest = Path(a.release_root), Path(a.out), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dev_m = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["fold"] = dev.month.map(qc.fold_of)
    rng_v3, rng_e1 = np.random.default_rng(0), np.random.default_rng(1)
    syn = {k: [] for k in ["on", "og", "og_e1"]}
    FQ, FM = [[] for _ in qc.FOLDS], [[] for _ in qc.FOLDS]
    DX, DW = [], []
    off, wbase = 0, 0
    for panel in tm.PANELS:
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
        g1, c1 = qx.selector_chunks(P, D, pid, rng_e1, "og", n_og)
        for nm, ch in [("on", o), ("og", g), ("og_e1", qx.add_fill(g1, D["U"], di))]:
            df = qc.tag(ch, L, off, fold_day, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        for w in dev[dev.panel == panel].itertuples():
            d = di[w.date]
            t = int(dev_m.loc[w.window_id, "tod"])
            T0 = pd.Timestamp(w.origin)
            X = tm.features(td.hybrid(P["F"][d], P["M"][d], t), t, P["vf"], P["cap"], P["lanes"], fq[w.fold], T0.weekday(), pid)
            yf = D["U"][d][t + 1:t + tm.H + 1]
            mk = ~np.isnan(D["S"][d][t + 1:t + tm.H + 1])
            yo = D["Qo"][d][t + 1:t + tm.H + 1] & mk
            pers = np.repeat(D["Qv"][d][t - 1][None], tm.H, 0)
            DX.append(X.assign(_wi=len(DW), _g=off + t * L + np.arange(L), _L=L))
            DW.append({"window_id": w.window_id, "panel": panel, "month": w.month, "sample_id": w.sample_id, "condition": w.condition,
                       "fold": w.fold, "t0": t, "L": L, "yf": yf, "yo": yo, "mk": mk,
                       "pers_fill": qd.iou(pers, yf), "pers_obs": qd.iou(pers, yo, mk)})
        off += tm.T * L
        print(panel, {"v3_on": len(o), "v3_og": n_og, "e1_og": len(g1) // tm.H, "e1_cand": c1}, f"{time.time()-s:.0f}s", flush=True)
        del P, D
    FQ = [np.concatenate(x) for x in FQ]
    FM = [np.concatenate(x) for x in FM]
    for nm in syn:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
    print({nm: (len(df), df.win.nunique(), int(df.y.notna().sum()), int(df.yf.notna().sum()) if "yf" in df else None) for nm, df in syn.items()}, flush=True)
    DXa = pd.concat(DX, ignore_index=True)
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("yf", "yo", "mk")} for w in DW])

    probs = {f"{s}|{y}": {} for s, y in MODELS}
    for k in range(len(qc.FOLDS)):
        s = time.time()
        for nm in syn:
            qc.set_freq(syn[nm], FQ[k], FM[k])
        dk = DXa[DXa._wi.isin(W.index[W.fold == k])]
        xo = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_onset")])]
        xg = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_ongoing")])]
        Xg = pd.concat([xg.assign(h=h, freq_h=FQ[k][xg._g.to_numpy() + h * xg._L.to_numpy()]) for h in range(1, tm.H + 1)], ignore_index=True)
        for src, ycol in MODELS:
            key = f"{src}|{ycol}"
            tr = syn[src][syn[src]._f != k]
            m, cols, n = qx.fit(tr, ycol, a.rounds)
            del tr
            if src == "on":
                pr = m.predict(xo[cols], num_threads=4)
                for wi, g in pd.Series(pr).groupby(xo._wi.to_numpy()):
                    probs[key][wi] = g.to_numpy()
            else:
                pr = m.predict(Xg[cols], num_threads=4)
                for (wi, h), g in pd.Series(pr).groupby([Xg._wi.to_numpy(), Xg.h.to_numpy()]):
                    probs[key].setdefault(wi, {})[h] = g.to_numpy()
            print("fold", k, key, n, f"{time.time()-s:.0f}s", flush=True)

    def prob(r, wi):
        on, og = RECIPES[r][:2]
        return probs[on if DW[wi]["condition"] == "queue_onset" else og][wi]

    for r, (_, _, thr_on, thr_og) in RECIPES.items():
        fi, oi = [], []
        for wi, w in enumerate(DW):
            M = qx.pred_mat(prob(r, wi), w, thr_on, thr_og)
            fi.append(qd.iou(M, w["yf"]))
            oi.append(qd.iou(M, w["yo"], w["mk"]))
        W[f"{r}_fill"] = fi
        W[f"{r}_obs"] = oi
    W.to_csv(dest / "final_cv_results.csv", index=False)
    ref = pd.read_csv(out / "exp3c_results.csv").set_index("window_id")
    chk = {"v3_fill_vs_exp3c": float((W.set_index("window_id").v3_fill - ref.v3_fill.reindex(W.window_id)).abs().max()),
           "e1_obs_onset_vs_exp3c": float((W.set_index("window_id").final_obs - ref.e1_obs.reindex(W.window_id))[W.set_index("window_id").condition == "queue_onset"].abs().max())}
    print("checks", chk, flush=True)

    names = ["pers"] + list(RECIPES)
    cols = [f"{n}_{t}" for n in names for t in ["fill", "obs"]]
    Wd = W.assign(**{f"d_{x}_v3_{t}": W[f"{x}_{t}"] - W[f"v3_{t}"] for x in ["e1_040", "final"] for t in ["fill", "obs"]})
    Wd = Wd.assign(**{f"d_final_e1_040_{t}": W[f"final_{t}"] - W[f"e1_040_{t}"] for t in ["fill", "obs"]})
    dcols = [c for c in Wd.columns if c.startswith("d_")]
    bs, rel_sc = qc.boot(Wd, cols + dcols, a.boot, np.random.default_rng(0))
    summ = []
    for i, c in enumerate(cols + dcols):
        summ.append({"metric": c, "level": "overall", "group": "all", "n": len(Wd), "mean": Wd[c].mean(), "se_window": Wd[c].std() / np.sqrt(len(Wd)),
                     "family_agg": qd.aggregate(Wd, c), "se_boot80": bs[:, i].std(), "pseudo_release_mean": rel_sc[:, i].mean(),
                     "pseudo_release_std": rel_sc[:, i].std(ddof=1)})
        for lv in ["condition", "panel", "fold"]:
            for g, x in Wd.groupby(lv):
                summ.append({"metric": c, "level": lv, "group": g, "n": len(x), "mean": x[c].mean(), "se_window": x[c].std() / np.sqrt(len(x))})
    idx = [i for i, w in enumerate(DW) if w["condition"] == "queue_ongoing"]
    for thr in qc.THR_OG:
        f = [qd.iou(qx.pred_mat(prob("final", wi), DW[wi], 0.35, thr), DW[wi]["yf"]) for wi in idx]
        o = [qd.iou(qx.pred_mat(prob("final", wi), DW[wi], 0.35, thr), DW[wi]["yo"], DW[wi]["mk"]) for wi in idx]
        summ.append({"metric": "final_og_thr", "level": "threshold_curve", "group": f"{thr:.2f}", "n": len(idx), "mean": float(np.mean(f)),
                     "se_window": float(np.std(f, ddof=1) / np.sqrt(len(f))), "obs_mean": float(np.mean(o))})
    S = pd.DataFrame(summ)
    S["runtime_s"] = round(time.time() - t_start, 1)
    S.to_csv(dest / "final_cv_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[S.level.isin(["overall", "condition"])].round(4).to_string(index=False), flush=True)
    print(S[(S.level == "fold") & S.metric.str.startswith("d_final_v3")].round(4).to_string(index=False), flush=True)
    print(S[S.level == "threshold_curve"].round(4).to_string(index=False), flush=True)
    res = {"runtime_s": round(time.time() - t_start, 1), "checks": chk, "family_fill": {r: qd.aggregate(W, f"{r}_fill") for r in names},
           "family_obs": {r: qd.aggregate(W, f"{r}_obs") for r in names}}
    (dest / "final_cv_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, default=float), flush=True)


if __name__ == "__main__":
    main()
