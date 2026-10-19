import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import queue_devset as qd
import task2_diag as td
import task2_model as tm

FOLDS = [["2030-06", "2030-07", "2030-08"], ["2030-09", "2030-10", "2030-11"], ["2030-12", "2031-01", "2031-02"]]
RECIPES = {"v3": {"mode": "full", "drop": [], "thr_on": 0.35, "thr_og": 0.5},
           "v4": {"mode": "hybrid", "drop": td.FREQ_COLS, "thr_on": 0.3, "thr_og": 0.35}}
AUX = ["_g", "_f", "_L"]
LB = {"pers": 0.30, "v3": 0.72, "v4": 0.77}
THR_ON = np.round(np.arange(0.2, 0.601, 0.05), 2)
THR_OG = np.round(np.arange(0.2, 0.701, 0.05), 2)


def fold_of(month):
    return next(k for k, f in enumerate(FOLDS) if month in f)


def tag(chunks, L, off, fold_day, wbase):
    out = []
    for X in chunks:
        p, date, t0 = X.win.iloc[0].split("|")
        X = X.copy()
        X["_g"] = (off + int(t0) * L + np.arange(L)).astype(np.int64)
        X["_f"] = np.int8(fold_day[date])
        X["_L"] = np.int64(L)
        out.append(X)
    if not out:
        return None
    df = pd.concat(out, ignore_index=True)
    codes, _ = pd.factorize(df.win)
    df["win"] = codes.astype(np.int64) + wbase
    return df


def set_freq(df, fq, fm):
    df["freq_t0"] = fq[df._g.to_numpy()]
    df["freq_max_h"] = fm[df._g.to_numpy()]
    if "freq_h" in df.columns:
        df["freq_h"] = fq[df._g.to_numpy() + df.h.to_numpy() * df._L.to_numpy()]


def win_ious(df, prob, thr, top1):
    d = pd.DataFrame({"win": df.win.to_numpy(), "y": df.y.to_numpy(), "p": prob})
    d["pred"] = d.p >= thr
    if top1:
        none = ~d.groupby("win").pred.transform("any")
        top = d.index.isin(d.groupby("win").p.idxmax().to_numpy())
        d["pred"] = d.pred | (none & top)
    d = d[d.y.notna()]
    y = d.y.astype(bool)
    g = pd.DataFrame({"win": d.win, "i": d.pred & y, "u": d.pred | y}).groupby("win")[["i", "u"]].sum()
    return np.where(g.u > 0, g.i / g.u.clip(lower=1), 1.0)


def boot(W, cols, n, rng):
    keys = ["panel", "month", "sample_id"]
    G = W.groupby(keys + ["condition"])[cols].mean().unstack("condition")
    S = pd.DataFrame({c: G[c].mean(1) for c in cols})
    pools = {p: S.xs(p, level="panel").to_numpy() for p in tm.PANELS}
    fam = sorted(set(qd.FAMILY.values()))
    out = np.zeros((n, len(cols)))
    for b in range(n):
        pv = {p: pools[p][rng.integers(len(pools[p]))] for p in tm.PANELS}
        out[b] = np.mean([np.mean([pv[p] for p in tm.PANELS if qd.FAMILY[p] == f], 0) for f in fam], 0)
    rel_rows = []
    for (m, k), g in W.groupby(["month", "sample_id"]):
        if g.panel.nunique() == len(tm.PANELS):
            rel_rows.append([qd.aggregate(g, c) for c in cols])
    return out, np.array(rel_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t_start = time.time()
    rel, out = Path(a.release_root), Path(a.out)
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dev_m = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["fold"] = dev.month.map(fold_of)
    rngs = {r: np.random.default_rng(0) for r in RECIPES}
    syn = {r: {"on": [], "og": []} for r in RECIPES}
    FQ, FM = [[] for _ in FOLDS], [[] for _ in FOLDS]
    DX, DW = [], []
    off, wbase = 0, 0
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        D = qd.load_panel(rel, panel)
        assert D["dates"] == P["dates"]
        L = len(P["order"])
        months = np.array([d[:7] for d in P["dates"]])
        fold_day = {d: fold_of(d[:7]) for d in P["dates"]}
        days = list(range(len(P["dates"])))
        fq = []
        for k, f in enumerate(FOLDS):
            tr = ~np.isin(months, f)
            q = np.nan_to_num(np.nanmean(P["Q"][tr], 0)).astype(np.float32)
            m6 = np.zeros_like(q)
            for t in range(tm.T - tm.H):
                m6[t] = q[t + 1:t + tm.H + 1].max(0)
            FQ[k].append(q.ravel())
            FM[k].append(m6.ravel())
            fq.append(q)
        zero = np.zeros((tm.T, L), np.float32)
        for r, cfg in RECIPES.items():
            o, g = td.gen(P, pid, cfg["mode"], rngs[r], 2, days, lambda dt: zero)
            for nm, ch in [("on", o), ("og", g)]:
                df = tag(ch, L, off, fold_day, wbase)
                wbase = int(df.win.max()) + 1
                syn[r][nm].append(df)
        di = {d: i for i, d in enumerate(P["dates"])}
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
        off += tm.T * L
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P, D
    FQ = [np.concatenate(x) for x in FQ]
    FM = [np.concatenate(x) for x in FM]
    for r in RECIPES:
        for nm in ["on", "og"]:
            syn[r][nm] = pd.concat(syn[r][nm], ignore_index=True)
        print(r, {nm: (len(syn[r][nm]), syn[r][nm].win.nunique()) for nm in ["on", "og"]}, flush=True)
    DXa = pd.concat(DX, ignore_index=True)
    W = pd.DataFrame([{k: v for k, v in w.items() if k not in ("yf", "yo", "mk")} for w in DW])
    chk = W.set_index("window_id").join(dev_m[["persistence_fill", "persistence_obs"]])
    pers_check = {"max_abs_diff_fill": float((chk.pers_fill - chk.persistence_fill).abs().max()), "max_abs_diff_obs": float((chk.pers_obs - chk.persistence_obs).abs().max())}
    print("persistence check", pers_check, flush=True)

    probs = {r: {} for r in RECIPES}
    decomp = []
    for r, cfg in RECIPES.items():
        drop = cfg["drop"] + AUX
        for k in range(len(FOLDS)):
            s = time.time()
            models = {}
            for nm, top1, thr in [("on", True, cfg["thr_on"]), ("og", False, cfg["thr_og"])]:
                df = syn[r][nm]
                if r == "v3":
                    set_freq(df, FQ[k], FM[k])
                tr = df[df._f != k]
                te = df[df._f == k]
                m, cols = td.fit(tr, drop, a.rounds)
                models[nm] = (m, cols)
                i_tr = win_ious(tr, m.predict(tr[cols], num_threads=4), thr, top1)
                i_te = win_ious(te, m.predict(te[cols], num_threads=4), thr, top1)
                cond = "queue_onset" if nm == "on" else "queue_ongoing"
                decomp.append({"recipe": r, "fold": k, "condition": cond, "level": "train", "iou": float(i_tr.mean()), "n": len(i_tr)})
                decomp.append({"recipe": r, "fold": k, "condition": cond, "level": "train_dev", "iou": float(i_te.mean()), "n": len(i_te)})
                del tr, te
            dk = DXa[DXa._wi.isin(W.index[W.fold == k])]
            xo = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_onset")])]
            po = models["on"][0].predict(xo[models["on"][1]], num_threads=4)
            for wi, g in pd.Series(po).groupby(xo._wi.to_numpy()):
                probs[r][wi] = g.to_numpy()
            xg = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_ongoing")])]
            Xg = pd.concat([xg.assign(h=h, freq_h=FQ[k][xg._g.to_numpy() + h * xg._L.to_numpy()]) for h in range(1, tm.H + 1)], ignore_index=True)
            pg = models["og"][0].predict(Xg[models["og"][1]], num_threads=4)
            for (wi, h), g in pd.Series(pg).groupby([Xg._wi.to_numpy(), Xg.h.to_numpy()]):
                probs[r].setdefault(wi, {})[h] = g.to_numpy()
            print(r, "fold", k, f"{time.time()-s:.0f}s", flush=True)

    def pred_mat(r, wi, thr_on, thr_og):
        w = DW[wi]
        P = np.zeros((tm.H, w["L"]), bool)
        if w["condition"] == "queue_onset":
            p = probs[r][wi]
            sel = p >= thr_on
            if not sel.any():
                sel[np.argmax(p)] = True
            P[-1] = sel
        else:
            for h in range(1, tm.H + 1):
                P[h - 1] = probs[r][wi][h] >= thr_og
        return P

    for r, cfg in RECIPES.items():
        fi, oi = [], []
        for wi, w in enumerate(DW):
            P = pred_mat(r, wi, cfg["thr_on"], cfg["thr_og"])
            fi.append(qd.iou(P, w["yf"]))
            oi.append(qd.iou(P, w["yo"], w["mk"]))
        W[f"{r}_fill"] = fi
        W[f"{r}_obs"] = oi
    W.to_csv(out / "cv_results.csv", index=False)

    names = ["pers", "v3", "v4"]
    cols = [f"{n}_{t}" for n in names for t in ["fill", "obs"]]
    rng = np.random.default_rng(0)
    Wd = W.assign(**{f"d_{x}_{y}_{t}": W[f"{x}_{t}"] - W[f"{y}_{t}"] for x, y in [("v3", "pers"), ("v4", "pers"), ("v4", "v3")] for t in ["fill", "obs"]})
    dcols = [c for c in Wd.columns if c.startswith("d_")]
    bs, rel_sc = boot(Wd, cols + dcols, a.boot, rng)
    summ = []
    for i, c in enumerate(cols + dcols):
        base = {"metric": c, "level": "overall", "group": "all", "n": len(Wd), "mean": Wd[c].mean(), "se_window": Wd[c].std() / np.sqrt(len(Wd)),
                "family_agg": qd.aggregate(Wd, c), "se_boot80": bs[:, i].std(), "pseudo_release_mean": rel_sc[:, i].mean(), "pseudo_release_std": rel_sc[:, i].std(ddof=1)}
        summ.append(base)
        for lv in ["condition", "panel", "fold", "month"]:
            for g, x in Wd.groupby(lv):
                summ.append({"metric": c, "level": lv, "group": g, "n": len(x), "mean": x[c].mean(), "se_window": x[c].std() / np.sqrt(len(x))})
        for (f, cn), x in Wd.groupby(["fold", "condition"]):
            summ.append({"metric": c, "level": "fold_condition", "group": f"{f}|{cn}", "n": len(x), "mean": x[c].mean(), "se_window": x[c].std() / np.sqrt(len(x))})
    S = pd.DataFrame(summ)
    S.to_csv(out / "cv_summary.csv", index=False)

    Dc = pd.DataFrame(decomp)
    ed = []
    for r in RECIPES:
        for cond in ["queue_onset", "queue_ongoing", "all"]:
            row = {"recipe": r, "condition": cond}
            for lv in ["train", "train_dev"]:
                x = Dc[(Dc.recipe == r) & (Dc.level == lv)]
                if cond != "all":
                    x = x[x.condition == cond]
                    row[lv] = float(np.average(x.iou, weights=x.n))
                else:
                    row[lv] = float(np.mean([np.average(y.iou, weights=y.n) for _, y in x.groupby("condition")]))
                row[f"{lv}_fold_std"] = float(x.groupby("fold").apply(lambda y: np.average(y.iou, weights=y.n)).std()) if cond != "all" else np.nan
            x = W if cond == "all" else W[W.condition == cond]
            row["dev_fill"] = float(x[f"{r}_fill"].mean())
            row["dev_obs"] = float(x[f"{r}_obs"].mean())
            row["dev_family_fill"] = qd.aggregate(W, f"{r}_fill") if cond == "all" else np.nan
            row["lb_est"] = LB[r] if cond == "all" else np.nan
            row["gap_bias_vs_1"] = 1 - row["train"]
            row["gap_variance"] = row["train"] - row["train_dev"]
            row["gap_mismatch"] = row["train_dev"] - row["dev_fill"]
            row["gap_dev_lb"] = row["dev_family_fill"] - row["lb_est"] if cond == "all" else np.nan
            ed.append(row)
    E = pd.DataFrame(ed)
    E.to_csv(out / "error_decomp.csv", index=False)

    tc = []
    for cond, grid, key in [("queue_onset", THR_ON, "thr_on"), ("queue_ongoing", THR_OG, "thr_og")]:
        idx = [i for i, w in enumerate(DW) if w["condition"] == cond]
        for thr in grid:
            f, o = [], []
            for wi in idx:
                P = pred_mat("v4", wi, thr, thr)
                f.append(qd.iou(P, DW[wi]["yf"]))
                o.append(qd.iou(P, DW[wi]["yo"], DW[wi]["mk"]))
            tc.append({"condition": cond, "thr": float(thr), "iou_fill": float(np.mean(f)), "iou_obs": float(np.mean(o)),
                       "se_fill": float(np.std(f, ddof=1) / np.sqrt(len(f))), "current": thr == RECIPES["v4"][key], "n": len(idx)})
    TC = pd.DataFrame(tc)
    TC.to_csv(out / "threshold_curve.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, cond in enumerate(["queue_onset", "queue_ongoing"]):
        x = TC[TC.condition == cond]
        ax[i].errorbar(x.thr, x.iou_fill, yerr=x.se_fill, marker="o", label="fill truth")
        ax[i].plot(x.thr, x.iou_obs, marker="s", label="obs truth")
        cur = RECIPES["v4"]["thr_on" if cond == "queue_onset" else "thr_og"]
        ax[i].axvline(cur, color="gray", ls="--", label=f"mevcut {cur}")
        ax[i].set_title(f"v4 {cond} dev IoU vs esik")
        ax[i].set_xlabel("esik")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "threshold_curve.png", dpi=110)

    dev_all = {n: qd.aggregate(W, f"{n}_fill") for n in names}
    rho = spearmanr([dev_all[n] for n in names], [LB[n] for n in names]).correlation
    res = {"runtime_s": round(time.time() - t_start, 1), "pers_check": pers_check, "dev_family_fill": dev_all, "lb_est": LB, "spearman_n3": float(rho)}
    (out / "cv_summary.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, default=float), flush=True)
    print(S[S.level.isin(["overall", "condition", "fold"]) & ~S.metric.str.startswith("d_")].round(4).to_string(index=False), flush=True)
    print(S[(S.level == "overall") & S.metric.str.startswith("d_")].round(4).to_string(index=False), flush=True)
    print(S[(S.level == "panel") & S.metric.str.endswith("_fill") & ~S.metric.str.startswith("d_")].pivot(index="group", columns="metric", values="mean").round(4).to_string(), flush=True)
    print(E.round(4).to_string(index=False), flush=True)
    print(TC.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
