import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import task2_diag as td
import task2_model as tm

R = "v3"
CFG = qc.RECIPES[R]
NEAR_V = 5.0
NEAR_P = 0.1
WRONG_K = 5
WRONG_J = 0.2
BOTTLENECK_Q = 0.8
DROP_FRAC = 0.4
WORST = 100
CATS = ["onset_timing", "extent_under", "extent_over", "dissipation_early", "dissipation_late", "wrong_link", "threshold_effect"]


def build_preds(rel, out, cache, rounds):
    dev = pd.read_parquet(out / "devset_windows.parquet")
    dev_m = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    dev["fold"] = dev.month.map(qc.fold_of)
    rng = np.random.default_rng(0)
    syn = {"on": [], "og": []}
    FQ, FM = [[] for _ in qc.FOLDS], [[] for _ in qc.FOLDS]
    DX, DW, store = [], [], {}
    off, wbase = 0, 0
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        D = qd.load_panel(rel, panel)
        assert D["dates"] == P["dates"]
        L = len(P["order"])
        months = np.array([d[:7] for d in P["dates"]])
        fold_day = {d: qc.fold_of(d[:7]) for d in P["dates"]}
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
        o, g = td.gen(P, pid, CFG["mode"], rng, 2, list(range(len(P["dates"]))), lambda dt: zero)
        for nm, ch in [("on", o), ("og", g)]:
            df = qc.tag(ch, L, off, fold_day, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        di = {d: i for i, d in enumerate(P["dates"])}
        lf = np.nan_to_num(np.nanmean(P["Q"], (0, 1))).astype(np.float32)
        ids, S, Y, UT = [], [], [], []
        for w in dev[dev.panel == panel].itertuples():
            d = di[w.date]
            t = int(dev_m.loc[w.window_id, "tod"])
            T0 = pd.Timestamp(w.origin)
            X = tm.features(td.hybrid(P["F"][d], P["M"][d], t), t, P["vf"], P["cap"], P["lanes"], fq[w.fold], T0.weekday(), pid)
            DX.append(X.assign(_wi=len(DW), _g=off + t * L + np.arange(L), _L=L))
            DW.append({"window_id": w.window_id, "panel": panel, "condition": w.condition, "fold": w.fold, "L": L})
            ids.append(w.window_id)
            S.append(D["S"][d][t - tm.HIST:t + tm.H + 1])
            Y.append(D["U"][d][t + 1:t + tm.H + 1])
            UT.append(D["U"][d][t - tm.HIST:t + 1])
        store[panel] = {"window_id": np.array(ids), "speed": np.stack(S).astype(np.float32), "yf": np.stack(Y), "u_hist": np.stack(UT),
                        "vc": D["vc"], "vf": P["vf"], "link_qfreq": lf}
        off += tm.T * L
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P, D
    FQ = [np.concatenate(x) for x in FQ]
    FM = [np.concatenate(x) for x in FM]
    for nm in ["on", "og"]:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
    DXa = pd.concat(DX, ignore_index=True)
    W = pd.DataFrame(DW)
    prob = {}
    drop = CFG["drop"] + qc.AUX
    for k in range(len(qc.FOLDS)):
        s = time.time()
        models = {}
        for nm in ["on", "og"]:
            df = syn[nm]
            qc.set_freq(df, FQ[k], FM[k])
            models[nm] = td.fit(df[df._f != k], drop, rounds)
        dk = DXa[DXa._wi.isin(W.index[W.fold == k])]
        xo = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_onset")])]
        po = models["on"][0].predict(xo[models["on"][1]], num_threads=4)
        for wi, g in pd.Series(po).groupby(xo._wi.to_numpy()):
            m = np.full((tm.H, W.L[wi]), np.nan, np.float32)
            m[-1] = g.to_numpy()
            prob[wi] = m
        xg = dk[dk._wi.isin(W.index[(W.fold == k) & (W.condition == "queue_ongoing")])]
        Xg = pd.concat([xg.assign(h=h, freq_h=FQ[k][xg._g.to_numpy() + h * xg._L.to_numpy()]) for h in range(1, tm.H + 1)], ignore_index=True)
        pg = models["og"][0].predict(Xg[models["og"][1]], num_threads=4)
        for (wi, h), g in pd.Series(pg).groupby([Xg._wi.to_numpy(), Xg.h.to_numpy()]):
            prob.setdefault(wi, np.full((tm.H, W.L[wi]), np.nan, np.float32))[h - 1] = g.to_numpy()
        print(R, "fold", k, f"{time.time()-s:.0f}s", flush=True)
    cache.mkdir(parents=True, exist_ok=True)
    for panel, st in store.items():
        idx = W.index[W.panel == panel]
        assert list(W.window_id[idx]) == list(st["window_id"])
        st["prob"] = np.stack([prob[i] for i in idx])
        np.savez_compressed(cache / f"{R}_{panel}.npz", **st)


def pred_mat(prob, cond):
    P = np.zeros(prob.shape, bool)
    if cond == "queue_onset":
        p = prob[-1]
        sel = p >= CFG["thr_on"]
        if not sel.any():
            sel[np.argmax(p)] = True
        P[-1] = sel
    else:
        P = prob >= CFG["thr_og"]
    return P


def centroid(m):
    x = np.where(m)[0]
    return x.mean() if len(x) else np.nan


def classify(P, Y, prob, sp, vc, uT, cond):
    thr = CFG["thr_on"] if cond == "queue_onset" else CFG["thr_og"]
    err = P ^ Y
    lab = np.full(P.shape, -1, np.int8)
    near = (np.abs(sp - vc) <= NEAR_V) | (np.abs(np.nan_to_num(prob, nan=-9) - thr) <= NEAR_P)
    if cond == "queue_onset":
        rows = slice(tm.H - 1, tm.H)
        lab[:-1][err[:-1]] = 0
        if not Y[-1].any():
            lab[-1][err[-1]] = 0
        pl, yl = P[-1], Y[-1]
    else:
        rows = slice(0, tm.H)
        pl, yl = P.any(0), Y.any(0)
    j = (pl & yl).sum() / max((pl | yl).sum(), 1)
    dist = abs(centroid(pl) - centroid(yl)) if pl.any() and yl.any() else np.nan
    wrong = bool(j < WRONG_J and dist > WRONG_K)
    free = (lab == -1) & err
    nr = np.zeros(P.shape, bool)
    nr[rows] = True
    free &= nr
    th = free & near
    lab[th] = 6
    free &= ~th
    if wrong:
        lab[free] = 5
        free[:] = False
    fn, fp = free & Y, free & P
    if cond == "queue_ongoing":
        pred_prev = np.vstack([np.zeros((1, P.shape[1]), bool), np.cumsum(P, 0)[:-1] > 0])
        true_prev = np.vstack([uT[None], np.cumsum(Y, 0)[:-1] > 0])
        de = fn & (uT[None] | pred_prev)
        dl = fp & true_prev
        lab[de] = 3
        lab[dl] = 4
        fn &= ~de
        fp &= ~dl
    lab[fn] = 1
    lab[fp] = 2
    lab_nt = lab.copy()
    if th.any():
        alt = classify_no_thr(P, Y, uT, cond, wrong, th)
        lab_nt[th] = alt[th]
    return lab, lab_nt, j, dist, wrong


def classify_no_thr(P, Y, uT, cond, wrong, cells):
    lab = np.full(P.shape, -1, np.int8)
    if wrong:
        lab[cells] = 5
        return lab
    fn, fp = cells & Y, cells & P
    if cond == "queue_ongoing":
        pred_prev = np.vstack([np.zeros((1, P.shape[1]), bool), np.cumsum(P, 0)[:-1] > 0])
        true_prev = np.vstack([uT[None], np.cumsum(Y, 0)[:-1] > 0])
        de = fn & (uT[None] | pred_prev)
        dl = fp & true_prev
        lab[de] = 3
        lab[dl] = 4
        fn &= ~de
        fp &= ~dl
    lab[fn] = 1
    lab[fp] = 2
    return lab


def incident(sp, vf, vc, nonbn):
    with np.errstate(all="ignore"):
        s = pd.DataFrame(sp).ffill().to_numpy()
        d1 = s[:-1] - s[1:]
        d2 = s[:-2] - s[2:]
        drop1 = (d1 >= DROP_FRAC * vf) & (s[1:] <= vc)
        drop2 = (d2 >= DROP_FRAC * vf) & (s[2:] <= vc)
    ev = np.zeros(sp.shape, bool)
    ev[1:] |= drop1
    ev[2:] |= drop2
    ev &= nonbn[None]
    h = tm.HIST + 1
    return bool(ev[:h].any()), bool(ev[h:].any())


def analyze(out, cache):
    cv = pd.read_csv(out / "cv_results.csv").set_index("window_id")
    dm = pd.read_parquet(out / "devset_windows_metrics.parquet").set_index("window_id")
    rows, cellrows, mats = [], [], {}
    for panel in tm.PANELS:
        z = np.load(cache / f"{R}_{panel}.npz")
        vc, vf, lf = z["vc"], z["vf"], z["link_qfreq"]
        nonbn = lf < np.quantile(lf, BOTTLENECK_Q)
        for i, wid in enumerate(z["window_id"]):
            cond = dm.loc[wid, "condition"]
            prob, Y, sp, uh = z["prob"][i], z["yf"][i], z["speed"][i], z["u_hist"][i]
            P = pred_mat(prob, cond)
            spH = sp[tm.HIST + 1:]
            uT = uh[-1]
            lab, lab_nt, j, dist, wrong = classify(P, Y, prob, spH, vc, uT, cond)
            u = (P | Y).sum()
            iou = 1.0 if u == 0 else (P & Y).sum() / u
            err = P ^ Y
            assert (lab[err] >= 0).all()
            share = {c: float((lab == k).sum() / u) if u else 0.0 for k, c in enumerate(CATS)}
            share_nt = {c: float((lab_nt == k).sum() / u) if u else 0.0 for k, c in enumerate(CATS)}
            miss = np.isnan(spH)
            share_na = {c: float(((lab == k) & miss).sum() / u) if u else 0.0 for k, c in enumerate(CATS)}
            cnt = np.array([(lab == k).sum() for k in range(len(CATS))])
            order = np.argsort(-cnt, kind="stable")
            prim = CATS[order[0]] if cnt.sum() else "none"
            sec = CATS[order[1]] if cnt.sum() and cnt[order[1]] >= 0.25 * cnt.sum() else ""
            inc_h, inc_f = incident(sp, vf, vc, nonbn)
            ut_cells = int(uT.sum())
            pmax = float(np.nanmax(prob))
            ptrue = float(np.nanmax(np.where(Y, prob, np.nan))) if (Y & ~np.isnan(prob)).any() else np.nan
            row = {"window_id": wid, "panel": panel, "month": dm.loc[wid, "month"], "condition": cond, "tod": int(dm.loc[wid, "tod"]), "hour": int(dm.loc[wid, "tod"]) // 12,
                   "iou": float(iou), "loss": float(1 - iou), "iou_cv": float(cv.loc[wid, "v3_fill"]), "union": int(u), "tp": int((P & Y).sum()),
                   "fp": int((P & ~Y).sum()), "fn": int((~P & Y).sum()), "precision": float((P & Y).sum() / max(P.sum(), 1)), "recall": float((P & Y).sum() / max(Y.sum(), 1)),
                   "pred_cells": int(P.sum()), "hist_q_cells": int(dm.loc[wid, "hist_q_cells"]), "hz_cells": int(Y.sum()), "hz_links": int(Y.any(0).sum()),
                   "q_links_T": ut_cells, "first_q_step": int(dm.loc[wid, "first_q_step"]), "truth_s1": int(Y[0].sum()), "truth_s6": int(Y[-1].sum()),
                   "growth": int(Y[-1].sum()) - ut_cells, "link_jaccard": float(j), "centroid_dist": float(dist) if dist == dist else np.nan, "wrong_flag": wrong,
                   "near_thr_err_frac": float(((lab == 6).sum()) / max(err.sum(), 1)), "incident_hist": inc_h, "incident_hz": inc_f,
                   "prob_max": pmax, "prob_max_truth": ptrue, "onset_fallback": bool(cond == "queue_onset" and pmax < CFG["thr_on"]),
                   "primary": prim, "secondary": sec, **{f"loss_{c}": share[c] for c in CATS}, **{f"lossnt_{c}": share_nt[c] for c in CATS}, **{f"lossna_{c}": share_na[c] for c in CATS}}
            rows.append(row)
            mats[wid] = (P, Y, prob, lab)
    E = pd.DataFrame(rows)
    E["dir"] = np.where(E.growth > 0, "buyuyor", np.where(E.growth < 0, "kuculuyor", "sabit"))
    E["worst100"] = E.index.isin(E.sort_values(["loss", "union"], ascending=False, kind="stable").index[:WORST])
    return E, mats


def cat_table(E):
    out = []
    tot = E.loss.sum()
    for cond in ["all", "queue_onset", "queue_ongoing"]:
        x = E if cond == "all" else E[E.condition == cond]
        w = x[x.worst100]
        for c in CATS:
            out.append({"condition": cond, "category": c, "worst100_primary": int((w.primary == c).sum()), "worst100_secondary": int((w.secondary == c).sum()),
                        "worst100_loss_share": float(w[f"loss_{c}"].sum() / max(w.loss.sum(), 1e-9)),
                        "loss_sum": float(x[f"loss_{c}"].sum()), "share_of_total_loss": float(x[f"loss_{c}"].sum() / tot),
                        "share_within_condition": float(x[f"loss_{c}"].sum() / x.loss.sum()),
                        "share_within_condition_no_thr": float(x[f"lossnt_{c}"].sum() / x.loss.sum()),
                        "unobserved_truth_frac": float(x[f"lossna_{c}"].sum() / max(x[f"loss_{c}"].sum(), 1e-9)),
                        "mean_iou_drop": float(x[f"loss_{c}"].sum() / len(x)), "n_windows": len(x), "n_worst100": len(w)})
    return pd.DataFrame(out)


def descriptors(E):
    out = []
    num = ["prob_max", "prob_max_truth", "hour", "hist_q_cells", "hz_cells", "hz_links", "q_links_T", "growth", "first_q_step", "union", "pred_cells", "near_thr_err_frac"]
    flags = {"incident_hist": E.incident_hist, "incident_hz": E.incident_hz, "buyuyor": E.dir == "buyuyor", "kuculuyor": E.dir == "kuculuyor",
             "onset": E.condition == "queue_onset", "am": E.hour < 12, "onset_fallback": E.onset_fallback}
    for cond in ["all", "queue_onset", "queue_ongoing"]:
        x = E if cond == "all" else E[E.condition == cond]
        a, b = x[x.worst100], x[~x.worst100]
        if len(a) < 5:
            continue
        for c in num:
            sd = x[c].std()
            out.append({"condition": cond, "descriptor": c, "kind": "mean", "worst": a[c].mean(), "rest": b[c].mean(), "worst_median": a[c].median(), "rest_median": b[c].median(),
                        "std_diff": (a[c].mean() - b[c].mean()) / sd if sd > 0 else 0.0, "n_worst": len(a), "n_rest": len(b)})
        for c, f in flags.items():
            f = f[x.index]
            p = f.mean()
            out.append({"condition": cond, "descriptor": c, "kind": "share", "worst": f[a.index].mean(), "rest": f[b.index].mean(),
                        "std_diff": (f[a.index].mean() - f[b.index].mean()) / np.sqrt(p * (1 - p)) if 0 < p < 1 else 0.0, "n_worst": len(a), "n_rest": len(b)})
        for lv in ["panel", "month"]:
            for g in sorted(x[lv].unique()):
                f = x[lv] == g
                p = f.mean()
                out.append({"condition": cond, "descriptor": f"{lv}={g}", "kind": "share", "worst": f[a.index].mean(), "rest": f[b.index].mean(),
                            "std_diff": (f[a.index].mean() - f[b.index].mean()) / np.sqrt(p * (1 - p)) if 0 < p < 1 else 0.0, "n_worst": len(a), "n_rest": len(b)})
    return pd.DataFrame(out)


def plots(E, C, mats, out):
    lab = {"queue_onset": "onset", "queue_ongoing": "ongoing"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(CATS))
    for i, cond in enumerate(["queue_onset", "queue_ongoing"]):
        c = C[C.condition == cond].set_index("category").loc[CATS]
        ax[0].bar(x + (i - 0.5) * 0.4, c.share_of_total_loss, 0.4, label=lab[cond])
        ax[1].bar(x + (i - 0.5) * 0.4, c.worst100_primary, 0.4, label=lab[cond])
    for a, t in zip(ax, ["Toplam IoU kaybinin payi (tum dev, 3360 pencere)", "En kotu 100 pencere: birincil kategori sayisi"]):
        a.set_xticks(x)
        a.set_xticklabels(CATS, rotation=30, ha="right")
        a.set_title(t)
        a.legend()
    fig.tight_layout()
    fig.savefig(out / "error_loss_share.png", dpi=110)
    plt.close(fig)
    pick = []
    w = E[E.worst100].sort_values("loss", ascending=False)
    for c in ["onset_timing", "extent_under", "extent_over", "dissipation_early", "dissipation_late", "wrong_link"]:
        g = w[(w.primary == c) & ~w.window_id.isin(pick)]
        if len(g):
            pick.append(g.window_id.iloc[0])
    cmap = ListedColormap(["#f2f2f2", "#2b8a3e", "#e8590c", "#1c7ed6"])
    fig, ax = plt.subplots(len(pick), 1, figsize=(12, 1.6 * len(pick) + 0.6), squeeze=False)
    for a, wid in zip(ax[:, 0], pick):
        P, Y, _, _ = mats[wid]
        m = np.zeros(P.shape, int)
        m[P & Y], m[P & ~Y], m[~P & Y] = 1, 2, 3
        cols = np.where((P | Y).any(0))[0]
        lo, hi = (max(cols.min() - 10, 0), min(cols.max() + 11, P.shape[1])) if len(cols) else (0, P.shape[1])
        a.imshow(m[:, lo:hi], cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest", extent=[lo - 0.5, hi - 0.5, 6.5, 0.5])
        r = E.set_index("window_id").loc[wid]
        a.set_title(f"{wid} | {lab[r.condition]} | IoU {r.iou:.2f} | birincil {r.primary}", fontsize=9)
        a.set_ylabel("adim (T+5k)")
        a.set_yticks(range(1, 7))
    ax[-1, 0].set_xlabel("link sirasi  (yesil TP, turuncu FP tahmin fazla, mavi FN kacan)")
    fig.tight_layout()
    fig.savefig(out / "error_examples.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.arange(0, 25)
    for nm, f in [("en kotu 100", E.worst100), ("diger", ~E.worst100)]:
        ax[0].hist(E[f].hour, bins=bins, density=True, alpha=0.5, label=nm)
        ax[1].hist(np.log10(1 + E[f].hz_cells), bins=20, density=True, alpha=0.5, label=nm)
        ax[2].hist(np.log10(1 + E[f].hist_q_cells), bins=20, density=True, alpha=0.5, label=nm)
    for a, t in zip(ax, ["origin saati (UTC)", "log10(1+ufuk kuyruk hucresi)", "log10(1+gecmis kuyruk hucresi)"]):
        a.set_title(t)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "error_descriptors.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--refit", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    out, cache = Path(a.out), Path(a.cache)
    if a.refit or not all((cache / f"{R}_{p}.npz").exists() for p in tm.PANELS):
        build_preds(Path(a.release_root), out, cache, a.rounds)
    E, mats = analyze(out, cache)
    chk = float((E.iou - E.iou_cv).abs().max())
    E.to_csv(out / "error_analysis.csv", index=False)
    C = cat_table(E)
    C.to_csv(out / "error_categories.csv", index=False)
    Dd = descriptors(E)
    Dd.to_csv(out / "error_descriptors.csv", index=False)
    plots(E, C, mats, out)
    res = {"runtime_s": round(time.time() - t0, 1), "max_abs_diff_vs_cv": chk, "mean_iou": float(E.iou.mean()),
           "worst100_iou_max": float(E[E.worst100].iou.max()), "worst100_onset": int((E[E.worst100].condition == "queue_onset").sum())}
    print(json.dumps(res), flush=True)
    pd.set_option("display.width", 250)
    print(C.round(4).to_string(index=False), flush=True)
    print(Dd.reindex(Dd.std_diff.abs().sort_values(ascending=False).index).head(40).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
