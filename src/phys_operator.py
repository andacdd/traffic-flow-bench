import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_interp import T
from task1_unet import SCORE_PANELS


def link_params(P):
    g = P.geo
    lanes = np.clip(np.nan_to_num(g["lanes"], nan=np.nanmedian(g["lanes"])), 1, None)
    vf = np.nan_to_num(g["free_speed_kmh"], nan=117.0)
    cap = np.nan_to_num(g["capacity_vph"], nan=np.nanmedian(g["capacity_vph"]))
    kc = np.nan_to_num(g["critical_density"], nan=np.nanmedian(g["critical_density"]))
    kj = np.nan_to_num(g["k_jam"], nan=np.nanmedian(g["k_jam"]))
    length = np.nan_to_num(g["length_km"], nan=np.nanmedian(g["length_km"]))
    return lanes, vf, cap, kc, kj, length


def stage1(v, q, c, prm, beta=0.70, fq=0.95):
    lanes, vf, cap, kc, kj, _ = prm
    ln = lanes[c]
    kcl, kjl, cpl = kc[c] / ln, kj[c] / ln, cap[c] / ln
    w = cpl / np.maximum(kjl - kcl, 1e-9)
    kl = q / np.maximum(v, 1.0) / ln
    qc = w * np.maximum(kjl - kl, 0.0) * ln
    qs = np.minimum(cpl, v * w * kjl / np.maximum(v + w, 1e-9)) * ln
    tgt = (1 - beta) * qc + beta * qs
    return v, np.where(kl > kcl, (1 - fq) * q + fq * tgt, q)


def stage2(v, q, d, t, c, ov, oq, prm, alpha=0.78):
    length = prm[5]
    obs = np.isfinite(ov) & np.isfinite(oq) & (ov > 1.0)
    Kobs = np.where(obs, oq / np.maximum(ov, 1.0), np.nan)
    idx = np.arange(T)[None, :, None]
    prev = np.maximum.accumulate(np.where(obs, idx, -1), axis=1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(obs, idx, T), 1), axis=1), 1)
    p, n = prev[d, t, c], nxt[d, t, c]
    has = (p >= 0) & (n < T)
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    fr = (t - pp) / np.maximum(nn - pp, 1)
    ki = Kobs[d, pp, c] * (1 - fr) + Kobs[d, nn, c] * fr
    run = np.where(has, nn - pp - 1, 10 ** 6)
    ki = np.where(has, ki, np.nan)
    kh = q / np.maximum(v, 1.0)
    ok = np.isfinite(ki) & (ki > 0) & (kh > 0) & (run <= 4)
    wn = np.where(run == 1, 0.14, np.where(run == 2, 0.08, 0.04))
    lb = np.log(np.maximum(kh, 1e-6))
    la = np.where(ok, (1 - wn) * lb + wn * np.log(np.maximum(ki, 1e-6)), lb)
    LK = np.log(np.maximum(Kobs, 1e-6))
    LK[d, t, c] = la
    isobs = obs.copy()
    isobs[d, t, c] = False
    hp = t > 0
    hn = t < T - 1
    tp, tn = np.clip(t - 1, 0, T - 1), np.clip(t + 1, 0, T - 1)
    hp &= np.isfinite(LK[d, tp, c])
    hn &= np.isfinite(LK[d, tn, c])
    op, on = hp & isobs[d, tp, c], hn & isobs[d, tn, c]
    both = op & on
    wp = np.where(~hp, 0.0, np.where(both, 0.06, np.where(op, 0.04, 0.05)))
    wq = np.where(~hn, 0.0, np.where(both, 0.06, np.where(on, 0.04, 0.05)))
    cur = LK[d, t, c]
    lp = np.where(hp, LK[d, tp, c], cur)
    lnx = np.where(hn, LK[d, tn, c], cur)
    fin = cur + wp * (lp - cur) + wq * (lnx - cur)
    dt = np.clip(fin - lb, -0.35, 0.35)
    return np.clip(v * np.exp(-(1 - alpha) * dt), 5.0, 132.0), np.clip(q * np.exp(alpha * dt), 0.0, 150000.0)


def stage_n(v, q, d, t, c, ov, oq, prm, w, alpha, max_run):
    length = prm[5]
    obs = np.isfinite(ov) & np.isfinite(oq)
    N = np.where(obs, oq / np.maximum(ov, 1.0), np.nan)
    idx = np.arange(T)[None, :, None]
    prev = np.maximum.accumulate(np.where(obs, idx, -1), axis=1)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(obs, idx, T), 1), axis=1), 1)
    p, n = prev[d, t, c], nxt[d, t, c]
    has = (p >= 0) & (n < T)
    pp, nn = np.maximum(p, 0), np.minimum(n, T - 1)
    fr = (t - pp) / np.maximum(nn - pp, 1)
    ki = np.where(has, N[d, pp, c] * (1 - fr) + N[d, nn, c] * fr, np.nan)
    run = np.where(has, nn - pp - 1, 10 ** 6)
    kh = q / np.maximum(v, 1.0)
    ok = np.isfinite(ki) & (ki > 0) & (kh > 0) & (run <= max_run)
    dl = np.where(ok, np.log(np.maximum(ki, 1e-9)) - np.log(np.maximum(kh, 1e-9)), 0.0) * w
    return np.clip(v * np.exp(-(1 - alpha) * dl), 1, 180), np.clip(q * np.exp(alpha * dl), 0, 150000)


def stage3(v, q, c, prm, gf=0.035, gc=0.120):
    lanes, vf, cap, kc, kj, _ = prm
    ln = lanes[c]
    kcl, kjl, cpl = kc[c] / ln, kj[c] / ln, cap[c] / ln
    w = cpl / np.maximum(kjl - kcl, 1e-9)
    kl = q / np.maximum(v, 1.0) / ln
    ql = np.maximum(q / ln, 1e-6)
    qfd = np.where(kl <= kcl, vf[c] * kl, w * np.maximum(kjl - kl, 0.0))
    g = np.where(kl <= kcl, gf, gc)
    gam = (1 - g) + g * np.clip(qfd / ql, 0.8, 1.2)
    vc = v * gam
    ok = (vc >= 5.0) & (vc <= 132.0)
    return np.where(ok, vc, v), np.where(ok, np.clip(q * gam, 0.0, 150000.0), q)


def apply(P, R, v, q, stages, beta=0.70, fq=0.95):
    prm = link_params(P)
    d, t, c = P.td[R], P.tt[R], P.tl[R]
    v, q = v.astype(float), q.astype(float)
    if "1" in stages:
        v, q = stage1(v, q, c, prm, beta, fq)
    if "2" in stages:
        v, q = stage2(v, q, d, t, c, P.obs["v"], P.obs["q"], prm)
    if "3" in stages:
        v, q = stage3(v, q, c, prm)
    return v, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--le-cache", default="cache/task1_resid2")
    ap.add_argument("--out", default="reports/phys/physop_f1")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--base", default="v33q20")
    ap.add_argument("--variants", nargs="+", default=["1", "2", "3", "123"])
    ap.add_argument("--fq", nargs="+", type=float, default=[])
    ap.add_argument("--beta", nargs="+", type=float, default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--nscan", action="store_true")
    a = ap.parse_args()
    rel, lec, out = Path(a.release_root), Path(a.le_cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in SCORE_PANELS:
        P = tr.Panel(rel, p)
        b = np.load(lec / f"{p}_f{a.fold}_UW_pred.npz")
        R, v0, q0 = b["R"], b[f"{a.base}_v"], b[f"{a.base}_q"]
        o = {"R": R, "base_v": v0, "base_q": q0}
        for s in a.variants:
            v, q = apply(P, R, v0, q0, s)
            v, q = tr.finalize(P.vclip[P.tl[R]], v, q)
            o[f"op{s}_v"], o[f"op{s}_q"] = v.astype(np.float32), q.astype(np.float32)
        for fq in a.fq:
            for be in a.beta:
                v, q = apply(P, R, v0, q0, "1", be, fq)
                v, q = tr.finalize(P.vclip[P.tl[R]], v, q)
                n = f"f{int(round(fq*100))}b{int(round(be*100))}"
                o[f"{n}_v"], o[f"{n}_q"] = v.astype(np.float32), q.astype(np.float32)
        if a.nscan:
            prm = link_params(P)
            d, t, c = P.td[R], P.tt[R], P.tl[R]
            v1, q1 = apply(P, R, v0, q0, "1")
            for w in (0.5, 1.0):
                for al in (0.0, 0.5, 1.0):
                    for mr in (2, 999):
                        v, q = stage_n(v1, q1, d, t, c, P.obs["v"], P.obs["q"], prm, w, al, mr)
                        v, q = tr.finalize(P.vclip[P.tl[R]], v, q)
                        n = f"n{int(w*100)}a{int(al*100)}r{mr}"
                        o[f"{n}_v"], o[f"{n}_q"] = v.astype(np.float32), q.astype(np.float32)
        np.savez(lec / f"{p}_f{a.fold}_PO{a.tag}_pred.npz", **o)
        rows += tr.score_panel(rel, P, [a.fold], [f"PO{a.tag}"], lec)
        del P
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out / f"physop{a.tag}_cv.csv", index=False)
    g = df.groupby("variant")[["S_state", "S_FD", "S_LWR", "S_physics", "rmse_v", "rmse_q_lane"]].mean()
    g["J"] = 0.35 * g.S_state + 0.15 * g.S_physics
    g["dJ"] = g.J - g.loc["base", "J"]
    print(g.sort_values("J", ascending=False).round(5).to_string(), flush=True)
    u = df.groupby(["variant", "panel"])[["S_state", "S_physics"]].mean()
    u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
    print(u.J.unstack(0).sub(u.J.unstack(0)["base"], axis=0).round(5).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
