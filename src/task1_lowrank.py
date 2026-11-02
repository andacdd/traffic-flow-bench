import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_unet import SCORE_PANELS, Grid


def soft_impute(X, obs, r, iters):
    F = np.where(obs, X, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(F, full_matrices=False)
        Lr = (U[:, :r] * s[:r]) @ Vt[:r]
        F = np.where(obs, X, Lr)
    return Lr


def complete(o, prof, unfold, r, iters):
    D, T, L = o.shape
    Z = o - prof
    obs = np.isfinite(Z)
    Z = np.where(obs, Z, 0.0)
    if unfold == "d":
        Lr = soft_impute(Z.reshape(D, T * L), obs.reshape(D, T * L), r, iters).reshape(D, T, L)
    else:
        Lr = soft_impute(Z.reshape(D * T, L), obs.reshape(D * T, L), r, iters).reshape(D, T, L)
    return prof + Lr


def grid_pred(g, months, hold_months, unfold, r, iters):
    V = np.full(g.ov.shape, np.nan, np.float32)
    Q = np.full(g.ov.shape, np.nan, np.float32)
    for m in hold_months:
        di = np.flatnonzero(months == m)
        pv = np.nan_to_num(g.pv[g.dow[di]].astype(np.float64))
        pq = np.nan_to_num(g.pq[g.dow[di]].astype(np.float64))
        V[di] = complete(g.ov[di].astype(np.float64), pv, unfold, r, iters)
        Q[di] = complete(g.oq[di].astype(np.float64), pq, unfold, r, iters)
    return V, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--grid-dir", default="cache/task1_unet")
    ap.add_argument("--le-cache", default="cache/task1_resid2")
    ap.add_argument("--out", default="reports/phys/lowrank_f1")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--configs", nargs="+", default=["d3", "d6", "t3", "t6"])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--w", nargs="+", type=float, default=[0.15, 0.3])
    ap.add_argument("--base", default="v33q20")
    a = ap.parse_args()
    rel, gd, lec, out = Path(a.release_root), Path(a.grid_dir), Path(a.le_cache), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in SCORE_PANELS:
        s = time.time()
        g = Grid(gd / f"{p}_f{a.fold}.npz")
        P = tr.Panel(rel, p)
        b = np.load(lec / f"{p}_f{a.fold}_UW_pred.npz")
        R = b["R"]
        o = {"R": R, "B_v": b[f"{a.base}_v"], "B_q": b[f"{a.base}_q"]}
        for cfg in a.configs:
            V, Q = grid_pred(g, P.month, tr.FOLDS[a.fold], "d" if cfg[0] == "d" else "t", int(cfg[1:]), a.iters)
            v, q = tr.finalize(g.vclip[P.tl[R]], V[P.td[R], P.tt[R], P.tl[R]], Q[P.td[R], P.tt[R], P.tl[R]])
            o[f"LR{cfg}_v"], o[f"LR{cfg}_q"] = v.astype(np.float32), q.astype(np.float32)
            for w in a.w:
                n = f"B_LR{cfg}w{int(round(w*100))}"
                o[f"{n}_v"] = ((1 - w) * o["B_v"] + w * v).astype(np.float32)
                o[f"{n}_q"] = ((1 - w) * o["B_q"] + w * q).astype(np.float32)
            print(p, cfg, f"{time.time()-s:.0f}s", flush=True)
        np.savez(lec / f"{p}_f{a.fold}_LR_pred.npz", **o)
        rows += tr.score_panel(rel, P, [a.fold], ["LR"], lec)
        del P, g
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out / "lowrank_cv.csv", index=False)
    gm = df.groupby("variant")[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
    gm["J"] = 0.35 * gm.S_state + 0.15 * gm.S_physics
    gm["dJ"] = gm.J - gm.loc["B", "J"]
    u = df.groupby(["variant", "panel"])[["S_state", "S_physics"]].mean()
    u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
    print(gm.sort_values("J", ascending=False).round(5).to_string(), flush=True)
    print((u.J.unstack(0).sub(u.J.unstack(0)["B"], axis=0)).round(5).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
