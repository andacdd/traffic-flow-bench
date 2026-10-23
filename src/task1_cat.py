import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
import task1_resid2 as r2

BLENDS = [0.3, 0.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--a-cache-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--score-panels", nargs="+", default=["D12_I5_N", "D7_I405_N", "D7_I210_W"])
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rel, acache, cache, out = Path(a.release_root), Path(a.a_cache_dir), Path(a.cache_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    parts = [r2.load_x(acache, cache, p, a.fold, "tr", True) for p in panels]
    X = np.concatenate([x for x, _ in parts])
    Bt = np.concatenate([d["Btr"] for _, d in parts]).astype(float)
    yv = np.concatenate([d["yv"] for _, d in parts]).astype(float)
    yq = np.concatenate([d["yq"] for _, d in parts]).astype(float)
    del parts
    gc.collect()
    print(f"train rows {len(X):,} feats {X.shape[1]}", flush=True)
    models = {}
    for tgt, y in [("dv", yv - Bt[:, 3]), ("dq", yq - Bt[:, 1])]:
        s = time.time()
        m = CatBoostRegressor(iterations=a.iters, depth=a.depth, learning_rate=a.lr, loss_function="RMSE", l2_leaf_reg=5,
                              thread_count=a.threads, random_seed=a.seed, verbose=250, allow_writing_files=False)
        m.fit(X, y)
        models[tgt] = m
        m.save_model(str(out / f"cat_f{a.fold}_{tgt}.cbm"))
        print(f"{tgt} trained {time.time()-s:.0f}s", flush=True)
    del X
    gc.collect()
    nA = len(tr.FEATS)
    for p in a.score_panels:
        s = time.time()
        X, d = r2.load_x(acache, cache, p, a.fold, "te", True)
        B, vc = d["Bte"].astype(float), d["vc_te"]
        v, q = tr.finalize(vc, B[:, 3] + models["dv"].predict(X, thread_count=a.threads), B[:, 1] + models["dq"].predict(X, thread_count=a.threads))
        le = np.load(cache / f"{p}_f{a.fold}_R2_pred.npz")
        o = {"R": d["Rte"], "LE_v": le["LE_v"], "LE_q": le["LE_q"], "C_v": v.astype(np.float32), "C_q": q.astype(np.float32)}
        for w in BLENDS:
            o[f"LEC{int(w*100)}_v"] = ((1 - w) * le["LE_v"] + w * v).astype(np.float32)
            o[f"LEC{int(w*100)}_q"] = ((1 - w) * le["LE_q"] + w * q).astype(np.float32)
        np.savez(cache / f"{p}_f{a.fold}_R2C_pred.npz", **o)
        print(p, f"predicted {time.time()-s:.0f}s", flush=True)
        del X, d
        gc.collect()
    rows = []
    for p in a.score_panels:
        P = tr.Panel(rel, p)
        rows += tr.score_panel(rel, P, [a.fold], ["R2C"], cache)
        del P
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out / "cat_cv.csv", index=False)
    g = df.groupby("variant")[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
    g["J"] = 0.35 * g.S_state + 0.15 * g.S_physics
    g["dJ"] = g.J - g.loc["LE", "J"]
    u = df.groupby(["variant", "panel"])[["S_state", "S_physics"]].mean()
    u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
    print(g.round(5).to_string(), flush=True)
    print(u.J.unstack(0).round(5).to_string(), flush=True)


if __name__ == "__main__":
    main()
