import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_interp import T

VARS = tr.VARS
OFFS = [-12, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 12]
RAW_OFFS = [o for o in OFFS if abs(o) > 2]
NB_LINKS = [-2, -1, 1, 2]
NB_OFFS = [-3, -1, 1, 3]
M = max(OFFS)


def l_feature_names():
    f = [f"ll{o:+d}_{c}" for c in VARS for o in RAW_OFFS]
    f += [f"llr{o:+d}_{c}" for c in VARS for o in OFFS]
    f += [f"{side}{r}_{c}" for r in ["", "r"] for c in VARS for side in ["prev", "next"]]
    f += ["prev_dist", "next_dist"]
    f += [f"nbt{lo:+d}t{o:+d}_{c}" for c in VARS for lo in NB_LINKS for o in NB_OFFS]
    return f


LFEATS = l_feature_names()
ALLFEATS = tr.FEATS + LFEATS
M2 = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}
MODELS = {"A": (tr.PARAMS, False, 0), "L": (tr.PARAMS, True, 0), "M2": (M2, True, 1000)}
ENS = {"E": ("A", "M2"), "LE": ("L", "M2")}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "lightgbm": lgb.__version__, "scipy": scipy.__version__}


def ll_features(P, di, rows, prof):
    t, l = P.tt[rows], P.tl[rows]
    dw = P.dow[di]
    obs = {c: P.obs[c][di] for c in VARS}
    rat = {c: obs[c] / np.where(prof[c][dw] > 0, prof[c][dw], np.nan) for c in VARS}
    cols = []
    for c in VARS:
        pad = np.pad(obs[c], ((M, M), (0, 0)), constant_values=np.nan)
        cols += [pad[t + M + o, l] for o in RAW_OFFS]
    for c in VARS:
        pad = np.pad(rat[c], ((M, M), (0, 0)), constant_values=np.nan)
        cols += [pad[t + M + o, l] for o in OFFS]
    miss = np.isnan(obs["v"])
    idx = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(~miss, idx, -1), axis=0)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(~miss, idx, T), 0), axis=0), 0)
    prev = np.vstack([np.full((1, P.L), -1), last[:-1]])[t, l]
    after = np.vstack([nxt[1:], np.full((1, P.L), T)])[t, l]
    hp, ha = prev >= 0, after < T
    pi, ai = np.clip(prev, 0, T - 1), np.clip(after, 0, T - 1)
    for src in [obs, rat]:
        for c in VARS:
            cols += [np.where(hp, src[c][pi, l], np.nan), np.where(ha, src[c][ai, l], np.nan)]
    cols += [np.where(hp, t - prev, np.nan).astype(float), np.where(ha, after - t, np.nan).astype(float)]
    for c in VARS:
        pad = np.pad(obs[c], ((3, 3), (2, 2)), constant_values=np.nan)
        for lo in NB_LINKS:
            cols += [pad[t + 3 + o, l + 2 + lo] for o in NB_OFFS]
    return np.column_stack(cols).astype(np.float32)


def build_fold_l(P, fold, n_sample, seed, acache, cache):
    s = time.time()
    hold = np.isin(P.month, tr.FOLDS[fold])
    prof = P.profiles(~hold)
    rng = np.random.default_rng(seed + fold)
    cand = np.flatnonzero(~hold[P.td])
    pick = np.zeros(len(P.td), dtype=bool)
    pick[rng.choice(cand, size=min(n_sample, len(cand)), replace=False)] = True
    pick |= hold[P.td]
    Xs, Rs = [], []
    for di in range(P.D):
        rows = P.day_rows.get(di)
        if rows is None:
            continue
        rows = rows[pick[rows]]
        if len(rows) == 0:
            continue
        Xs.append(ll_features(P, di, rows, prof))
        Rs.append(rows)
    X, R = np.concatenate(Xs), np.concatenate(Rs)
    trn = ~hold[P.td[R]]
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = trn & np.isfinite(yv) & np.isfinite(yq)
    a = np.load(acache / f"{P.name}_f{fold}.npz")
    assert np.array_equal(a["Rte"], R[~trn]) and len(a["Xtr"]) == int(good.sum()) and np.allclose(a["yv"], yv[good])
    np.savez(cache / f"{P.name}_f{fold}_L.npz", Xtr=X[good], Xte=X[~trn])
    tr.tick("features_L", s)
    return int(good.sum()), int((~trn).sum())


def load_x(acache, cache, p, fold, part, with_l):
    a = np.load(acache / f"{p}_f{fold}.npz")
    X = a[f"X{part}"]
    if with_l:
        X = np.hstack([X, np.load(cache / f"{p}_f{fold}_L.npz")[f"X{part}"]])
    return X, a


def fit_fold(panels, fold, args, acache, cache, imp):
    s = time.time()
    models = {}
    for with_l in [False, True]:
        parts = [load_x(acache, cache, p, fold, "tr", with_l) for p in panels]
        X = np.concatenate([x for x, _ in parts])
        Bt = np.concatenate([a["Btr"] for _, a in parts]).astype(float)
        yv = np.concatenate([a["yv"] for _, a in parts]).astype(float)
        yq = np.concatenate([a["yq"] for _, a in parts]).astype(float)
        del parts
        names = ALLFEATS if with_l else tr.FEATS
        for m, (params, ml, ds) in MODELS.items():
            if ml != with_l:
                continue
            sd = args.seed + ds
            for tgt, y in [("dv", yv - Bt[:, 3]), ("dq", yq - Bt[:, 1])]:
                p = dict(params, num_threads=args.threads, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
                b = lgb.train(p, lgb.Dataset(X, y, feature_name=names, free_raw_data=True), args.rounds)
                models.setdefault(m, {})[tgt] = b
                g = b.feature_importance("gain")
                imp.append(pd.DataFrame({"model": m, "target": tgt, "fold": fold, "feature": names, "gain": g, "gain_frac": g / max(g.sum(), 1e-12)}))
                print(f"  fold {fold} {m} {tgt} trained {time.time()-s:.0f}s peak {tr.peak_gb():.2f} GB", flush=True)
        del X
        gc.collect()
    tr.tick("train", s)
    return models


def predict_panel(p, fold, models, acache, cache):
    s = time.time()
    X, a = load_x(acache, cache, p, fold, "te", True)
    B, vc = a["Bte"].astype(float), a["vc_te"]
    nA = len(tr.FEATS)
    raw = {}
    for m, (_, ml, _) in MODELS.items():
        Xm = X if ml else X[:, :nA]
        raw[m] = (models[m]["dv"].predict(Xm, num_threads=4), models[m]["dq"].predict(Xm, num_threads=4))
    for e, (m1, m2) in ENS.items():
        raw[e] = ((raw[m1][0] + raw[m2][0]) / 2, (raw[m1][1] + raw[m2][1]) / 2)
    out = {"R": a["Rte"]}
    for name, (dv, dq) in raw.items():
        v, q = tr.finalize(vc, B[:, 3] + dv, B[:, 1] + dq)
        out[f"{name}_v"], out[f"{name}_q"] = v.astype(np.float32), q.astype(np.float32)
    np.savez(cache / f"{p}_f{fold}_R2_pred.npz", **out)
    tr.tick("predict", s)


def unit_table(df):
    u = df.groupby(["variant", "fold", "panel"])[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
    u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
    return u


def summarize(out_dir, df, base):
    met = ["S_state", "S_FD", "S_LWR", "S_physics", "rmse_v", "rmse_q_lane", "q_rmse_v", "q_rmse_q_lane", "q_mae_k", "nq_rmse_v", "nq_rmse_q_lane", "nq_mae_k"]
    full = pd.concat([base.assign(variant="A_ref"), df], ignore_index=True)
    g = full.groupby("variant")
    s = g[met].mean()
    s["n_cells"] = g.n.sum()
    s["q_bias_k"] = g.q_sum_dk.sum() / g.q_n.sum()
    s["q_mean_kt"] = g.q_sum_kt.sum() / g.q_n.sum()
    s["nq_bias_k"] = g.nq_sum_dk.sum() / g.nq_n.sum()
    s["q_bias_rel"] = s.q_bias_k / s.q_mean_kt
    s["J"] = 0.35 * s.S_state + 0.15 * s.S_physics
    ref = s.loc["A_ref"]
    worse = (s.q_bias_rel.abs() - abs(ref.q_bias_rel)) > 0.02
    s["guard"] = np.where((s.q_bias_rel < -0.05) | worse, "REJECTED", "ok")
    ua = unit_table(full)
    uref = ua.loc["A_ref"]
    for c in ["S_state", "S_physics", "J"]:
        d = ua[c] - uref[c].reindex(ua.index.droplevel(0)).to_numpy()
        s[f"d_{c}_mean"] = d.groupby(level=0).mean()
        s[f"d_{c}_std"] = d.groupby(level=0).std()
        s[f"d_{c}_nwin"] = (d > 0).groupby(level=0).sum()
    s = s.reset_index()
    s.to_csv(out_dir / "resid2_summary.csv", index=False)
    reg = full.groupby(["variant", "regime"])[["S_state", "S_physics", "q_rmse_v", "q_mae_k"]].mean()
    pan = ua.groupby(level=[0, 2])[["S_state", "S_physics", "J"]].mean()
    return s, reg, pan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a-cache-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--panels", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--sample-per-panel", type=int, default=300000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--a-ref", required=True)
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out_dir, cache, acache = Path(args.out_dir), Path(args.cache_dir), Path(args.a_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    panels = args.panels or sorted(p.name for p in (rel / "corridors").iterdir())
    (out_dir / "versions.json").write_text(json.dumps(versions(), indent=2))
    t_all = time.time()
    for p in panels:
        need = [f for f in args.folds if not (args.reuse_cache and (cache / f"{p}_f{f}_L.npz").exists())]
        if not need:
            continue
        P = tr.Panel(rel, p)
        for f in need:
            ntr, nte = build_fold_l(P, f, args.sample_per_panel, args.seed, acache, cache)
            print(f"{p} fold {f}: train {ntr:,} held-out {nte:,} | {len(LFEATS)} L feats | peak {tr.peak_gb():.2f} GB", flush=True)
        del P
        gc.collect()
    imp = []
    for f in args.folds:
        models = fit_fold(panels, f, args, acache, cache, imp)
        for p in panels:
            predict_panel(p, f, models, acache, cache)
        del models
        gc.collect()
        print(f"fold {f} trained+predicted | {({k: round(v) for k, v in tr.STAGES.items()})} | peak {tr.peak_gb():.2f} GB", flush=True)
    rows = []
    for p in panels:
        P = tr.Panel(rel, p)
        rows += tr.score_panel(rel, P, args.folds, ["R2"], cache)
        del P
        gc.collect()
    df = pd.DataFrame(rows)
    for v in df.variant.unique():
        df[df.variant == v].assign(exp=v).to_csv(out_dir / f"resid2_cv_{v}.csv", index=False)
    pd.concat(imp).to_csv(out_dir / "resid2_importance.csv", index=False)
    base = pd.read_csv(args.a_ref)
    base = base[base.panel.isin(panels) & base.fold.isin(args.folds)]
    chk = df[df.variant == "A"].merge(base, on=["fold", "panel", "regime"], suffixes=("", "_ref"))
    print("A reproduction max |dS_state|", float((chk.S_state - chk.S_state_ref).abs().max()), "max |dS_physics|", float((chk.S_physics - chk.S_physics_ref).abs().max()), flush=True)
    s, reg, pan = summarize(out_dir, df, base)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    print(s[["variant", "S_state", "S_FD", "S_LWR", "S_physics", "J", "q_rmse_v", "q_rmse_q_lane", "q_mae_k", "q_bias_rel", "nq_rmse_v", "nq_rmse_q_lane", "nq_mae_k", "nq_bias_k", "guard"]].round(4).to_string(index=False))
    print(s[["variant"] + [c for c in s.columns if c.startswith("d_")]].round(5).to_string(index=False))
    print(reg.round(4).unstack("regime").to_string())
    print(pan.round(4).unstack(0)["J"].to_string())
    print(f"stages {({k: round(v) for k, v in tr.STAGES.items()})} total {time.time()-t_all:.0f}s peak {tr.peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
