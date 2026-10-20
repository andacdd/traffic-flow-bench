import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
import task1_resid2 as r2
from task1_interp import T
from task1.baseline_task1_historical_mean import REGIMES

VARIANTS = ["B1", "B2", "B3"]
MEMBERS = ["L", "M2"]
NB = [-2, -1, 1, 2]
WIN = 12
QFEATS = ["q_pct0", "q_imp0"] + [f"q{kind}{o:+d}_{c}" for kind in ["nb", "lag"] for c in ["pct", "el"] for o in NB] + ["q_win_el", "q_win_obs", "q_win_hid"]
REG_RATE = {tr.REG[k]: v for k, v in REGIMES.items()}


def fold_rows(P, fold, n_sample, seed):
    hold = np.isin(P.month, tr.FOLDS[fold])
    rng = np.random.default_rng(seed + fold)
    cand = np.flatnonzero(~hold[P.td])
    pick = np.zeros(len(P.td), dtype=bool)
    pick[rng.choice(cand, size=min(n_sample, len(cand)), replace=False)] = True
    pick |= hold[P.td]
    R = np.concatenate([r[pick[r]] for di in range(P.D) if (r := P.day_rows.get(di)) is not None and pick[r].any()])
    trn = ~hold[P.td[R]]
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = trn & np.isfinite(yv) & np.isfinite(yq)
    return R[good], R[~trn], hold


def load_quality(rel, P):
    s = time.time()
    lix = pd.Index(P.order)
    shape = (P.D, T, P.L)
    Q = {c: np.full(shape, np.nan, dtype=np.float32) for c in ["pct", "el", "imp"]}
    for f in sorted((rel / "corridors" / P.name / "train" / "mainline_states_masked").glob("**/*.parquet")):
        t = pq.ParquetFile(f).read(columns=["date", "timestamp", "link_id", "pct_observed", "is_score_eligible", "is_imputed"])
        di, tod = tr.grid_index(t, P.days)
        li = lix.get_indexer(t["link_id"].to_numpy(zero_copy_only=False))
        ok = (li >= 0) & (di >= 0)
        for c, col in [("pct", "pct_observed"), ("el", "is_score_eligible"), ("imp", "is_imputed")]:
            Q[c][di[ok], tod[ok], li[ok]] = t[col].to_numpy(zero_copy_only=False)[ok]
    tr.tick("quality_load", s)
    return Q


def window_counts(P, Q):
    el = np.nan_to_num(Q["el"]) > 0
    obs = ~np.isnan(P.obs["v"])
    out = {}
    for name, a in [("el", el), ("obs", obs), ("hid", el & ~obs)]:
        pad = np.pad(a.astype(np.int16), ((0, 0), (WIN + 1, WIN), (0, 0)))
        out[name] = (np.cumsum(pad, axis=1, dtype=np.int32), a)
    return out


def q_features(P, Q, W, rows):
    d, t, l = P.td[rows], P.tt[rows], P.tl[rows]
    cols = [Q["pct"][d, t, l], Q["imp"][d, t, l]]
    for c in ["pct", "el"]:
        pad = np.pad(Q[c], ((0, 0), (0, 0), (2, 2)), constant_values=np.nan)
        cols += [pad[d, t, l + 2 + o] for o in NB]
    for c in ["pct", "el"]:
        pad = np.pad(Q[c], ((0, 0), (2, 2), (0, 0)), constant_values=np.nan)
        cols += [pad[d, t + 2 + o, l] for o in NB]
    for name in ["el", "obs", "hid"]:
        cs, a = W[name]
        cols.append((cs[d, t + 2 * WIN + 1, l] - cs[d, t, l] - a[d, t, l]).astype(np.float32))
    return np.column_stack(cols).astype(np.float32)


def profile_sums(P, keep):
    dow = P.dow[keep]
    out = {}
    for c in tr.VARS:
        x = P.obs[c][keep]
        S = np.zeros((7, T, P.L))
        N = np.zeros((7, T, P.L))
        for d in range(7):
            s = x[dow == d]
            S[d] = np.nansum(s, axis=0)
            N[d] = (~np.isnan(s)).sum(axis=0)
        out[c] = (S, N)
    return out


def aug_rows(P, fold, hold, prof, n_aug, seed):
    s = time.time()
    rng = np.random.default_rng(seed + 100 + fold)
    sums = profile_sums(P, ~hold)
    days, blanks, cands = [], [], []
    for di in range(P.D):
        rows = P.day_rows.get(di)
        if hold[di] or rows is None:
            continue
        rate = REG_RATE[int(P.treg[rows[0]])]
        ok = ~np.isnan(P.obs["v"][di]) & np.isfinite(P.tv[di]) & np.isfinite(P.tq[di])
        b = (rng.random((T, P.L)) < rate) & ok
        days.append(di)
        blanks.append(b)
        cands.append(np.flatnonzero(b))
    sizes = np.array([len(c) for c in cands])
    n = min(n_aug, int(sizes.sum()))
    chosen = np.sort(rng.choice(int(sizes.sum()), size=n, replace=False))
    owner = np.searchsorted(np.cumsum(sizes), chosen, side="right")
    off = np.r_[0, np.cumsum(sizes)[:-1]]
    XA, XL, Bs, ys = [], [], [], []
    for k in np.unique(owner):
        di, b = days[k], blanks[k]
        flat = cands[k][chosen[owner == k] - off[k]]
        t, l = np.unravel_index(flat, (T, P.L))
        dw = P.dow[di]
        aug, pa = {}, {}
        for c in tr.VARS:
            x = P.obs[c][di]
            aug[c] = np.where(b, np.nan, x)[None]
            S, N = sums[c]
            loo = np.divide(S[dw] - np.nan_to_num(x), N[dw] - 1, out=np.full((T, P.L), np.nan), where=N[dw] > 1)
            p = prof[c].copy()
            p[dw] = np.where(b & np.isfinite(loo), loo, prof[c][dw])
            pa[c] = p
        reg = int(P.treg[P.day_rows[di][0]])
        S1 = SimpleNamespace(tt=t, tl=l, dow=np.array([dw]), obs=aug, pos=P.pos, vf=P.vf, treg=np.full(len(t), reg), ramp_on=P.ramp_on[di][None],
                             ramp_off=P.ramp_off[di][None], geo=P.geo, L=P.L)
        r = np.arange(len(t))
        X, B = tr.day_features(S1, 0, r, pa)
        XA.append(X)
        XL.append(r2.ll_features(S1, 0, r, pa))
        Bs.append(B)
        ys.append(np.column_stack([P.tv[di][t, l], P.tq[di][t, l]]))
    tr.tick("aug_features", s)
    y = np.concatenate(ys)
    return {"XA": np.concatenate(XA), "XL": np.concatenate(XL), "B": np.concatenate(Bs), "yv": y[:, 0], "yq": y[:, 1], "n_cand": int(sizes.sum())}


def prep_panel(rel, p, folds, variants, args, acache, cache):
    P = tr.Panel(rel, p)
    need_q = "B1" in variants
    Q = load_quality(rel, P) if need_q else None
    W = window_counts(P, Q) if need_q else None
    for f in folds:
        s = time.time()
        Rtr, Rte, hold = fold_rows(P, f, args.sample_per_panel, args.seed)
        a = np.load(acache / f"{p}_f{f}.npz")
        assert np.array_equal(a["Rte"], Rte) and np.array_equal(a["yv"], P.tv[P.td[Rtr], P.tt[Rtr], P.tl[Rtr]])
        months = sorted(set(P.month[~hold]))
        out = {"Rtr": Rtr, "es": P.month[P.td[Rtr]] == months[-1]}
        if need_q:
            out["Qtr"], out["Qte"] = q_features(P, Q, W, Rtr), q_features(P, Q, W, Rte)
        if args.check_rows:
            di = int(P.td[Rtr[0]])
            sel = np.flatnonzero(P.td[Rtr] == di)
            X, _ = tr.day_features(P, di, Rtr[sel], P.profiles(~hold))
            assert np.array_equal(X, a["Xtr"][sel], equal_nan=True)
        if "B3" in variants:
            ag = aug_rows(P, f, hold, P.profiles(~hold), int(args.aug_frac * len(Rtr)), args.seed)
            out.update({f"aug_{k}": v for k, v in ag.items()})
            print(f"  {p} fold {f}: aug {len(ag['yv']):,} of {ag['n_cand']:,} candidates", flush=True)
        np.savez(cache / f"{p}_f{f}_R3{variants[0]}.npz", **out)
        tr.tick("prep", s)
        print(f"{p} fold {f}: train {len(Rtr):,} es {int(out['es'].sum()):,} held-out {len(Rte):,} | peak {tr.peak_gb():.2f} GB", flush=True)
    del P, Q, W
    gc.collect()


def load_part(acache, lcache, cache, p, fold, part, variant):
    a = np.load(acache / f"{p}_f{fold}.npz")
    blocks = [a[f"X{part}"], np.load(lcache / f"{p}_f{fold}_L.npz")[f"X{part}"]]
    m = np.load(cache / f"{p}_f{fold}_R3{variant}.npz")
    if variant == "B1":
        blocks.append(m[f"Q{part}"])
    return np.hstack(blocks), a, m


def train_matrix(panels, fold, variant, acache, lcache, cache):
    n, k = 0, None
    for p in panels:
        m = np.load(cache / f"{p}_f{fold}_R3{variant}.npz")
        n += len(m["Rtr"]) + (len(m["aug_yv"]) if variant == "B3" else 0)
    k = len(r2.ALLFEATS) + (len(QFEATS) if variant == "B1" else 0)
    X = np.empty((n, k), dtype=np.float32)
    Bt, yv, yq, es = np.empty((n, 4)), np.empty(n), np.empty(n), np.zeros(n, dtype=bool)
    i = 0
    for p in panels:
        x, a, m = load_part(acache, lcache, cache, p, fold, "tr", variant)
        j = i + len(x)
        X[i:j], Bt[i:j], yv[i:j], yq[i:j], es[i:j] = x, a["Btr"], a["yv"], a["yq"], m["es"]
        i = j
        if variant == "B3":
            j = i + len(m["aug_yv"])
            X[i:j] = np.hstack([m["aug_XA"], m["aug_XL"]])
            Bt[i:j], yv[i:j], yq[i:j] = m["aug_B"], m["aug_yv"], m["aug_yq"]
            i = j
        del x, a, m
    assert i == n
    return X, Bt, yv, yq, es


def fit_fold(panels, fold, variant, args, acache, lcache, cache, imp, iters):
    s = time.time()
    X, Bt, yv, yq, es = train_matrix(panels, fold, variant, acache, lcache, cache)
    names = r2.ALLFEATS + (QFEATS if variant == "B1" else [])
    print(f"  fold {fold} {variant}: train matrix {X.shape} es rows {int(es.sum()):,} peak {tr.peak_gb():.2f} GB", flush=True)
    models = {}
    for m in MEMBERS:
        params, _, ds = r2.MODELS[m]
        sd = args.seed + ds
        for tgt, y in [("dv", yv - Bt[:, 3]), ("dq", yq - Bt[:, 1])]:
            p = dict(params, num_threads=args.threads, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
            rounds = args.rounds
            if variant == "B2":
                p = dict(p, learning_rate=args.es_lr)
                dtr = lgb.Dataset(X[~es], y[~es], feature_name=names, free_raw_data=True)
                dva = lgb.Dataset(X[es], y[es], reference=dtr, free_raw_data=True)
                e = lgb.train(p, dtr, args.es_max_rounds, valid_sets=[dva], valid_names=["es"], callbacks=[lgb.early_stopping(args.es_patience, verbose=False)])
                rounds = int(e.best_iteration)
                iters.append({"fold": fold, "model": m, "target": tgt, "best_iteration": rounds, "best_l2": float(e.best_score["es"]["l2"])})
                print(f"  fold {fold} {m} {tgt} early stop best_iteration {rounds} l2 {e.best_score['es']['l2']:.4f} {time.time()-s:.0f}s", flush=True)
                del e, dtr, dva
            b = lgb.train(p, lgb.Dataset(X, y, feature_name=names, free_raw_data=True), rounds)
            models.setdefault(m, {})[tgt] = b
            g = b.feature_importance("gain")
            imp.append(pd.DataFrame({"variant": variant, "model": m, "target": tgt, "fold": fold, "feature": names, "gain": g, "gain_frac": g / max(g.sum(), 1e-12)}))
            print(f"  fold {fold} {variant} {m} {tgt} trained {rounds} rounds {time.time()-s:.0f}s peak {tr.peak_gb():.2f} GB", flush=True)
    del X
    gc.collect()
    tr.tick("train", s)
    return models


def predict_panel(p, fold, variant, models, acache, lcache, cache):
    s = time.time()
    X, a, _ = load_part(acache, lcache, cache, p, fold, "te", variant)
    B, vc = a["Bte"].astype(float), a["vc_te"]
    raw = {m: (models[m]["dv"].predict(X, num_threads=4), models[m]["dq"].predict(X, num_threads=4)) for m in MEMBERS}
    raw[variant] = tuple((raw["L"][i] + raw["M2"][i]) / 2 for i in range(2))
    out = {"R": a["Rte"]}
    for name, (dv, dq) in raw.items():
        v, q = tr.finalize(vc, B[:, 3] + dv, B[:, 1] + dq)
        key = name if name == variant else f"{variant}_{name}"
        out[f"{key}_v"], out[f"{key}_q"] = v.astype(np.float32), q.astype(np.float32)
    np.savez(cache / f"{p}_f{fold}_{variant}_pred.npz", **out)
    tr.tick("predict", s)


def summarize(out_dir, ref):
    frames = [pd.read_csv(f) for f in sorted(out_dir.glob("resid3_cv_*.csv"))]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    full = pd.concat([ref.assign(variant="LE_ref"), df], ignore_index=True)
    full = full[full.set_index(["fold", "panel"]).index.isin(df.set_index(["fold", "panel"]).index)]
    met = ["S_state", "S_FD", "S_LWR", "S_physics", "rmse_v", "rmse_q_lane", "q_rmse_v", "q_rmse_q_lane", "q_mae_k", "nq_rmse_v", "nq_rmse_q_lane", "nq_mae_k"]
    g = full.groupby("variant")
    s = g[met].mean()
    s["n_cells"] = g.n.sum()
    s["q_bias_rel"] = (g.q_sum_dk.sum() / g.q_n.sum()) / (g.q_sum_kt.sum() / g.q_n.sum())
    s["nq_bias_k"] = g.nq_sum_dk.sum() / g.nq_n.sum()
    s["J"] = 0.35 * s.S_state + 0.15 * s.S_physics
    worse = (s.q_bias_rel.abs() - abs(s.loc["LE_ref", "q_bias_rel"])) > 0.02
    s["guard"] = np.where((s.q_bias_rel < -0.05) | worse, "REJECTED", "ok")
    ua = r2.unit_table(full)
    uref = ua.loc["LE_ref"]
    for c in ["S_state", "S_physics", "J"]:
        d = ua[c] - uref[c].reindex(ua.index.droplevel(0)).to_numpy()
        s[f"d_{c}_mean"] = d.groupby(level=0).mean()
        s[f"d_{c}_std"] = d.groupby(level=0).std()
        s[f"d_{c}_nwin"] = (d > 0).groupby(level=0).sum()
        s[f"d_{c}_n"] = d.groupby(level=0).size()
    s["J_gain_over_2std"] = s.d_J_mean > 2 * s.d_J_std
    s = s.reset_index()
    s.to_csv(out_dir / "resid3_summary.csv", index=False)
    reg = full.groupby(["variant", "regime"])[["S_state", "S_physics", "S_LWR", "q_rmse_v", "q_mae_k", "nq_rmse_v"]].mean()
    reg.to_csv(out_dir / "resid3_regime.csv")
    pan = ua.groupby(level=[0, 2])[["S_state", "S_physics", "J"]].mean()
    pan.to_csv(out_dir / "resid3_panel.csv")
    return s, reg, pan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--a-cache-dir", required=True)
    ap.add_argument("--l-cache-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--le-ref", required=True)
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--panels", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--sample-per-panel", type=int, default=300000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--es-max-rounds", type=int, default=2000)
    ap.add_argument("--es-lr", type=float, default=0.05)
    ap.add_argument("--es-patience", type=int, default=100)
    ap.add_argument("--aug-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--check-rows", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out_dir, cache, acache, lcache = Path(args.out_dir), Path(args.cache_dir), Path(args.a_cache_dir), Path(args.l_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (out_dir / "versions.json").write_text(json.dumps(r2.versions(), indent=2))
    panels = args.panels or sorted(p.name for p in (rel / "corridors").iterdir())
    ref = pd.read_csv(args.le_ref)
    ref = ref[ref.panel.isin(panels) & ref.fold.isin(args.folds)]
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    t_all = time.time()
    v = args.variant
    if not args.summary_only:
        for p in panels:
            if args.reuse_cache and all((cache / f"{p}_f{f}_R3{v}.npz").exists() for f in args.folds):
                continue
            prep_panel(rel, p, args.folds, [v], args, acache, cache)
        imp, iters = [], []
        for f in args.folds:
            models = fit_fold(panels, f, v, args, acache, lcache, cache, imp, iters)
            for p in panels:
                predict_panel(p, f, v, models, acache, lcache, cache)
            del models
            gc.collect()
            print(f"fold {f} trained+predicted | {({k: round(x) for k, x in tr.STAGES.items()})} | peak {tr.peak_gb():.2f} GB", flush=True)
        rows = []
        for p in panels:
            P = tr.Panel(rel, p)
            rows += tr.score_panel(rel, P, args.folds, [v], cache)
            del P
            gc.collect()
        pd.DataFrame(rows).to_csv(out_dir / f"resid3_cv_{v}.csv", index=False)
        pd.concat(imp).to_csv(out_dir / f"resid3_importance_{v}.csv", index=False)
        if iters:
            pd.DataFrame(iters).to_csv(out_dir / f"resid3_es_iters_{v}.csv", index=False)
        run = {"variant": v, "panels": panels, "folds": args.folds, "args": vars(args), "stages": {k: round(x, 1) for k, x in tr.STAGES.items()},
               "runtime_s": round(time.time() - t_all, 1), "peak_gb": round(tr.peak_gb(), 2)}
        (out_dir / f"resid3_run_{v}.json").write_text(json.dumps(run, indent=2))
    res = summarize(out_dir, ref)
    if res is not None:
        s, reg, pan = res
        print(s[["variant", "S_state", "S_FD", "S_LWR", "S_physics", "J", "q_rmse_v", "q_rmse_q_lane", "q_mae_k", "q_bias_rel", "nq_rmse_v", "nq_rmse_q_lane", "nq_mae_k", "guard"]].round(4).to_string(index=False))
        print(s[["variant"] + [c for c in s.columns if c.startswith("d_")] + ["J_gain_over_2std"]].round(5).to_string(index=False))
        print(reg.round(4).unstack("regime").to_string())
        print(pan.round(4).unstack(0)["J"].to_string())
    print(f"stages {({k: round(x) for k, x in tr.STAGES.items()})} total {time.time()-t_all:.0f}s peak {tr.peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
