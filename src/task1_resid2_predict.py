import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
import task1_resid2 as r2
import task1_resid_predict as rp

KEY = rp.KEY
SPLITS = rp.SPLITS
ENS = "LE"
MEMBERS = r2.ENS[ENS]
TARGETS = ["dv", "dq"]


def pick_rows(P, hold, n_sample, seed):
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(~hold[P.td])
    pick = np.zeros(len(P.td), dtype=bool)
    pick[rng.choice(cand, size=min(n_sample, len(cand)), replace=False)] = True
    pick |= hold[P.td]
    return pick


def all_rows(P, prof, rows_of=None):
    Xs, Bs, Rs = [], [], []
    for di in range(P.D):
        rows = P.day_rows.get(di)
        if rows is None:
            continue
        if rows_of is not None:
            rows = rows[rows_of[rows]]
        if len(rows) == 0:
            continue
        X, B = tr.day_features(P, di, rows, prof)
        Xs.append(np.hstack([X, r2.ll_features(P, di, rows, prof)]))
        Bs.append(B)
        Rs.append(rows)
    return np.concatenate(Xs), np.concatenate(Bs), np.concatenate(Rs)


def build_train(rel, p, n_sample, seed, cache, aug_frac=0.0):
    s = time.time()
    P = tr.Panel(rel, p)
    prof = P.profiles(np.ones(P.D, dtype=bool))
    pick = pick_rows(P, np.zeros(P.D, dtype=bool), n_sample, seed)
    X, B, R = all_rows(P, prof, pick)
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = np.isfinite(yv) & np.isfinite(yq)
    X, B, yv, yq = X[good], B[good], yv[good], yq[good]
    if aug_frac > 0:
        import task1_resid3 as r3
        ag = r3.aug_rows(P, 0, np.zeros(P.D, dtype=bool), prof, int(aug_frac * len(yv)), seed)
        X = np.vstack([X, np.hstack([ag["XA"], ag["XL"]]).astype(X.dtype)])
        B = np.vstack([B, ag["B"].astype(B.dtype)])
        yv, yq = np.r_[yv, ag["yv"]], np.r_[yq, ag["yq"]]
        print(f"  {p}: aug {len(ag['yv']):,} of {ag['n_cand']:,} candidates", flush=True)
    good = np.ones(len(yv), dtype=bool)
    np.savez(cache / f"{p}_train.npz", X=X, B=B, yv=yv, yq=yq)
    np.savez(cache / f"{p}_prof.npz", **prof)
    tr.tick("train_features", s)
    n = int(good.sum())
    del P
    gc.collect()
    return prof, n


def build_split(rel, p, split, prof, cache=None):
    s = time.time()
    SP = rp.SplitPanel(rel, p, split)
    assert SP.n_tpl_drop == 0, (p, split, SP.n_tpl_drop)
    X, B, R = all_rows(SP, prof)
    assert len(R) == len(SP.tpl) and np.array_equal(np.sort(R), np.arange(len(SP.tpl)))
    out = {"X": X, "B": B, "R": R, "vf": SP.vf[SP.tl[R]], "vc": SP.vclip[SP.tl[R]]}
    if cache is not None:
        np.savez(cache / f"{p}_{split}.npz", **out)
    tr.tick("split_features", s)
    return out, SP.tpl


def fit(X, y, m, threads, rounds, seed):
    params, _, ds = r2.MODELS[m]
    sd = seed + ds
    p = dict(params, num_threads=threads, seed=sd, bagging_seed=sd, feature_fraction_seed=sd, data_random_seed=sd)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=r2.ALLFEATS, free_raw_data=True), rounds)


def fit_members(X, B, yv, yq, args, cache=None):
    models = {}
    for m in MEMBERS:
        for tgt, y in [("dv", yv - B[:, 3]), ("dq", yq - B[:, 1])]:
            s = time.time()
            models.setdefault(m, {})[tgt] = fit(X, y, m, args.threads, args.rounds, args.seed)
            if cache is not None:
                models[m][tgt].save_model(str(cache / f"model_{m}_{tgt}.txt"))
            print(f"  {m} {tgt} trained {time.time()-s:.0f}s peak {tr.peak_gb():.2f} GB", flush=True)
    return models


def raw_members(models, X, threads):
    return {m: tuple(models[m][t].predict(X, num_threads=threads) for t in TARGETS) for m in MEMBERS}


def ensemble(raw):
    a, b = (raw[m] for m in MEMBERS)
    return (a[0] + b[0]) / 2, (a[1] + b[1]) / 2


def predict_frame(d, tpl, models, threads):
    B = d["B"].astype(float)
    dv, dq = ensemble(raw_members(models, d["X"], threads))
    v, q = tr.finalize(d["vc"], B[:, 3] + dv, B[:, 1] + dq)
    sp, fl = np.full(len(tpl), np.nan), np.full(len(tpl), np.nan)
    sp[d["R"]], fl[d["R"]] = v, q
    return tpl.assign(speed_kmh=sp, flow_vph=fl)


def train_models(panels, cache, args):
    s = time.time()
    parts = [np.load(cache / f"{p}_train.npz") for p in panels]
    X = np.concatenate([d["X"] for d in parts])
    B = np.concatenate([d["B"] for d in parts]).astype(float)
    yv = np.concatenate([d["yv"] for d in parts]).astype(float)
    yq = np.concatenate([d["yq"] for d in parts]).astype(float)
    del parts
    models = fit_members(X, B, yv, yq, args, cache)
    n = len(X)
    del X
    gc.collect()
    tr.tick("train", s)
    return models, n


def load_models(cache):
    return {m: {t: lgb.Booster(model_file=str(cache / f"model_{m}_{t}.txt")) for t in TARGETS} for m in MEMBERS}


def cv_check(rel, args):
    s = time.time()
    p, fold = "D12_I5_N", 1
    acv, lcv = Path(args.cv_a_cache), Path(args.cv_l_cache)
    panels = sorted(x.name for x in (rel / "corridors").iterdir())
    P = tr.Panel(rel, p)
    res = {"panel": p, "fold": fold}
    if args.prod_a_cache:
        prof = P.profiles(np.ones(P.D, dtype=bool))
        X, B, R = all_rows(P, prof, pick_rows(P, np.zeros(P.D, dtype=bool), args.sample_per_panel, args.seed))
        yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
        good = np.isfinite(yv) & np.isfinite(yq)
        ref = np.load(Path(args.prod_a_cache) / f"{p}_train.npz")
        nA = len(tr.FEATS)
        res["prod_train_A_part_identical"] = {"X": bool(np.array_equal(X[good][:, :nA], ref["X"], equal_nan=True)), "B": bool(np.array_equal(B[good], ref["B"])),
                                             "yv": bool(np.array_equal(yv[good], ref["yv"])), "yq": bool(np.array_equal(yq[good], ref["yq"]))}
        del X, B, R, prof
        gc.collect()
    hold = np.isin(P.month, tr.FOLDS[fold])
    prof = P.profiles(~hold)
    X, B, R = all_rows(P, prof, pick_rows(P, hold, args.sample_per_panel, args.seed + fold))
    trn = ~hold[P.td[R]]
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = trn & np.isfinite(yv) & np.isfinite(yq)
    a, l = np.load(acv / f"{p}_f{fold}.npz"), np.load(lcv / f"{p}_f{fold}_L.npz")
    nA = len(tr.FEATS)
    res["cv_features_identical"] = {"Xtr_A": bool(np.array_equal(X[good][:, :nA], a["Xtr"], equal_nan=True)), "Xtr_L": bool(np.array_equal(X[good][:, nA:], l["Xtr"], equal_nan=True)),
                                    "Xte_A": bool(np.array_equal(X[~trn][:, :nA], a["Xte"], equal_nan=True)), "Xte_L": bool(np.array_equal(X[~trn][:, nA:], l["Xte"], equal_nan=True)),
                                    "Btr": bool(np.array_equal(B[good], a["Btr"])), "Bte": bool(np.array_equal(B[~trn], a["Bte"])), "Rte": bool(np.array_equal(R[~trn], a["Rte"])),
                                    "yv": bool(np.array_equal(yv[good], a["yv"])), "yq": bool(np.array_equal(yq[good], a["yq"]))}
    print(json.dumps(res), flush=True)
    Xte, Bte, vc = X[~trn], B[~trn].astype(float), a["vc_te"]
    del P, X, B, R, a, l
    gc.collect()
    parts = []
    for q in panels:
        aq = np.load(acv / f"{q}_f{fold}.npz")
        parts.append((np.hstack([aq["Xtr"], np.load(lcv / f"{q}_f{fold}_L.npz")["Xtr"]]), aq["Btr"], aq["yv"], aq["yq"]))
    Xt = np.concatenate([x[0] for x in parts])
    Bt = np.concatenate([x[1] for x in parts]).astype(float)
    yv = np.concatenate([x[2] for x in parts]).astype(float)
    yq = np.concatenate([x[3] for x in parts]).astype(float)
    del parts
    res["cv_train_rows"] = len(Xt)
    models = fit_members(Xt, Bt, yv, yq, args)
    del Xt
    gc.collect()
    raw = raw_members(models, Xte, args.threads)
    raw[ENS] = ensemble(raw)
    ref = np.load(lcv / f"{p}_f{fold}_R2_pred.npz")
    for name, (dv, dq) in raw.items():
        v, q = tr.finalize(vc, Bte[:, 3] + dv, Bte[:, 1] + dq)
        res[f"{name}_max_abs_diff_v"] = float(np.max(np.abs(v.astype(np.float32) - ref[f"{name}_v"])))
        res[f"{name}_max_abs_diff_q"] = float(np.max(np.abs(q.astype(np.float32) - ref[f"{name}_q"])))
    res["runtime_s"] = round(time.time() - s, 1)
    res["peak_gb"] = round(tr.peak_gb(), 2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--panels", nargs="+")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--sample-per-panel", type=int, default=300000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug-frac", type=float, default=0.0)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--check-panel")
    ap.add_argument("--cv-check", action="store_true")
    ap.add_argument("--cv-a-cache")
    ap.add_argument("--cv-l-cache")
    ap.add_argument("--prod-a-cache")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out, cache = Path(args.out_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (out / "versions.json").write_text(json.dumps(rp.versions(), indent=2))
    t_all = time.time()
    if args.cv_check:
        res = cv_check(rel, args)
        print(json.dumps(res, indent=1, default=float), flush=True)
        (out / "cv_check.json").write_text(json.dumps(res, indent=2, default=float))
        return
    if args.check_panel:
        p = args.check_panel
        prof = dict(np.load(cache / f"{p}_prof.npz"))
        models = load_models(cache)
        res = {"panel": p}
        for split in args.splits:
            d, tpl = build_split(rel, p, split, prof)
            c = np.load(cache / f"{p}_{split}.npz")
            res[f"{split}_features_identical"] = all(np.array_equal(d[k], c[k], equal_nan=True) for k in c.files)
            new = predict_frame(d, tpl, models, args.threads)
            sub = pd.read_csv(out / f"state_{split}.csv", dtype={k: str for k in KEY})
            sub = sub[sub.panel == p].reset_index(drop=True)
            res[f"{split}_keys_identical"] = bool((sub[KEY].to_numpy() == new[KEY].to_numpy()).all())
            res[f"{split}_max_abs_diff"] = float(np.max(np.abs(np.r_[sub.speed_kmh - new.speed_kmh, sub.flow_vph - new.flow_vph])))
        res["runtime_s"] = round(time.time() - t_all, 1)
        print(json.dumps(res), flush=True)
        (out / f"repro_{p}.json").write_text(json.dumps(res, indent=2))
        return
    panels = args.panels or sorted(p.name for p in (rel / "corridors").iterdir())
    ntrain = {}
    for p in panels:
        s = time.time()
        if args.reuse_cache and (cache / f"{p}_train.npz").exists() and all((cache / f"{p}_{sp}.npz").exists() for sp in args.splits):
            continue
        prof, ntrain[p] = build_train(rel, p, args.sample_per_panel, args.seed, cache, args.aug_frac)
        for split in args.splits:
            build_split(rel, p, split, prof, cache)
        del prof
        gc.collect()
        print(f"{p}: train rows {ntrain[p]:,} | {time.time()-s:.0f}s | {({k: round(v) for k, v in tr.STAGES.items()})} | peak {tr.peak_gb():.2f} GB", flush=True)
    models, n = train_models(panels, cache, args)
    print(f"trained {MEMBERS} dv,dq on {n:,} rows | {({k: round(v) for k, v in tr.STAGES.items()})}", flush=True)
    res = {"panels": panels, "train_rows": n, "n_features": len(r2.ALLFEATS),
           "params": {m: dict(r2.MODELS[m][0], num_threads=args.threads, seed=args.seed + r2.MODELS[m][2]) for m in MEMBERS}, "rounds": args.rounds,
           "sample_per_panel": args.sample_per_panel, "aug_frac": args.aug_frac, "base": f"interp v:0.6 + mean of {MEMBERS} LGBM dv,dq (exp {ENS})"}
    frames = []
    for split in args.splits:
        s = time.time()
        parts = []
        for p in panels:
            d = dict(np.load(cache / f"{p}_{split}.npz"))
            tpl = pd.read_csv(rel / "task1" / p / split / "sample_submission_state.csv", dtype={k: str for k in KEY})[KEY]
            parts.append(predict_frame(d, tpl, models, args.threads))
            del d
        sub = pd.concat(parts, ignore_index=True)
        n_nan = int(sub[["speed_kmh", "flow_vph"]].isna().sum().sum())
        n_tpl = sum(sum(1 for _ in open(rel / "task1" / p / split / "sample_submission_state.csv")) - 1 for p in panels)
        sub.to_csv(out / f"state_{split}.csv", index=False)
        frames.append(sub)
        res[split] = {"rows": len(sub), "template_rows": n_tpl, "nan": n_nan, "speed_min": float(sub.speed_kmh.min()), "speed_max": float(sub.speed_kmh.max()),
                      "flow_min": float(sub.flow_vph.min()), "flow_max": float(sub.flow_vph.max()), "runtime_s": round(time.time() - s, 1)}
        tr.tick("predict", s)
        print(split, res[split], flush=True)
    vp = pd.concat(frames, ignore_index=True)
    vp.to_csv(out / "state_vp.csv", index=False)
    res["vp_rows"] = len(vp)
    if args.ref:
        res["vs_ref"] = rp.compare(vp, Path(args.ref))
        print(json.dumps(res["vs_ref"], indent=1), flush=True)
    res["stages"] = {k: round(v, 1) for k, v in tr.STAGES.items()}
    res["runtime_s"] = round(time.time() - t_all, 1)
    res["peak_gb"] = round(tr.peak_gb(), 2)
    (out / "run.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"total {res['runtime_s']}s peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
