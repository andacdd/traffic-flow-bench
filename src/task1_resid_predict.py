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
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_interp import T, link_order

KEY = ["panel", "timestamp", "station_id", "link_id", "mask_regime"]
SPLITS = ["validation", "private"]


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "lightgbm": lgb.__version__, "scipy": scipy.__version__}


class SplitPanel(tr.Panel):
    def __init__(self, rel, panel, split):
        s = time.time()
        self.name = panel
        pdir = rel / "corridors" / panel
        self.order, self.pos = link_order(rel, panel)
        self.L = len(self.order)
        lix = pd.Index(self.order)
        mfiles = sorted((pdir / split / "mainline_states_masked").glob("**/*.parquet"))
        dates = sorted({pq.ParquetFile(f).read(columns=["date"])["date"][0].as_py().__str__()[:10] for f in mfiles})
        self.days = pd.Index(dates)
        self.D = len(dates)
        self.dow = np.array([pd.Timestamp(d).weekday() for d in dates])
        shape = (self.D, T, self.L)
        self.obs = {c: np.full(shape, np.nan) for c in tr.VARS}
        for f in mfiles:
            t = pq.ParquetFile(f).read(columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph"])
            di, tod = tr.grid_index(t, self.days)
            li = lix.get_indexer(t["link_id"].to_numpy(zero_copy_only=False))
            ok = (li >= 0) & (di >= 0)
            sp = t["speed_kmh"].to_numpy(zero_copy_only=False).astype(float)
            fl = t["flow_vph"].to_numpy(zero_copy_only=False).astype(float)
            self.obs["v"][di[ok], tod[ok], li[ok]] = sp[ok]
            self.obs["q"][di[ok], tod[ok], li[ok]] = fl[ok]
            self.obs["k"][di[ok], tod[ok], li[ok]] = (fl / np.clip(sp, 1.0, None))[ok]
        tpl = pcsv.read_csv(
            rel / "task1" / panel / split / "sample_submission_state.csv",
            convert_options=pcsv.ConvertOptions(include_columns=KEY, column_types={c: "string" for c in KEY}),
        )
        self.tpl = tpl.to_pandas()
        tpl = tpl.append_column("date", pc.utf8_slice_codeunits(tpl["timestamp"], 0, 10))
        di, tod = tr.grid_index(tpl, self.days)
        li = lix.get_indexer(pd.Series(tpl["link_id"].to_numpy(zero_copy_only=False)))
        self.n_tpl_drop = int(((li < 0) | (di < 0)).sum())
        self.td, self.tt, self.tl = di, tod, li
        self.treg = pd.Series(tpl["mask_regime"].to_numpy(zero_copy_only=False)).map(tr.REG).to_numpy()
        fd = pd.read_csv(pdir / "network" / "fd_parameters.csv").set_index("link_id")
        topo = pd.read_csv(pdir / "network" / "lwr_mainline_topology.csv").set_index("mainline_link_id").loc[self.order]
        self.geo = {c: fd[c].reindex(self.order).to_numpy(dtype=float) for c in tr.GEO}
        self.vf = fd.free_speed_kmh.reindex(self.order).fillna(117.0).to_numpy()
        self.vclip = np.minimum(fd.free_speed_kmh.reindex(self.order).fillna(120.75).to_numpy() + 4.5, 120.75)
        self.geo["pos_frac"] = self.pos / max(self.pos[-1], 1e-9)
        self.geo["has_sensor"] = topo.has_sensor.to_numpy(dtype=float)
        self.ramp_on, self.ramp_off = split_ramps(self, pdir, topo, split)
        self.day_rows = pd.Series(np.arange(len(self.td))).groupby(self.td).indices
        tr.tick("split_load", s)


def split_ramps(P, pdir, topo, split):
    files = sorted((pdir / split / "ramp_states").glob("**/*.parquet"))
    rid = sorted(set(";".join(topo.on_ramp_link_ids.fillna("").astype(str).tolist() + topo.off_ramp_link_ids.fillna("").astype(str).tolist()).split(";")) - {""})
    rix = pd.Index(rid)
    R = np.full((P.D, T, len(rid)), np.nan, dtype=np.float32)
    for f in files:
        t = pq.ParquetFile(f).read(columns=["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
        di, tod = tr.grid_index(t, P.days)
        ri = rix.get_indexer(t["ramp_link_id"].to_numpy(zero_copy_only=False))
        ok = (ri >= 0) & (di >= 0) & (t["pct_observed"].to_numpy() >= 75) & (t["is_score_eligible"].to_numpy() == 1)
        R[di[ok], tod[ok], ri[ok]] = t["flow_vph"].to_numpy(zero_copy_only=False)[ok]
    out = []
    for col in ["on_ramp_link_ids", "off_ramp_link_ids"]:
        arr = np.zeros((P.D, T, P.L), dtype=np.float32)
        for j, ids in enumerate(topo[col].fillna("").astype(str)):
            k = [rix.get_loc(x) for x in ids.split(";") if x in rix]
            if k:
                arr[:, :, j] = R[:, :, k].sum(axis=2) if len(k) > 1 else R[:, :, k[0]]
        out.append(arr)
    return out


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
        Xs.append(X)
        Bs.append(B)
        Rs.append(rows)
    return np.concatenate(Xs), np.concatenate(Bs), np.concatenate(Rs)


def build_train(rel, p, n_sample, seed, cache):
    s = time.time()
    P = tr.Panel(rel, p)
    prof = P.profiles(np.ones(P.D, dtype=bool))
    rng = np.random.default_rng(seed)
    pick = np.zeros(len(P.td), dtype=bool)
    pick[rng.choice(len(P.td), size=min(n_sample, len(P.td)), replace=False)] = True
    X, B, R = all_rows(P, prof, pick)
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = np.isfinite(yv) & np.isfinite(yq)
    np.savez(cache / f"{p}_train.npz", X=X[good], B=B[good], yv=yv[good], yq=yq[good])
    np.savez(cache / f"{p}_prof.npz", **prof)
    tr.tick("train_features", s)
    n = int(good.sum())
    del P
    gc.collect()
    return prof, n


def build_split(rel, p, split, prof, cache=None):
    s = time.time()
    SP = SplitPanel(rel, p, split)
    assert SP.n_tpl_drop == 0, (p, split, SP.n_tpl_drop)
    X, B, R = all_rows(SP, prof)
    assert len(R) == len(SP.tpl) and np.array_equal(np.sort(R), np.arange(len(SP.tpl)))
    out = {"X": X, "B": B, "R": R, "vf": SP.vf[SP.tl[R]], "vc": SP.vclip[SP.tl[R]]}
    if cache is not None:
        np.savez(cache / f"{p}_{split}.npz", **out)
    tr.tick("split_features", s)
    return out, SP.tpl


def predict_frame(d, tpl, models):
    B = d["B"].astype(float)
    v, q = tr.finalize(d["vc"], B[:, 3] + models["dv"].predict(d["X"]), B[:, 1] + models["dq"].predict(d["X"]))
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
    models = {}
    for name, y in [("dv", yv - B[:, 3]), ("dq", yq - B[:, 1])]:
        models[name] = tr.train(X, y, args.threads, args.rounds, args.seed)
        models[name].save_model(str(cache / f"model_{name}.txt"))
    n = len(X)
    del X
    gc.collect()
    tr.tick("train", s)
    return models, n


def compare(new, ref_path):
    ref = pd.read_csv(ref_path, dtype={c: str for c in KEY})
    same_keys = bool((ref[KEY].to_numpy() == new[KEY].to_numpy()).all()) if len(ref) == len(new) else False
    if not same_keys:
        ref = new[KEY].merge(ref, on=KEY, how="left")
    out = {"ref": str(ref_path), "same_key_order": same_keys}
    for c in ["speed_kmh", "flow_vph"]:
        d = new[c].to_numpy() - ref[c].to_numpy()
        out[f"mean_abs_diff_{c}"] = float(np.nanmean(np.abs(d)))
        out[f"mean_{c}_new"] = float(new[c].mean())
        out[f"mean_{c}_ref"] = float(ref[c].mean())
        out[f"share_changed_{c}"] = float(np.mean(np.abs(d) > 1e-6))
    out["share_cells_changed"] = float(np.mean((np.abs(new.speed_kmh.to_numpy() - ref.speed_kmh.to_numpy()) > 1e-6) | (np.abs(new.flow_vph.to_numpy() - ref.flow_vph.to_numpy()) > 1e-6)))
    by = new.assign(dv=np.abs(new.speed_kmh.to_numpy() - ref.speed_kmh.to_numpy()), dq=np.abs(new.flow_vph.to_numpy() - ref.flow_vph.to_numpy()))
    out["by_regime"] = by.groupby("mask_regime")[["dv", "dq"]].mean().round(4).to_dict()
    return out


def cv_check(rel, args, cache, scratch):
    s = time.time()
    p = "D12_I5_N"
    panels = sorted(x.name for x in (rel / "corridors").iterdir())
    src = Path(args.cv_cache)
    fc = scratch / "cv_feat"
    fc.mkdir(parents=True, exist_ok=True)
    P = tr.Panel(rel, p)
    tr.build_fold(P, 1, args.sample_per_panel, args.seed, fc)
    a, b = np.load(fc / f"{p}_f1.npz"), np.load(src / f"{p}_f1.npz")
    feat_diff = {k: bool(np.array_equal(a[k], b[k], equal_nan=True)) for k in b.files}
    sc = scratch / "cv_cache"
    sc.mkdir(parents=True, exist_ok=True)
    for q in panels:
        dst = sc / f"{q}_f1.npz"
        if not dst.exists():
            dst.symlink_to(src / f"{q}_f1.npz")
    imp = []
    models = tr.fit_fold(panels, 1, ["A"], args, sc, imp)
    tr.predict_panel(p, 1, ["A"], models, sc)
    rows = pd.DataFrame(tr.score_panel(rel, P, [1], ["A"], sc))
    ref = pd.read_csv(Path(args.cv_ref))
    ref = ref[(ref.fold == 1) & (ref.panel == p)].set_index("regime")
    rows = rows.set_index("regime")
    met = ["S_state", "rmse_v", "rmse_q_lane", "S_FD", "E_LWR", "S_LWR", "S_physics", "q_rmse_v", "q_mae_k"]
    diff = (rows[met] - ref.loc[rows.index, met]).abs()
    res = {"features_identical": feat_diff, "max_abs_diff": float(diff.to_numpy().max()), "per_metric": diff.max().to_dict(),
           "new": rows[["S_state", "S_physics"]].round(6).to_dict(), "runtime_s": round(time.time() - s, 1)}
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
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--ref")
    ap.add_argument("--check-panel")
    ap.add_argument("--cv-check", action="store_true")
    ap.add_argument("--cv-cache")
    ap.add_argument("--cv-ref")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out, cache = Path(args.out_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (out / "versions.json").write_text(json.dumps(versions(), indent=2))
    t_all = time.time()
    if args.cv_check:
        res = cv_check(rel, args, cache, cache / "cvcheck")
        print(json.dumps(res, indent=1, default=float), flush=True)
        (out / "cv_check.json").write_text(json.dumps(res, indent=2, default=float))
        return
    if args.check_panel:
        p = args.check_panel
        prof = dict(np.load(cache / f"{p}_prof.npz"))
        models = {n: lgb.Booster(model_file=str(cache / f"model_{n}.txt")) for n in ["dv", "dq"]}
        res = {"panel": p}
        for split in args.splits:
            d, tpl = build_split(rel, p, split, prof)
            c = np.load(cache / f"{p}_{split}.npz")
            res[f"{split}_features_identical"] = all(np.array_equal(d[k], c[k], equal_nan=True) for k in c.files)
            new = predict_frame(d, tpl, models)
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
        prof, ntrain[p] = build_train(rel, p, args.sample_per_panel, args.seed, cache)
        for split in args.splits:
            build_split(rel, p, split, prof, cache)
        del prof
        gc.collect()
        print(f"{p}: train rows {ntrain[p]:,} | {time.time()-s:.0f}s | {({k: round(v) for k, v in tr.STAGES.items()})} | peak {tr.peak_gb():.2f} GB", flush=True)
    models, n = train_models(panels, cache, args)
    print(f"trained dv,dq on {n:,} rows | {({k: round(v) for k, v in tr.STAGES.items()})}", flush=True)
    res = {"panels": panels, "train_rows": n, "params": dict(tr.PARAMS, num_threads=args.threads, seed=args.seed), "rounds": args.rounds,
           "sample_per_panel": args.sample_per_panel, "base": "interp v:0.6 + LGBM dv,dq (exp A)"}
    frames = []
    for split in args.splits:
        s = time.time()
        parts = []
        for p in panels:
            d = dict(np.load(cache / f"{p}_{split}.npz"))
            tpl = pd.read_csv(rel / "task1" / p / split / "sample_submission_state.csv", dtype={k: str for k in KEY})[KEY]
            parts.append(predict_frame(d, tpl, models))
        sub = pd.concat(parts, ignore_index=True)
        n_nan = int(sub[["speed_kmh", "flow_vph"]].isna().sum().sum())
        n_tpl = sum(sum(1 for _ in open(rel / "task1" / p / split / "sample_submission_state.csv")) - 1 for p in panels)
        sub.to_csv(out / f"state_{split}.csv", index=False)
        frames.append(sub)
        res[split] = {"rows": len(sub), "template_rows": n_tpl, "nan": n_nan, "speed_min": float(sub.speed_kmh.min()), "speed_max": float(sub.speed_kmh.max()),
                      "flow_min": float(sub.flow_vph.min()), "runtime_s": round(time.time() - s, 1)}
        print(split, res[split], flush=True)
    vp = pd.concat(frames, ignore_index=True)
    vp.to_csv(out / "state_vp.csv", index=False)
    res["vp_rows"] = len(vp)
    if args.ref:
        res["vs_ref"] = compare(vp, Path(args.ref))
        print(json.dumps(res["vs_ref"], indent=1), flush=True)
    res["stages"] = {k: round(v, 1) for k, v in tr.STAGES.items()}
    res["runtime_s"] = round(time.time() - t_all, 1)
    res["peak_gb"] = round(tr.peak_gb(), 2)
    (out / "run.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"total {res['runtime_s']}s peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
