import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
import task1_resid2 as r2
import task1_resid2_predict as base
import task1_resid3 as r3

KEY = base.KEY
SPLITS = base.SPLITS
MEMBERS = base.MEMBERS
VARIANT = "B3"


def build_train(rel, p, n_sample, aug_frac, seed, cache):
    s = time.time()
    P = tr.Panel(rel, p)
    hold = np.zeros(P.D, dtype=bool)
    prof = P.profiles(~hold)
    X, B, R = base.all_rows(P, prof, base.pick_rows(P, hold, n_sample, seed))
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = np.isfinite(yv) & np.isfinite(yq)
    np.savez(cache / f"{p}_train.npz", X=X[good], B=B[good], yv=yv[good], yq=yq[good])
    np.savez(cache / f"{p}_prof.npz", **prof)
    n = int(good.sum())
    del X, B, R
    gc.collect()
    tr.tick("train_features", s)
    ag = r3.aug_rows(P, 0, hold, P.profiles(~hold), int(aug_frac * n), seed)
    np.savez(cache / f"{p}_aug.npz", X=np.hstack([ag["XA"], ag["XL"]]), B=ag["B"], yv=ag["yv"], yq=ag["yq"])
    info = {"train_rows": n, "aug_rows": len(ag["yv"]), "aug_candidates": ag["n_cand"]}
    del P, ag
    gc.collect()
    return prof, info


def train_models(panels, cache, args):
    s = time.time()
    files = [cache / f"{p}_{k}.npz" for p in panels for k in ["train", "aug"]]
    n = sum(len(np.load(f)["yv"]) for f in files)
    X = np.empty((n, len(r2.ALLFEATS)), dtype=np.float32)
    B, yv, yq = np.empty((n, 4)), np.empty(n), np.empty(n)
    i = 0
    for f in files:
        d = np.load(f)
        j = i + len(d["yv"])
        X[i:j], B[i:j], yv[i:j], yq[i:j] = d["X"], d["B"], d["yv"], d["yq"]
        i = j
        del d
    assert i == n
    print(f"train matrix {X.shape} peak {tr.peak_gb():.2f} GB", flush=True)
    models = base.fit_members(X, B, yv, yq, args, cache)
    del X
    gc.collect()
    tr.tick("train", s)
    return models, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--panels", nargs="+")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--sample-per-panel", type=int, default=300000)
    ap.add_argument("--aug-frac", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--ref")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out, cache = Path(args.out_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    (out / "versions.json").write_text(json.dumps(base.rp.versions(), indent=2))
    t_all = time.time()
    panels = args.panels or sorted(p.name for p in (rel / "corridors").iterdir())
    info = {}
    for p in panels:
        s = time.time()
        names = ["train", "aug", "prof"] + list(args.splits)
        if args.reuse_cache and all((cache / f"{p}_{k}.npz").exists() for k in names):
            continue
        prof, info[p] = build_train(rel, p, args.sample_per_panel, args.aug_frac, args.seed, cache)
        for split in args.splits:
            base.build_split(rel, p, split, prof, cache)
        del prof
        gc.collect()
        print(f"{p}: {info[p]} | {time.time()-s:.0f}s | {({k: round(v) for k, v in tr.STAGES.items()})} | peak {tr.peak_gb():.2f} GB", flush=True)
    models, n = train_models(panels, cache, args)
    print(f"trained {MEMBERS} dv,dq on {n:,} rows | {({k: round(v) for k, v in tr.STAGES.items()})}", flush=True)
    res = {"panels": panels, "train_rows": n, "panel_rows": info, "n_features": len(r2.ALLFEATS),
           "params": {m: dict(r2.MODELS[m][0], num_threads=args.threads, seed=args.seed + r2.MODELS[m][2]) for m in MEMBERS}, "rounds": args.rounds,
           "sample_per_panel": args.sample_per_panel, "aug_frac": args.aug_frac,
           "base": f"interp v:0.6 + mean of {MEMBERS} LGBM dv,dq trained with mask augmentation (exp {VARIANT})"}
    frames = []
    for split in args.splits:
        s = time.time()
        parts = []
        for p in panels:
            d = dict(np.load(cache / f"{p}_{split}.npz"))
            tpl = pd.read_csv(rel / "task1" / p / split / "sample_submission_state.csv", dtype={k: str for k in KEY})[KEY]
            parts.append(base.predict_frame(d, tpl, models, args.threads))
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
        res["vs_ref"] = base.rp.compare(vp, Path(args.ref))
        print(json.dumps(res["vs_ref"], indent=1), flush=True)
    res["stages"] = {k: round(v, 1) for k, v in tr.STAGES.items()}
    res["runtime_s"] = round(time.time() - t_all, 1)
    res["peak_gb"] = round(tr.peak_gb(), 2)
    (out / "run.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"total {res['runtime_s']}s peak {res['peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
