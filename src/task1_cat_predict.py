import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_resid2_predict import KEY, SPLITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--base-vp", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=float, nargs="+", default=[0.5])
    a = ap.parse_args()
    rel, cache, out = Path(a.release_root), Path(a.cache_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    t0 = time.time()
    models = {}
    for tgt in ("dv", "dq"):
        if (out / f"cat_{tgt}.cbm").exists():
            models[tgt] = CatBoostRegressor()
            models[tgt].load_model(str(out / f"cat_{tgt}.cbm"))
            continue
        parts = [np.load(cache / f"{p}_train.npz") for p in panels]
        X = np.concatenate([d["X"] for d in parts])
        B = np.concatenate([d["B"] for d in parts]).astype(float)
        y = np.concatenate([d["yv" if tgt == "dv" else "yq"] for d in parts]).astype(float) - B[:, 3 if tgt == "dv" else 1]
        del parts, B
        gc.collect()
        s = time.time()
        m = CatBoostRegressor(iterations=a.iters, depth=a.depth, learning_rate=a.lr, loss_function="RMSE", l2_leaf_reg=5,
                              thread_count=a.threads, random_seed=a.seed, verbose=250, allow_writing_files=False)
        m.fit(X, y)
        m.save_model(str(out / f"cat_{tgt}.cbm"))
        models[tgt] = m
        print(f"{tgt} trained on {len(X):,} rows {time.time()-s:.0f}s", flush=True)
        del X, y
        gc.collect()
    frames = []
    for split in SPLITS:
        for p in panels:
            d = dict(np.load(cache / f"{p}_{split}.npz"))
            tpl = pd.read_csv(rel / "task1" / p / split / "sample_submission_state.csv", dtype={k: str for k in KEY})[KEY]
            B = d["B"].astype(float)
            v, q = tr.finalize(d["vc"], B[:, 3] + models["dv"].predict(d["X"], thread_count=a.threads), B[:, 1] + models["dq"].predict(d["X"], thread_count=a.threads))
            sp, fl = np.full(len(tpl), np.nan), np.full(len(tpl), np.nan)
            sp[d["R"]], fl[d["R"]] = v, q
            frames.append(tpl.assign(speed_kmh=sp, flow_vph=fl))
        print(split, "predicted", flush=True)
    cat = pd.concat(frames, ignore_index=True)
    cat.to_csv(out / "state_vp_cat.csv", index=False)
    base = pd.read_csv(a.base_vp, dtype={k: str for k in KEY})
    assert (base[KEY].to_numpy() == cat[KEY].to_numpy()).all()
    info = {"rows": len(cat), "nan_cat": int(cat[["speed_kmh", "flow_vph"]].isna().sum().sum()),
            "rmse_speed_vs_base": float(np.sqrt(np.nanmean((cat.speed_kmh - base.speed_kmh) ** 2))),
            "rmse_flow_vs_base": float(np.sqrt(np.nanmean((cat.flow_vph - base.flow_vph) ** 2)))}
    for w in a.weights:
        b = base.copy()
        b["speed_kmh"] = (1 - w) * base.speed_kmh + w * cat.speed_kmh
        b["flow_vph"] = (1 - w) * base.flow_vph + w * cat.flow_vph
        b.to_csv(out / f"state_vp_lec{int(w*100)}.csv", index=False)
    info["runtime_s"] = round(time.time() - t0)
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
