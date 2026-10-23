import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
import task1_resid_predict as rp
from task1_interp import T
from task1_unet import Grid, UNet, predict_grid

KEY, SPLITS = rp.KEY, rp.SPLITS


def split_grid(SP, train_grid):
    g = Grid.__new__(Grid)
    g.D, g.L = SP.D, SP.L
    g.ov, g.oq = SP.obs["v"].astype(np.float32), SP.obs["q"].astype(np.float32)
    g.reg = np.zeros((SP.D, T, SP.L), dtype=np.uint8)
    g.reg[SP.td, SP.tt, SP.tl] = SP.treg
    g.ron, g.roff = np.nan_to_num(SP.ramp_on), np.nan_to_num(SP.ramp_off)
    g.dow = SP.dow
    for k in ("pv", "pq", "geo", "lanes", "vclip"):
        setattr(g, k, getattr(train_grid, k))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--grid-dir", default="cache/task1_unet")
    ap.add_argument("--model-dir", default="submissions/state_unet")
    ap.add_argument("--tags", nargs="+", default=["_s42", "_s43", "_s44"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--le-vp", default="submissions/runall_b3/state/state_vp.csv")
    ap.add_argument("--cat-vp", default="submissions/state_cat/state_vp_cat.csv")
    ap.add_argument("--wv", type=float, default=0.33)
    ap.add_argument("--wq", type=float, default=0.2)
    a = ap.parse_args()
    rel, gd, md = Path(a.release_root), Path(a.grid_dir), Path(a.model_dir)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models = []
    for t in a.tags:
        m = UNet(a.width).to(dev)
        m.load_state_dict(torch.load(md / f"unet_f0{t}.pt", map_location=dev))
        models.append(m)
    t0 = time.time()
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    frames = []
    for split in SPLITS:
        for p in panels:
            s = time.time()
            tg = Grid(gd / f"{p}_f0.npz")
            SP = rp.SplitPanel(rel, p, split)
            assert SP.n_tpl_drop == 0
            g = split_grid(SP, tg)
            days = np.unique(SP.td)
            Vs, Qs = zip(*(predict_grid(m, g, days, dev) for m in models))
            V, Q = np.mean(Vs, axis=0), np.mean(Qs, axis=0)
            v, q = tr.finalize(g.vclip[SP.tl], V[SP.td, SP.tt, SP.tl], Q[SP.td, SP.tt, SP.tl])
            frames.append(SP.tpl[KEY].assign(speed_kmh=v, flow_vph=q))
            print(split, p, f"{time.time()-s:.0f}s", flush=True)
            del tg, SP, g, Vs, Qs, V, Q
            gc.collect()
    un = pd.concat(frames, ignore_index=True)
    un.to_csv(md / "state_vp_unet.csv", index=False)
    le = pd.read_csv(a.le_vp, dtype={k: str for k in KEY})
    cat = pd.read_csv(a.cat_vp, dtype={k: str for k in KEY})
    assert (le[KEY].to_numpy() == un[KEY].to_numpy()).all() and (le[KEY].to_numpy() == cat[KEY].to_numpy()).all()
    b = le.copy()
    b["speed_kmh"] = (1 - a.wv) / 2 * (le.speed_kmh + cat.speed_kmh) + a.wv * un.speed_kmh
    b["flow_vph"] = (1 - a.wq) / 2 * (le.flow_vph + cat.flow_vph) + a.wq * un.flow_vph
    name = f"state_vp_v{int(round(a.wv*100))}q{int(round(a.wq*100))}.csv"
    b.to_csv(md / name, index=False)
    info = {"rows": len(un), "nan_un": int(un[["speed_kmh", "flow_vph"]].isna().sum().sum()),
            "rmse_speed_un_vs_le": float(np.sqrt(np.mean((un.speed_kmh - le.speed_kmh) ** 2))),
            "rmse_flow_un_vs_le": float(np.sqrt(np.mean((un.flow_vph - le.flow_vph) ** 2))),
            "blend": name, "args": vars(a), "runtime_s": round(time.time() - t0)}
    (md / "predict.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
