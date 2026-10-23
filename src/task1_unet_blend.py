import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_unet import SCORE_PANELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--le-cache", default="cache/task1_resid2")
    ap.add_argument("--out", default="reports/phys/unet_f1")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--tags", nargs="+", default=["", "_s43", "_s44"])
    ap.add_argument("--wv", nargs="+", type=float, default=[0.33, 0.5, 0.6])
    ap.add_argument("--wq", nargs="+", type=float, default=[0.0, 0.2, 0.33])
    a = ap.parse_args()
    rel, lec, out = Path(a.release_root), Path(a.le_cache), Path(a.out)
    for p in SCORE_PANELS:
        b = np.load(lec / f"{p}_f{a.fold}_R2C_pred.npz")
        us = [np.load(lec / f"{p}_f{a.fold}_UN{t}_pred.npz") for t in a.tags]
        R = b["R"]
        assert all((u["R"] == R).all() for u in us)
        uv = np.mean([u["UN_v"] for u in us], axis=0)
        uq = np.mean([u["UN_q"] for u in us], axis=0)
        o = {"R": R, "LE_v": b["LE_v"], "LE_q": b["LE_q"], "UNavg_v": uv, "UNavg_q": uq}
        o["LECUN1_v"] = (b["LE_v"] + b["C_v"] + us[0]["UN_v"]) / 3
        o["LECUN1_q"] = (b["LE_q"] + b["C_q"] + us[0]["UN_q"]) / 3
        for wv in a.wv:
            for wq in a.wq:
                n = f"v{int(round(wv*100))}q{int(round(wq*100))}"
                o[f"{n}_v"] = ((1 - wv) / 2 * (b["LE_v"] + b["C_v"]) + wv * uv).astype(np.float32)
                o[f"{n}_q"] = ((1 - wq) / 2 * (b["LE_q"] + b["C_q"]) + wq * uq).astype(np.float32)
        np.savez(lec / f"{p}_f{a.fold}_UW_pred.npz", **o)
    rows = []
    for p in SCORE_PANELS:
        P = tr.Panel(rel, p)
        rows += tr.score_panel(rel, P, [a.fold], ["UW"], lec)
        del P
        gc.collect()
    df = pd.DataFrame(rows)
    df.to_csv(out / "unet_blend_cv.csv", index=False)
    g = df.groupby("variant")[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
    g["J"] = 0.35 * g.S_state + 0.15 * g.S_physics
    g["dJ"] = g.J - g.loc["LE", "J"]
    u = df.groupby(["variant", "panel"])[["S_state", "S_physics"]].mean()
    u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
    print(g.sort_values("J", ascending=False).round(5).to_string(), flush=True)
    print((u.J.unstack(0).sub(u.J.unstack(0)["LE"], axis=0)).round(5).T.to_string(), flush=True)


if __name__ == "__main__":
    main()
