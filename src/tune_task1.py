import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
import task1_interp as ti


def truth(rel, panel, tpl):
    files = sorted((rel / "corridors" / panel / "train" / "mainline_states").glob("**/*.parquet"))
    df = pd.concat([pq.ParquetFile(f).read(columns=["timestamp", "link_id", "speed_kmh", "flow_vph"]).to_pandas() for f in files])
    m = tpl[["timestamp", "link_id", "mask_regime"]].merge(df, on=["timestamp", "link_id"], how="left")
    lanes = pd.read_csv(rel / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").lanes
    m["lanes"] = m.link_id.map(lanes).to_numpy()
    return m


def score(pred, tr):
    out = []
    for r in ["R1", "R2", "R3"]:
        k = (tr.mask_regime == r).to_numpy()
        es = np.sqrt(np.nanmean((pred.speed_kmh.to_numpy()[k] - tr.speed_kmh.to_numpy()[k]) ** 2))
        eq = np.sqrt(np.nanmean(((pred.flow_vph.to_numpy()[k] - tr.flow_vph.to_numpy()[k]) / tr.lanes.to_numpy()[k]) ** 2))
        out.append((0.54 * max(0, 1 - es / 25) + 0.46 * max(0, 1 - eq / 600), es, eq))
    return np.mean([o[0] for o in out]), np.mean([o[1] for o in out]), np.mean([o[2] for o in out])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--panel", default="D12_I5_N")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    order, _ = ti.link_order(rel, a.panel)
    prof = ti.profiles(ti.masked_frame(rel, a.panel, "train"), order)
    grid = {"t0": [0.5, 1.0, 2.0], "p": [1.0, 2.0], "s_km": [0.25, 0.5, 1.0, 1e9], "w_prof": [0.0, 0.02, 0.1]}
    rows, tr = [], None
    for vals in itertools.product(*grid.values()):
        prm = dict(zip(grid.keys(), vals))
        s = time.time()
        pred, _ = ti.build(rel, a.panel, "train", prm, train_prof=prof)
        if tr is None:
            tr = truth(rel, a.panel, pred)
        sc = score(pred, tr)
        rows.append({**prm, "S_state": sc[0], "rmse_v": sc[1], "rmse_q_lane": sc[2]})
        print(rows[-1], f"{time.time()-s:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
