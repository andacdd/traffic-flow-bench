import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr

FAMILY = lambda p: p.rsplit("_", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--qmult", type=float, default=1.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel, cache, work = Path(a.release_root), Path(a.cache), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    rows = []
    for p in panels:
        for f in [1, 2, 3]:
            z = np.load(cache / f"{p}_f{f}_B3_pred.npz")
            np.savez(work / f"{p}_f{f}_PX_pred.npz", R=z["R"], base_v=z["B3_v"], base_q=z["B3_q"], qx_v=z["B3_v"], qx_q=(z["B3_q"] * a.qmult).astype(np.float32))
        P = tr.Panel(rel, p)
        rows += tr.score_panel(rel, P, [1, 2, 3], ["PX"], work)
        del P
        gc.collect()
        print(p, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    met = ["S_state", "S_FD", "S_LWR", "S_physics"]
    g = df.assign(fam=df.panel.map(FAMILY)).groupby(["variant", "fam"])[met].mean().groupby("variant").mean()
    g["J"] = 0.35 * g.S_state + 0.15 * g.S_physics
    print(g.round(5).to_string())
    d = g.loc["qx"] - g.loc["base"]
    print("delta qx-base:", d.round(5).to_dict())


if __name__ == "__main__":
    main()
