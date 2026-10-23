import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm
from online_refit import RARE


def main():
    rel, bt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    r = pd.read_csv(bt)
    wins = r[["win", "cond"]].drop_duplicates()
    p = wins.win.str.split("|", expand=True)
    wins["pid"], wins["date"], wins["t0"] = p[1].astype(int), p[2], p[3].astype(int)
    share = {}
    for pid, panel in enumerate(tm.PANELS):
        P = td.load_panel(rel, pid, panel)
        rare = np.nan_to_num(np.nanmean(P["Q"], 0)) < RARE
        vd, VA = tm.load_days(rel, panel, "validation", "mainline_states_masked", P["order"])
        di = {d: i for i, d in enumerate(vd)}
        for x in wins[wins.pid == pid].itertuples():
            Q = tm.queue_matrix(VA[di[x.date], 0], P["vf"])
            hs = slice(x.t0 + tm.H, x.t0 + tm.H + 1) if x.cond == "onset" else slice(x.t0 + 1, x.t0 + tm.H + 1)
            y = np.nan_to_num(Q[hs]) > 0
            share[(x.win, x.cond)] = y[rare[hs]].sum() / y.sum() if y.sum() else np.nan
        del P
    r["rare_share"] = [share[k] for k in zip(r.win, r.cond)]
    r["kind"] = np.where(r.rare_share.isna(), "empty", np.where(r.rare_share > 0.5, "rare", "normal"))
    r.to_csv(out, index=False)
    t = r.groupby(["cond", "kind", "variant"]).iou.mean().unstack().round(4)
    t["n"] = r[r.variant == "base"].groupby(["cond", "kind"]).size()
    print(t.to_string())


if __name__ == "__main__":
    main()
