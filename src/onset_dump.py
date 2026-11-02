import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from online_refit import WEEKS, build, fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--out", default="reports/queue/onset_struct/onset_dump.parquet")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--weight", type=float, default=3.0)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tr_on, _, ev_on, _ = build(rel, 1)
    res = []
    for k in range(1, len(WEEKS)):
        te = ev_on[ev_on.wk == k]
        if te.empty:
            continue
        m, cols = fit(tr_on[(tr_on.src == 0) | (tr_on.wk < k)], a.rounds, a.weight, True)
        b, bcols = fit(tr_on[tr_on.src == 0], a.rounds, 1.0, False)
        p, pb = m.predict(te[cols]), b.predict(te[bcols])
        for win, idx in te.groupby("win").indices.items():
            res.append(pd.DataFrame({"wk": k, "win": win, "pid": int(win.split("|")[1]), "c": np.arange(len(idx)),
                                     "p": p[idx], "p_base": pb[idx], "y": te.y.to_numpy()[idx], "freq": te.freq_max_h.to_numpy()[idx]}))
        print("week", k, te.win.nunique(), flush=True)
    pd.concat(res, ignore_index=True).to_parquet(out, index=False)


if __name__ == "__main__":
    main()
