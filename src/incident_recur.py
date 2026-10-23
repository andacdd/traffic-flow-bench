import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

LAGS = [1, 3, 7]


def main():
    rel = Path(sys.argv[1])
    out = Path(sys.argv[2])
    res = []
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        _, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        freq = np.nanmean(np.stack([tm.queue_matrix(F[d, 0], vf) for d in range(len(F))]), 0)
        del F
        rare = freq < 0.02
        for split in ["train", "validation", "private"]:
            sub = "mainline_states" if split == "train" else "mainline_states_masked"
            dates, A = tm.load_days(rel, panel, split, sub, order)
            Q = np.stack([tm.queue_matrix(A[d, 0], vf) for d in range(len(dates))])
            del A
            ords = pd.to_datetime(dates).values.astype("datetime64[D]").astype(np.int64)
            o = np.argsort(ords)
            Q, ords = Q[o], ords[o]
            obs = ~np.isnan(Q)
            R = (np.nan_to_num(Q) > 0) & rare[None]
            link = R.any(1)
            lobs = obs.any(1)
            for L in LAGS:
                prior = np.zeros_like(link)
                for d in range(len(dates)):
                    prior[d] = link[(ords >= ords[d] - L) & (ords < ords[d])].any(0)
                m = lobs & rare.any(0)[None]
                res.append({"panel": panel, "split": split, "lag": L,
                            "p_given_prior": link[prior & m].mean() if (prior & m).any() else np.nan,
                            "p_base": link[m].mean(), "n_prior": int((prior & m).sum())})
            print(panel, split, flush=True)
    r = pd.DataFrame(res)
    r.to_csv(out, index=False)
    g = r.groupby(["split", "lag"])[["p_given_prior", "p_base"]].mean()
    g["lift"] = g.p_given_prior / g.p_base
    print(g.round(4).to_string())


if __name__ == "__main__":
    main()
