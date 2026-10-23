import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    rows = []
    for panel in tm.PANELS:
        order, vf, _, _ = tm.network(rel, panel)
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        months = np.array([d[:7] for d in dates])
        Q = (F[:, 0] <= 0.6 * vf).astype(np.float32)
        Q[np.isnan(F[:, 0])] = np.nan
        lomo = {m: np.nan_to_num(np.nanmean(Q[months != m], 0)) for m in set(months)}
        di = {d: i for i, d in enumerate(dates)}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + tm.H >= tm.T:
                continue
            Y = np.nan_to_num(Q[d, t0 + 1:t0 + tm.H + 1])
            fr = lomo[r.month][t0 + 1:t0 + tm.H + 1]
            n = Y.sum()
            rows.append({"window_id": r.window_id, "panel": panel, "condition": r.condition, "month": r.month,
                         "rare_share": float((Y * (fr < 0.02)).sum() / max(n, 1)), "mean_freq": float((Y * fr).sum() / max(n, 1))})
        print(panel, flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
