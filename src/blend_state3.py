import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["panel", "timestamp", "station_id", "link_id"]


def load(path):
    return pd.read_csv(path, dtype={"task": str}).sort_values("submission_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--base", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--third", nargs="+", required=True)
    ap.add_argument("--w", type=float, nargs=3, default=[1 / 3, 1 / 3, 1 / 3])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    base, O, A = load(a.base), load(a.ours), load(a.pub)
    assert (base.submission_id.values == O.submission_id.values).all() and (O.submission_id.values == A.submission_id.values).all()
    key = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task"] + KEY, dtype=str)
    key["submission_id"] = key.submission_id.astype(int)
    key = key.sort_values("submission_id").reset_index(drop=True)
    st = (base.task == "state").to_numpy()
    ks = key[st]
    Y = pd.concat([pd.read_csv(f, dtype={c: str for c in KEY}) for f in a.third], ignore_index=True)
    Y = Y.drop_duplicates(KEY).set_index(KEY).reindex(pd.MultiIndex.from_frame(ks[KEY]))
    assert Y.speed_kmh.notna().all() and Y.flow_vph.notna().all(), int(Y.speed_kmh.isna().sum())
    out = base.copy()
    wo, wa, wy = a.w
    info = {"args": vars(a)}
    for c in ("speed_kmh", "flow_vph"):
        y = Y[c].to_numpy(float)
        out.loc[st, c] = wo * O.loc[st, c].to_numpy() + wa * A.loc[st, c].to_numpy() + wy * y
        info[f"rmse_{c}_third_vs_ours"] = float(np.sqrt(np.mean((y - O.loc[st, c].to_numpy()) ** 2)))
        info[f"rmse_{c}_third_vs_pub"] = float(np.sqrt(np.mean((y - A.loc[st, c].to_numpy()) ** 2)))
        info[f"rmse_{c}_vs_base"] = float(np.sqrt(np.mean((out.loc[st, c].to_numpy() - base.loc[st, c].to_numpy()) ** 2)))
    info["non_state_identical"] = bool((out.loc[~st].fillna(-1).values == base.loc[~st].fillna(-1).values).all())
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "submission.csv", index=False)
    (od / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
