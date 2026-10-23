import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    return pd.read_csv(path, dtype={"task": str}).sort_values("submission_id").reset_index(drop=True)


def conditions(rel, sub):
    k = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp"], dtype=str)
    k = k[k.task == "queue"]
    wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(rel / "task2" / "*" / "*" / "window_index.csv"))])
    k["condition"] = k.window_id.map(dict(zip(wi.window_id, wi.condition)))
    k["split"] = k.window_id.map(dict(zip(wi.window_id, wi.split)))
    k["submission_id"] = k.submission_id.astype(int)
    q = sub[["submission_id"]].merge(k, on="submission_id", how="left")
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--state-w", type=float, default=0.5)
    ap.add_argument("--onset", choices=["ours", "theirs", "union", "inter"], default="ours")
    ap.add_argument("--ongoing", choices=["ours", "theirs", "union", "inter"], default="ours")
    ap.add_argument("--odme-w", type=float, default=0.0)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = load(a.ours), load(a.theirs)
    assert (A.submission_id.values == B.submission_id.values).all() and (A.task.values == B.task.values).all()
    C = A.copy()
    st = (C.task == "state").to_numpy()
    for c in ["speed_kmh", "flow_vph"]:
        C.loc[st, c] = (1 - a.state_w) * A.loc[st, c] + a.state_w * B.loc[st, c]
    od = (C.task == "odme").to_numpy()
    C.loc[od, "path_flow"] = (1 - a.odme_w) * A.loc[od, "path_flow"] + a.odme_w * B.loc[od, "path_flow"]
    qm = (C.task == "queue").to_numpy()
    cond = conditions(rel, C[qm])
    qa, qb = A.loc[qm, "queue_pred"].to_numpy() > 0, B.loc[qm, "queue_pred"].to_numpy() > 0
    rule = {"ours": qa, "theirs": qb, "union": qa | qb, "inter": qa & qb}
    new = qa.copy()
    for name, mode in (("queue_onset", a.onset), ("queue_ongoing", a.ongoing)):
        m = (cond.condition == name).to_numpy()
        new[m] = rule[mode][m]
    C.loc[qm, "queue_pred"] = new.astype(int)
    stats = {}
    for name in ("queue_onset", "queue_ongoing"):
        m = (cond.condition == name).to_numpy()
        stats[name] = {"ours_pos": int(qa[m].sum()), "theirs_pos": int(qb[m].sum()), "both": int((qa & qb)[m].sum()),
                       "jaccard": float((qa & qb)[m].sum() / max((qa | qb)[m].sum(), 1)), "final_pos": int(new[m].sum())}
    stats["state_rmse_speed"] = float(np.sqrt(((A.loc[st, "speed_kmh"] - B.loc[st, "speed_kmh"]) ** 2).mean()))
    stats["state_rmse_flow"] = float(np.sqrt(((A.loc[st, "flow_vph"] - B.loc[st, "flow_vph"]) ** 2).mean()))
    stats["args"] = vars(a)
    C.to_csv(out / "submission.csv", index=False)
    (out / "blend.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
