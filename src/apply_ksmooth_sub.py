import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task2_model as tm
from apply_physop_sub import panel_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=float, default=0.03)
    a = ap.parse_args()
    rel = Path(a.release_root)
    sub = pd.read_csv(a.sub, dtype={"task": str}).sort_values("submission_id").reset_index(drop=True)
    key = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task", "split", "panel", "timestamp", "link_id"], dtype=str)
    key["submission_id"] = key.submission_id.astype(int)
    key = key.sort_values("submission_id").reset_index(drop=True)
    assert (key.submission_id.values == sub.submission_id.values).all() and (key.task.values == sub.task.values).all()
    out = sub.copy()
    st = (sub.task == "state").to_numpy()
    info = {"args": vars(a), "panels": {}}
    for (p, split), idx in key[st].groupby(["panel", "split"]).groups.items():
        order, _ = panel_params(rel, p)
        dates, A = tm.load_days(rel, p, split, "mainline_states_masked", list(order))
        di = {d: i for i, d in enumerate(dates)}
        ts = pd.to_datetime(key.loc[idx, "timestamp"], utc=True)
        d = ts.dt.strftime("%Y-%m-%d").map(di).to_numpy()
        t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        c = order.get_indexer(key.loc[idx, "link_id"])
        assert (c >= 0).all() and not np.isnan(d.astype(float)).any(), p
        d = d.astype(int)
        V, Q = A[:, 0].astype(np.float64), A[:, 1].astype(np.float64)
        obs = np.isfinite(V) & np.isfinite(Q) & (V > 1.0)
        K = np.where(obs, Q / np.maximum(V, 5.0), np.nan)
        v = sub.loc[idx, "speed_kmh"].to_numpy(float)
        q = sub.loc[idx, "flow_vph"].to_numpy(float)
        inner = (t > 0) & (t < tm.T - 1)
        tp, tn = np.clip(t - 1, 0, tm.T - 1), np.clip(t + 1, 0, tm.T - 1)
        both = inner & obs[d, tp, c] & obs[d, tn, c]
        k = q / np.maximum(v, 5.0)
        kt = np.where(both, (1 - 2 * a.a) * k + a.a * (np.nan_to_num(K[d, tp, c]) + np.nan_to_num(K[d, tn, c])), k)
        q2 = np.where(both, np.clip(kt * v, 0.0, None), q)
        out.loc[idx, "flow_vph"] = q2
        info["panels"][f"{p}|{split}"] = {"rows": len(idx), "changed": int(both.sum()), "rmse_flow_change": float(np.sqrt(np.mean((q2 - q) ** 2)))}
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "submission.csv", index=False)
    info["changed_total"] = sum(x["changed"] for x in info["panels"].values())
    info["non_state_identical"] = bool((out.loc[~st].fillna(-1).values == sub.loc[~st].fillna(-1).values).all())
    (od / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "panels"}, indent=2))
    print(pd.DataFrame(info["panels"]).T.to_string())


if __name__ == "__main__":
    main()
