import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_interp import link_order
from phys_operator import stage1


def panel_params(rel, p):
    order, _ = link_order(rel, p)
    fd = pd.read_csv(rel / "corridors" / p / "network" / "fd_parameters.csv").set_index("link_id")
    g = {c: fd[c].reindex(order).to_numpy(dtype=float) for c in tr.GEO}
    lanes = np.clip(np.nan_to_num(g["lanes"], nan=np.nanmedian(g["lanes"])), 1, None)
    vf = np.nan_to_num(g["free_speed_kmh"], nan=117.0)
    cap = np.nan_to_num(g["capacity_vph"], nan=np.nanmedian(g["capacity_vph"]))
    kc = np.nan_to_num(g["critical_density"], nan=np.nanmedian(g["critical_density"]))
    kj = np.nan_to_num(g["k_jam"], nan=np.nanmedian(g["k_jam"]))
    length = np.nan_to_num(g["length_km"], nan=np.nanmedian(g["length_km"]))
    return pd.Index(order), (lanes, vf, cap, kc, kj, length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--sub", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fq", type=float, default=0.95)
    ap.add_argument("--beta", type=float, default=0.70)
    a = ap.parse_args()
    rel = Path(a.release_root)
    sub = pd.read_csv(a.sub, dtype={"task": str}).sort_values("submission_id").reset_index(drop=True)
    key = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task", "panel", "link_id"], dtype=str)
    key["submission_id"] = key.submission_id.astype(int)
    key = key.sort_values("submission_id").reset_index(drop=True)
    assert (key.submission_id.values == sub.submission_id.values).all() and (key.task.values == sub.task.values).all()
    out = sub.copy()
    st = (sub.task == "state").to_numpy()
    info = {"args": vars(a), "panels": {}}
    for p, idx in key[st].groupby("panel").groups.items():
        order, prm = panel_params(rel, p)
        c = order.get_indexer(key.loc[idx, "link_id"])
        assert (c >= 0).all(), p
        v = sub.loc[idx, "speed_kmh"].to_numpy(float)
        q = sub.loc[idx, "flow_vph"].to_numpy(float)
        _, q2 = stage1(v, q, c, prm, a.beta, a.fq)
        q2 = np.clip(q2, 0.0, None)
        out.loc[idx, "flow_vph"] = q2
        info["panels"][p] = {"rows": len(idx), "changed": int((np.abs(q2 - q) > 1e-9).sum()), "rmse_flow_change": float(np.sqrt(np.mean((q2 - q) ** 2)))}
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
