import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm
from task2_predict_sel import eiou_select


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--base-sub", required=True)
    ap.add_argument("--ours-probs", required=True)
    ap.add_argument("--pub-probs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--pub-mode", choices=["lgb", "full"], default="lgb")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pub = pd.read_parquet(a.pub_probs)
    pub["p_pub"] = pub.p_lgb if a.pub_mode == "lgb" else 0.5 * pub.p_lgb + 0.5 * pub.p_cat
    pub = pub[["window_id", "link_id", "p_pub"]]
    wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(rel / "task2" / "*" / "*" / "window_index.csv"))]).set_index("window_id")
    order = {p: tm.network(rel, p)[0] for p in tm.PANELS}
    rows = []
    for split in ("validation", "private"):
        z = np.load(Path(a.ours_probs) / f"probs_{split}.npz")
        for k in z.files:
            wid, name = k.split("|")
            if name == "v4" and z[k].ndim == 1:
                rows.append(pd.DataFrame({"window_id": wid, "link_id": order[wi.panel[wid]], "p_ours": z[k]}))
    m = pd.concat(rows, ignore_index=True).merge(pub, on=["window_id", "link_id"], how="left")
    assert m.p_pub.notna().all()
    m["p"] = (1 - a.w) * m.p_ours + a.w * m.p_pub
    m["pred"] = 0
    for wid, idx in m.groupby("window_id").indices.items():
        m.loc[m.index[idx], "pred"] = eiou_select(m.p.to_numpy()[idx]).astype(int)
    m["timestamp"] = (pd.to_datetime(m.window_id.map(wi.forecast_origin), utc=True) + pd.Timedelta(minutes=5 * tm.H)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    k = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str)
    k = k[(k.task == "queue") & k.window_id.isin(m.window_id.unique())].copy()
    k = k.merge(m[["window_id", "timestamp", "link_id", "pred"]], on=["window_id", "timestamp", "link_id"], how="left")
    k["pred"] = k.pred.fillna(0).astype(int)
    k["submission_id"] = k.submission_id.astype(int)
    s = pd.read_csv(a.base_sub).set_index("submission_id")
    before = s.loc[k.submission_id, "queue_pred"].to_numpy()
    s.loc[k.submission_id.to_numpy(), "queue_pred"] = k.pred.to_numpy()
    s.reset_index().to_csv(out / "submission.csv", index=False)
    info = {"onset_cells": int(len(k)), "pos_before": int(before.sum()), "pos_after": int(k.pred.sum()), "changed": int((before != k.pred.to_numpy()).sum()),
            "windows_changed": int(k[before != k.pred.to_numpy()].window_id.nunique()), "args": vars(a)}
    (out / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
