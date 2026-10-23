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
    ap.add_argument("--pub-queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--rule", default="eiou")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pub = pd.read_parquet(a.pub_probs)
    pq = pd.read_csv(a.pub_queue, dtype={"link_id": str})
    wi = pd.concat([pd.read_csv(f) for f in glob.glob(str(rel / "task2" / "*" / "*" / "window_index.csv"))])
    origin = dict(zip(wi.window_id, pd.to_datetime(wi.forecast_origin, utc=True)))
    pq["h"] = ((pd.to_datetime(pq.timestamp, utc=True) - pq.window_id.map(origin)).dt.total_seconds() // 300).astype(int)
    chk = pub.merge(pq, on=["window_id", "h", "link_id"])
    agree = float(((chk.p_pub >= 0.4).astype(int) == chk.queue_pred).mean())
    order = {p: tm.network(rel, p)[0] for p in tm.PANELS}
    rows = []
    for split in ("validation", "private"):
        z = np.load(Path(a.ours_probs) / f"probs_{split}.npz")
        for k in z.files:
            wid, name = k.split("|")
            if name != "v4":
                continue
            P = z[k]
            if P.ndim != 2:
                continue
            panel = wi.set_index("window_id").panel[wid]
            rows.append(pd.DataFrame({"window_id": wid, "h": np.repeat(np.arange(1, tm.H + 1), P.shape[1]),
                                      "link_id": np.tile(order[panel], tm.H), "p_ours": P.ravel()}))
    ours = pd.concat(rows, ignore_index=True)
    m = ours.merge(pub, on=["window_id", "h", "link_id"], how="left")
    assert m.p_pub.notna().all(), int(m.p_pub.isna().sum())
    m["p"] = (1 - a.w) * m.p_ours + a.w * m.p_pub
    m["pred"] = 0
    for wid, idx in m.groupby("window_id").indices.items():
        p = m.p.to_numpy()[idx]
        m.loc[m.index[idx], "pred"] = (eiou_select(p) if a.rule == "eiou" else p >= float(a.rule)).astype(int)
    k = pd.read_csv(rel / "submission_key.csv", usecols=["submission_id", "task", "window_id", "timestamp", "link_id"], dtype=str)
    k = k[k.task == "queue"].copy()
    k["h"] = ((pd.to_datetime(k.timestamp, utc=True) - k.window_id.map(origin)).dt.total_seconds() // 300).astype(int)
    k = k.merge(m[["window_id", "h", "link_id", "pred"]], on=["window_id", "h", "link_id"], how="inner")
    k["submission_id"] = k.submission_id.astype(int)
    sub = pd.read_csv(a.base_sub)
    s = sub.set_index("submission_id")
    before = s.loc[k.submission_id, "queue_pred"].to_numpy()
    s.loc[k.submission_id.to_numpy(), "queue_pred"] = k.pred.to_numpy()
    s.reset_index().to_csv(out / "submission.csv", index=False)
    info = {"agree_pub_labels_at_0.4": agree, "ongoing_cells": int(len(k)), "pos_before": int(before.sum()), "pos_after": int(k.pred.sum()),
            "changed": int((before != k.pred.to_numpy()).sum()), "args": vars(a)}
    (out / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
