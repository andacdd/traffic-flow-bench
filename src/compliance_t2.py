import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ramp_feats as rf
import task2_diag as td
import task2_model as tm
from online_refit import RARE, calendar, memory
from task2_predict_sel import gap_features


def window_X(day, rday, t0, T0, pid, m, wrow, mem):
    X = tm.features(day, t0, m["vf"], m["cap"], m["lanes"], m["freq"], T0.weekday(), pid)
    X = X.assign(**rf.features(rday, t0, m["ramp_meta"], m["ramp_scale"]))
    X = X.assign(gap_any=wrow.gap_any, gap_same=wrow.gap_same, at_gate=wrow.at_gate)
    return X.assign(**mem)


def same(a, b):
    a, b = a.to_numpy(np.float64), b.to_numpy(np.float64)
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--out", default="reports/queue/compliance_t2.json")
    ap.add_argument("--splits", nargs="+", default=["validation", "private"])
    a = ap.parse_args()
    rel = Path(a.release_root)
    res = {"windows": 0, "mismatch": [], "cols": None}
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        full, _ = td.month_freqs(P, days, True)
        rare = full < RARE
        _, Rtr, rm = rf.load(rel, panel, "train", P["order"])
        m = {"vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full, "ramp_meta": rm, "ramp_scale": rf.scale(Rtr)}
        TQ = np.stack([tm.queue_matrix(P["M"][d, 0], P["vf"]) for d in days])
        data = {}
        for split in ["validation", "private"]:
            sd, SA = tm.load_days(rel, panel, split, "mainline_states_masked", P["order"])
            rd, RR, _ = rf.load(rel, panel, split, P["order"])
            data[split] = (list(sd), SA, list(rd), RR)
        alld = list(P["dates"]) + data["validation"][0] + data["private"][0]
        allQ = np.concatenate([TQ] + [np.stack([tm.queue_matrix(data[sp][1][i, 0], P["vf"]) for i in range(len(data[sp][0]))]) for sp in ("validation", "private")])
        Rfull, gidx = calendar(alld, allQ, rare)
        for split in a.splits:
            sd, SA, rd, RR = data[split]
            di, rdi = {d: i for i, d in enumerate(sd)}, {d: i for i, d in enumerate(rd)}
            w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
            w["origin"] = pd.to_datetime(w.forecast_origin, utc=True)
            start = pd.Timestamp(str(w.origin.min().date())[:7] + "-01", tz="UTC")
            wf = gap_features(w, start).set_index("window_id")
            for r in w.itertuples():
                T0 = r.origin
                t0 = T0.hour * 12 + T0.minute // 5
                d = str(T0.date())
                Xf = window_X(SA[di[d]], RR[rdi[d]], t0, T0, pid, m, wf.loc[r.window_id], memory(Rfull, gidx[d], t0))
                day = SA[di[d]].copy()
                day[:, t0 + 1:] = np.nan
                rday = RR[rdi[d]].copy()
                rday[t0 + 1:] = np.nan
                wc = gap_features(w[w.origin <= T0].copy(), start).set_index("window_id")
                keep = [i for i, x in enumerate(alld) if x < d] + [alld.index(d)]
                Qc = allQ[keep].copy()
                Qc[-1, t0 + 1:] = np.nan
                Rc, gc_ = calendar([alld[i] for i in keep], Qc, rare)
                Xc = window_X(day, rday, t0, T0, pid, m, wc.loc[r.window_id], memory(Rc, gc_[d], t0))
                res["windows"] += 1
                res["cols"] = list(Xf.columns)
                bad = [c for c in Xf.columns if not same(Xf[c], Xc[c])]
                if bad:
                    res["mismatch"].append({"window_id": r.window_id, "cols": bad})
        print(panel, f"{time.time()-s:.0f}s", f"windows {res['windows']}", f"mismatch {len(res['mismatch'])}", flush=True)
    res["pass"] = not res["mismatch"]
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "cols"})[:2000], flush=True)


if __name__ == "__main__":
    main()
