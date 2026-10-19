import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_cv as qc
import queue_devset as qd
import queue_exp3c as qx
import task1_resid_predict as trp
import task2_diag as td
import task2_model as tm

THR_ON, THR_OG = 0.35, 0.40
SPLITS = ["validation", "private"]


def official_windows(rel, split, meta, panels):
    rows = []
    for panel in panels:
        pid = tm.PANELS.index(panel)
        w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
        m = meta[panel]
        dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", m["order"])
        di = {d: i for i, d in enumerate(dates)}
        for r in w.itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            X = tm.features(A[di[r.date]], t0, m["vf"], m["cap"], m["lanes"], m["freq"], T0.weekday(), pid)
            X["link_id"] = m["order"]
            X["window_id"] = r.window_id
            X["condition"] = r.condition
            X["t0"] = t0
            X["T0"] = T0
            rows.append((X, None))
    return rows


def compare(sub, ref_path, wi):
    ref = pd.read_csv(ref_path)
    m = sub.merge(ref, on=["window_id", "timestamp", "link_id"], how="left", suffixes=("", "_ref"))
    m = m.merge(wi, on="window_id", how="left")
    out = {"ref": str(ref_path), "rows": len(m), "missing_in_ref": int(m.queue_pred_ref.isna().sum()), "agree_all": float((m.queue_pred == m.queue_pred_ref).mean())}
    for c, g in m.groupby("condition"):
        w = g.groupby("window_id").apply(lambda x: (x.queue_pred == x.queue_pred_ref).all(), include_groups=False)
        out[c] = {"cells_new": int(g.queue_pred.sum()), "cells_ref": int(g.queue_pred_ref.sum()), "agree": float((g.queue_pred == g.queue_pred_ref).mean()),
                  "windows_identical": int(w.sum()), "windows": int(len(w))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--panels", default=",".join(tm.PANELS))
    ap.add_argument("--ref", nargs="*", default=[])
    a = ap.parse_args()
    t_start = time.time()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "versions.json").write_text(json.dumps(trp.versions(), indent=2))
    panels = a.panels.split(",")
    rng_v3, rng_e1 = np.random.default_rng(0), np.random.default_rng(1)
    syn = {"on": [], "og_e1": []}
    FQ, FM, meta, counts = [], [], {}, []
    off, wbase = 0, 0
    for panel in panels:
        pid = tm.PANELS.index(panel)
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        D = qd.load_panel(rel, panel)
        assert D["dates"] == P["dates"]
        L = len(P["order"])
        fold_day = {d: 0 for d in P["dates"]}
        di = {d: i for i, d in enumerate(P["dates"])}
        q = np.nan_to_num(np.nanmean(P["Q"], 0)).astype(np.float32)
        m6 = np.zeros_like(q)
        for t in range(tm.T - tm.H):
            m6[t] = q[t + 1:t + tm.H + 1].max(0)
        FQ.append(q.ravel())
        FM.append(m6.ravel())
        meta[panel] = {"order": P["order"], "vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": q}
        tm.FREQ[pid] = q
        zero = np.zeros((tm.T, L), np.float32)
        o, g = td.gen(P, pid, "full", rng_v3, 2, list(range(len(P["dates"]))), lambda dt: zero)
        n_og = len(g) // tm.H
        g1, c1 = qx.selector_chunks(P, D, pid, rng_e1, "og", n_og)
        for nm, ch in [("on", o), ("og_e1", qx.add_fill(g1, D["U"], di))]:
            df = qc.tag(ch, L, off, fold_day, wbase)
            wbase = int(df.win.max()) + 1
            syn[nm].append(df)
        off += tm.T * L
        counts.append({"panel": panel, "days": len(P["dates"]), "on_win": len(o), "og_e1_win": len(g1) // tm.H, "og_e1_cand": c1})
        print(counts[-1], f"{time.time()-s:.0f}s", flush=True)
        del P, D
    FQ, FM = np.concatenate(FQ), np.concatenate(FM)
    for nm in syn:
        syn[nm] = pd.concat(syn[nm], ignore_index=True)
        qc.set_freq(syn[nm], FQ, FM)
    s = time.time()
    m_on, c_on, n_on = qx.fit(syn["on"], "y", a.rounds)
    m_og, c_og, n_og = qx.fit(syn["og_e1"], "yf", a.rounds)
    m_on.save_model(str(out / "model_onset.txt"))
    m_og.save_model(str(out / "model_ongoing.txt"))
    t_train = time.time() - s
    print(f"trained onset {n_on:,} rows, ongoing {n_og:,} rows, {t_train:.0f}s", flush=True)
    res = {"recipe": "v3 onset (full-input, observed y, thr 0.35 T+30 top-1) + E1 selector-mimic ongoing (fill yf, thr 0.40)", "thr_onset": THR_ON,
           "thr_ongoing": THR_OG, "rounds": a.rounds, "train_rows": {"onset": n_on, "ongoing": n_og}, "counts": counts, "train_s": round(t_train, 1)}
    frames = []
    for split in SPLITS:
        s = time.time()
        sub, _ = tm.predict_windows(official_windows(rel, split, meta, panels), m_on, c_on, m_og, c_og, THR_ON, THR_OG)
        sub.to_csv(out / f"queue_{split}.csv", index=False)
        frames.append(sub)
        res[split] = {"rows": len(sub), "windows": int(sub.window_id.nunique()), "mean": float(sub.queue_pred.mean()), "values": sorted(sub.queue_pred.unique().tolist()),
                      "dup_keys": int(sub.duplicated(["window_id", "timestamp", "link_id"]).sum()), "runtime_s": round(time.time() - s, 1)}
        print(split, res[split], flush=True)
    vp = pd.concat(frames, ignore_index=True)
    vp.to_csv(out / "queue_vp.csv", index=False)
    wi = pd.concat([pd.read_csv(rel / "task2" / p / sp / "window_index.csv", usecols=["window_id", "condition"]) for p in panels for sp in SPLITS])
    tpl = pd.concat([pd.read_csv(rel / "task2" / p / sp / "sample_submission_queue.csv", usecols=["window_id", "timestamp", "link_id"]) for p in panels for sp in SPLITS])
    k = ["window_id", "timestamp", "link_id"]
    res["template_rows"] = len(tpl)
    res["template_keys_match"] = bool(len(tpl.merge(vp[k], on=k)) == len(tpl) == len(vp))
    res["comparisons"] = [compare(vp, Path(r), wi) for r in a.ref]
    res["runtime_s"] = round(time.time() - t_start, 1)
    (out / "run.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: res[k] for k in ["template_rows", "template_keys_match", "comparisons", "runtime_s"]}, indent=1, default=float), flush=True)


if __name__ == "__main__":
    main()
