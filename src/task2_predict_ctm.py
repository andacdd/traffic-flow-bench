import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import queue_ctm as qc
import task2_diag as td
import task2_model as tm

CFG = dict(qc.CONFIGS["zsh6"], mask=None)
CTM_COLS = ["ctm_r", "ctm_q"]


def ctm_batch(days, t0s, p, vf):
    if not len(days):
        return np.zeros((0, tm.H, len(vf)))
    Sv = np.stack([d[0] for d in days]).astype(np.float64)
    Sq = np.stack([d[1] for d in days]).astype(np.float64)
    prof = np.zeros((tm.T, len(vf)))
    V = qc.forecast(Sv, Sq, np.array(t0s), p, prof, CFG)
    return V / vf[None, None, :]


def ongoing_rows(X, R, vf, t0, freq, y=None, win=None, day=None):
    out = []
    for h in range(1, tm.H + 1):
        r = R[h - 1]
        Xh = X.assign(h=h, freq_h=freq[min(t0 + h, tm.T - 1)], ctm_r=r.astype(np.float32), ctm_q=(r <= 0.6).astype(np.float32))
        if y is not None:
            Xh = Xh.assign(y=y[h - 1], win=win, day=day)
        out.append(Xh)
    return out


def gen(P, pid, rng, ongoing_per_day, days, freq_of, p):
    on, og = [], []
    for d in days:
        date = P["dates"][d]
        dow = pd.Timestamp(date).weekday()
        freq = freq_of(date)
        for t0 in tm.onset_origins(P["Q"][d]):
            if t0 + tm.H >= tm.T:
                continue
            day = td.hybrid(P["F"][d], P["M"][d], t0)
            X = tm.features(day, t0, P["vf"], P["cap"], P["lanes"], freq, dow, pid, 0.0, rng)
            X["y"] = P["Q"][d, t0 + tm.H]
            X["win"] = f"{pid}|{date}|{t0}"
            X["day"] = d
            on.append(X)
        cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(P["Q"][d, t0]) > 0]
        sel = list(rng.choice(cand, min(ongoing_per_day, len(cand)), replace=False)) if cand else []
        dd = [td.hybrid(P["F"][d], P["M"][d], t0) for t0 in sel]
        R = ctm_batch(dd, sel, p, P["vf"])
        for k, t0 in enumerate(sel):
            X = tm.features(dd[k], t0, P["vf"], P["cap"], P["lanes"], freq, dow, pid, 0.0, rng)
            og += ongoing_rows(X, R[k], P["vf"], t0, freq, P["Q"][d, t0 + 1:t0 + tm.H + 1], f"{pid}|{date}|{t0}", d)
    return on, og


def official(rel, split, meta, params):
    rows = []
    for pid, panel in enumerate(tm.PANELS):
        w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
        m = meta[panel]
        dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", m["order"])
        di = {d: i for i, d in enumerate(dates)}
        T0s = [pd.Timestamp(r.forecast_origin) for r in w.itertuples()]
        t0s = [x.hour * 12 + x.minute // 5 for x in T0s]
        days = [A[di[r.date]] for r in w.itertuples()]
        R = ctm_batch(days, t0s, params[panel], m["vf"])
        for k, r in enumerate(w.itertuples()):
            X = tm.features(days[k], t0s[k], m["vf"], m["cap"], m["lanes"], m["freq"], T0s[k].weekday(), pid)
            X["link_id"] = m["order"]
            X["window_id"] = r.window_id
            X["condition"] = r.condition
            X["t0"] = t0s[k]
            X["T0"] = T0s[k]
            rows.append((X, R[k]))
    return rows


def predict(rows, m_og, c_og, thr_og):
    out = []
    for X, R in rows:
        if X.condition.iloc[0] != "queue_ongoing":
            continue
        t0 = int(X.t0.iloc[0])
        freq = tm.FREQ[int(X.panel.iloc[0])]
        T0 = X.T0.iloc[0]
        for h, Xh in enumerate(ongoing_rows(X, R, None, t0, freq), 1):
            ts = (T0 + pd.Timedelta(minutes=5 * h)).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append(pd.DataFrame({"window_id": X.window_id.iloc[0], "timestamp": ts, "link_id": X.link_id,
                                     "queue_pred": (m_og.predict(Xh[c_og]) >= thr_og).astype(int)}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-queue", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--thr-ongoing", type=float, default=0.35)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    og, meta, params = [], {}, {}
    for pid, panel in enumerate(tm.PANELS):
        P = td.load_panel(rel, pid, panel)
        params[panel] = qc.params(rel, panel, P["order"])
        days = list(range(len(P["dates"])))
        full, by_m = td.month_freqs(P, days, True)
        _, g = gen(P, pid, rng, 2, days, lambda dt: by_m[dt[:7]], params[panel])
        og += g
        meta[panel] = {"order": P["order"], "vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full}
        tm.FREQ[pid] = full
        print(panel, len(g), flush=True)
        del P
    og = pd.concat(og, ignore_index=True)
    m_og, c_og = td.fit(og, td.FREQ_COLS, a.rounds)
    imp = pd.Series(m_og.feature_importance("gain"), index=c_og)
    print("ctm gain share", (imp[CTM_COLS].sum() / imp.sum()).round(4), imp.sort_values(ascending=False).head(8).round(0).to_dict(), flush=True)
    base = pd.read_csv(a.base_queue, dtype=str, keep_default_na=False)
    stats = {}
    for split in ["validation", "private"]:
        new = predict(official(rel, split, meta, params), m_og, c_og, a.thr_ongoing)
        new.to_csv(out / f"queue_ongoing_{split}.csv", index=False)
        stats[split] = {"rows": len(new), "mean": float(new.queue_pred.mean())}
        print(split, stats[split], flush=True)
    new = pd.concat([pd.read_csv(out / f"queue_ongoing_{s}.csv", dtype=str, keep_default_na=False) for s in ["validation", "private"]], ignore_index=True)
    key = ["window_id", "timestamp", "link_id"]
    m = base.merge(new.rename(columns={"queue_pred": "new_pred"}), on=key, how="left")
    hit = m.new_pred.notna() & (m.new_pred != "")
    stats["replaced_rows"] = int(hit.sum())
    stats["changed_rows"] = int((hit & (m.new_pred != m.queue_pred)).sum())
    stats["base_ongoing_mean"] = float(pd.to_numeric(m.loc[hit, "queue_pred"]).mean())
    m.loc[hit, "queue_pred"] = m.loc[hit, "new_pred"]
    m[base.columns].to_csv(out / "queue_vp.csv", index=False)
    stats["ctm_gain_share"] = float(imp[CTM_COLS].sum() / imp.sum())
    (out / "config.json").write_text(json.dumps({"variant": "v4 onset + v4 ongoing + CTM zsh6 features", "thr_ongoing": a.thr_ongoing, "rounds": a.rounds, "stats": stats}, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
