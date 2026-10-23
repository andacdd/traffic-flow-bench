import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm
import ramp_feats as rf

SPACING = 72
GAP_COLS = ["is_sel", "gap_any", "gap_same", "at_gate"]
STEP = pd.Timedelta(minutes=5)


def gap_features(w, start):
    w = w.sort_values("origin").copy()
    prev_any, prev_same = [], []
    last = {}
    p_any = start
    for r in w.itertuples():
        prev_any.append((r.origin - p_any) / STEP)
        prev_same.append((r.origin - last[r.condition]) / STEP if r.condition in last else np.nan)
        p_any = r.origin
        last[r.condition] = r.origin
    w["gap_any"] = np.minimum(prev_any, 3 * tm.T)
    w["gap_same"] = np.minimum(prev_same, 3 * tm.T)
    w["at_gate"] = (np.abs(w.gap_any - SPACING) < 0.5).astype(float)
    return w


def dev_gaps(D):
    out = []
    for (panel, month, sid), g in D.groupby(["panel", "month", "sample_id"]):
        if sid == 0:
            start = pd.Timestamp(month + "-01", tz="UTC")
        else:
            prev = D[(D.panel == panel) & (D.month == month) & (D.sample_id == sid - 1)].origin.max()
            start = prev.normalize() + pd.Timedelta(days=1)
        out.append(gap_features(g, start))
    return pd.concat(out)


def og_rows(X, freq, t0, Y=None, win=None, day=None, gaps=None):
    rows = []
    for h in range(1, tm.H + 1):
        Xh = X.assign(h=h, freq_h=freq[min(t0 + h, tm.T - 1)])
        if gaps is not None:
            Xh = Xh.assign(**gaps)
        if Y is not None:
            Xh = Xh.assign(y=Y[h - 1], win=win, day=day)
        rows.append(Xh)
    return rows


def build(rel, D):
    rng = np.random.default_rng(0)
    on, og, sel_og, sel_on, meta = [], [], [], [], {}
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        full, by_m = td.month_freqs(P, days, True)
        o, g = td.gen(P, pid, "hybrid", rng, 2, days, lambda dt: by_m[dt[:7]])
        rdates, R, rm = rf.load(rel, panel, "train", P["order"])
        sc = rf.scale(R)
        rdi = {x: i for i, x in enumerate(rdates)}

        def ramp(d, t0):
            k = rdi.get(P["dates"][d])
            if k is None:
                return {c: np.full(len(P["order"]), np.nan, np.float32) for c in rf.RAMP_COLS}
            return rf.features(R[k], t0, rm, sc)

        for x in o + g:
            for c, v in ramp(int(x.day.iloc[0]), int(x.tod.iloc[0])).items():
                x[c] = v
        mon = {d: P["dates"][d][:7] for d in days}
        for x in o:
            x["month"] = mon[int(x.day.iloc[0])]
        for x in g:
            x["month"] = mon[int(x.day.iloc[0])]
        on += o
        og += g
        di = {d: i for i, d in enumerate(P["dates"])}
        for r in D[D.panel == panel].itertuples():
            d = di[str(r.date)[:10]]
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            if t0 + tm.H >= tm.T:
                continue
            freq = by_m[P["dates"][d][:7]]
            X = tm.features(td.hybrid(P["F"][d], P["M"][d], t0), t0, P["vf"], P["cap"], P["lanes"], freq, pd.Timestamp(P["dates"][d]).weekday(), pid)
            X = X.assign(**ramp(d, t0))
            gaps = {"is_sel": 1.0, "gap_any": r.gap_any, "gap_same": r.gap_same, "at_gate": r.at_gate}
            win = f"S|{r.window_id}"
            if r.condition == "queue_ongoing":
                for x in og_rows(X, freq, t0, P["Q"][d, t0 + 1:t0 + tm.H + 1], win, d, gaps):
                    sel_og.append(x.assign(month=r.month))
            else:
                sel_on.append(X.assign(y=P["Q"][d, t0 + tm.H], win=win, day=d, month=r.month, **gaps))
        meta[panel] = {"order": P["order"], "vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full, "ramp_meta": rm, "ramp_scale": sc}
        tm.FREQ[pid] = full
        print(panel, f"{time.time()-s:.0f}s", len(o), len(g) // tm.H, flush=True)
        del P
    cat = lambda x: pd.concat(x, ignore_index=True)
    return cat(on), cat(og), cat(sel_og), cat(sel_on), meta


ID_COLS = ["pos", "panel", "tod", "dow", "lanes", "cap"]


def fit(df, use_gaps, rounds, weight=1.0, ramp=False, drop_id=False):
    d = df[df.y.notna()]
    gap_drop = [] if use_gaps is True else (GAP_COLS if not use_gaps else [c for c in GAP_COLS if c not in use_gaps])
    drop = ["y", "win", "day", "month"] + td.FREQ_COLS + gap_drop + ([] if ramp else rf.RAMP_COLS) + (ID_COLS if drop_id else [])
    X = d.drop(columns=[c for c in drop if c in d.columns])
    w = np.where(d.get("is_sel", pd.Series(0.0, index=d.index)).fillna(0).to_numpy() > 0, weight, 1.0)
    return lgb.train(tm.PARAMS, lgb.Dataset(X, d.y, weight=w), rounds), list(X.columns)


def eiou_select(p, kmin=1):
    o = np.argsort(-p)
    ps = p[o]
    cs = np.cumsum(ps)
    tot = cs[-1]
    k = np.arange(1, len(p) + 1)
    e = cs / (k + tot - cs)
    best = int(np.argmax(e[kmin - 1:])) + kmin
    sel = np.zeros(len(p), bool)
    sel[o[:best]] = True
    return sel


def decide(P, cond, rule, thr):
    if cond == "queue_onset":
        if rule == "eiou":
            return eiou_select(P)
        s = P >= thr
        if not s.any():
            s[np.argmax(P)] = True
        return s
    if rule == "eiou":
        return eiou_select(P.ravel()).reshape(P.shape)
    return P >= thr


def iou(p, y):
    m = ~np.isnan(y)
    p, y = p[m], y[m].astype(bool)
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def cv(on, og, sel_og, sel_on, rounds, out, tag="", onset_only=False):
    months = sorted(og.month.unique())
    folds = [months[i::3] for i in range(3)]
    res = []
    for k, fm in enumerate(folds):
        s = time.time()
        tr_on, tr_og = on[~on.month.isin(fm)], og[~og.month.isin(fm)]
        tr_sel = sel_og[~sel_og.month.isin(fm)]
        te_og, te_on = sel_og[sel_og.month.isin(fm)], sel_on[sel_on.month.isin(fm)]
        tr_sel_on = sel_on[~sel_on.month.isin(fm)]
        on_models = {"v4": fit(tr_on, False, rounds)}
        both_on = pd.concat([tr_on, tr_sel_on], ignore_index=True)
        on_models["v4_noid"] = fit(tr_on, False, rounds, drop_id=True)
        on_models["v4_noid_ramp"] = fit(tr_on, False, rounds, ramp=True, drop_id=True)
        both = pd.concat([tr_og, tr_sel], ignore_index=True)
        models = {} if onset_only else {"v4": fit(tr_og, False, rounds), "v4_noid": fit(tr_og, False, rounds, drop_id=True)}
        for name, (m, c) in models.items():
            te = te_og.copy()
            for col in GAP_COLS:
                if col not in te:
                    te[col] = np.nan
            te["p"] = m.predict(te[c])
            for win, g in te.groupby("win"):
                g = g.sort_values(["h"])
                Pm = g.p.to_numpy().reshape(tm.H, -1)
                Y = g.y.to_numpy().reshape(tm.H, -1)
                for rule, thr in [("thr", 0.35), ("thr", 0.45), ("eiou", None)]:
                    res.append({"fold": k, "cond": "ongoing", "model": name, "rule": f"{rule}{thr or ''}", "iou": iou(decide(Pm, "queue_ongoing", rule, thr), Y), "win": win})
        for name, (m_on, c_on) in on_models.items():
            tt = te_on.assign(p=m_on.predict(te_on[c_on]))
            for win, g in tt.groupby("win"):
                for rule, thr in [("thr", 0.3), ("thr", 0.4), ("thr", 0.55), ("eiou", None)]:
                    res.append({"fold": k, "cond": "onset", "model": name, "rule": f"{rule}{thr or ''}", "iou": iou(decide(g.p.to_numpy(), "queue_onset", rule, thr), g.y.to_numpy()), "win": win})
        print(f"fold {k} {time.time()-s:.0f}s", flush=True)
    r = pd.DataFrame(res)
    r.to_csv(out / f"sel_cv{tag}.csv", index=False)
    print(r.groupby(["cond", "model", "rule"]).iou.agg(["mean", "count"]).round(4).to_string(), flush=True)


def official(rel, split, meta):
    rows = []
    for pid, panel in enumerate(tm.PANELS):
        w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
        w["origin"] = pd.to_datetime(w.forecast_origin, utc=True)
        start = pd.Timestamp(str(w.origin.min().date())[:7] + "-01", tz="UTC")
        w = gap_features(w, start)
        m = meta[panel]
        dates, A = tm.load_days(rel, panel, split, "mainline_states_masked", m["order"])
        di = {d: i for i, d in enumerate(dates)}
        rdates, R, _ = rf.load(rel, panel, split, m["order"])
        rdi = {x: i for i, x in enumerate(rdates)}
        for r in w.itertuples():
            t0 = r.origin.hour * 12 + r.origin.minute // 5
            X = tm.features(A[di[r.date]], t0, m["vf"], m["cap"], m["lanes"], m["freq"], r.origin.weekday(), pid)
            X = X.assign(**rf.features(R[rdi[r.date]], t0, m["ramp_meta"], m["ramp_scale"]))
            X = X.assign(link_id=m["order"], window_id=r.window_id, condition=r.condition, t0=t0, T0=r.origin,
                         is_sel=1.0, gap_any=r.gap_any, gap_same=r.gap_same, at_gate=r.at_gate)
            rows.append(X)
    return rows


def predict(rows, m_on, c_on, models, out_dir, split):
    probs = {}
    for X in rows:
        wid, cond, t0 = X.window_id.iloc[0], X.condition.iloc[0], int(X.t0.iloc[0])
        if cond == "queue_onset":
            probs[(wid, "v4")] = m_on.predict(X[c_on])
        else:
            freq = tm.FREQ[int(X.panel.iloc[0])]
            for name, (m, c) in models.items():
                probs[(wid, name)] = np.stack([m.predict(Xh[c]) for Xh in og_rows(X, freq, t0)])
    np.savez_compressed(out_dir / f"probs_{split}.npz", **{f"{k[0]}|{k[1]}": v for k, v in probs.items()})
    return probs


def write(rows, probs, onset_rule, og_model, og_rule, thr_on, thr_og):
    out = []
    for X in rows:
        wid, cond = X.window_id.iloc[0], X.condition.iloc[0]
        Pm = np.zeros((tm.H, len(X)), bool)
        if cond == "queue_onset":
            Pm[-1] = decide(probs[(wid, "v4")], cond, onset_rule, thr_on)
        else:
            Pm = decide(probs[(wid, og_model)], cond, og_rule, thr_og)
        for h in range(1, tm.H + 1):
            ts = (X.T0.iloc[0] + STEP * h).strftime("%Y-%m-%dT%H:%M:%SZ")
            out.append(pd.DataFrame({"window_id": wid, "timestamp": ts, "link_id": X.link_id, "queue_pred": Pm[h - 1].astype(int)}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--onset-only", action="store_true")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(a.windows)
    D["origin"] = pd.to_datetime(D.origin, utc=True)
    D = dev_gaps(D)
    on, og, sel_og, sel_on, meta = build(rel, D)
    print("gap_any at_gate share (dev):", D.groupby("condition").at_gate.mean().round(3).to_dict(), flush=True)
    if a.cv:
        cv(on, og, sel_og, sel_on, a.rounds, out, a.tag, a.onset_only)
        return
    m_on, c_on = fit(on, False, a.rounds)
    both = pd.concat([og, sel_og], ignore_index=True)
    models = {"v4": fit(og, False, a.rounds), "sel": fit(both, ["is_sel"], a.rounds, 3.0), "sel_ramp": fit(both, ["is_sel"], a.rounds, 3.0, ramp=True)}
    variants = {"v4thr": ("thr", "v4", "thr", 0.35), "eiou": ("eiou", "v4", "eiou", None), "sel": ("thr", "sel", "thr", 0.45),
                "sel_eiou": ("eiou", "sel", "eiou", None), "sel_ramp": ("thr", "sel_ramp", "thr", 0.45)}
    parts = {k: [] for k in variants}
    for split in ["validation", "private"]:
        rows = official(rel, split, meta)
        probs = predict(rows, m_on, c_on, models, out, split)
        for k, (onr, ogm, ogr, thr) in variants.items():
            parts[k].append(write(rows, probs, onr, ogm, ogr, 0.3, thr))
    stats = {}
    for k, p in parts.items():
        d = out / k
        d.mkdir(exist_ok=True)
        q = pd.concat(p, ignore_index=True)
        q.to_csv(d / "queue_vp.csv", index=False)
        stats[k] = float(q.queue_pred.mean())
    (out / "config.json").write_text(json.dumps({"variants": variants, "pred_rate": stats, "rounds": a.rounds}, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
