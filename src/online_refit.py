import argparse
import gc
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm
from task2_predict_sel import eiou_select, iou

WEEKS = [1, 8, 15, 22]
MEM_L = [1, 3, 7]
MEM_TOD = 12
MEM_COLS = [f"mem_{k}{L}" for k in ("any", "tod") for L in MEM_L]
EXT_COLS = ["mem_last", "mem_cnt14", "mem_h30_3", "mem_h30_7", "mem_nb1", "mem_nb3"]
NB_SPAN = 2
RARE = 0.02
MIN_OBS = 0.5


def calendar(dates, Qs, rare):
    ords = pd.to_datetime(list(dates)).values.astype("datetime64[D]").astype(np.int64)
    base = ords.min()
    R = np.zeros((ords.max() - base + 1,) + Qs.shape[1:], bool)
    R[ords - base] = (np.nan_to_num(Qs) > 0) & rare[None]
    return R, {d: int(o - base) for d, o in zip(dates, ords)}


def memory(R, g, t0):
    cols = {}
    for L in MEM_L:
        w = R[max(0, g - L):g]
        if len(w) == 0:
            z = np.zeros(R.shape[2], np.float32)
            cols[f"mem_any{L}"], cols[f"mem_tod{L}"] = z, z
            continue
        cols[f"mem_any{L}"] = w.any(1).mean(0).astype(np.float32)
        cols[f"mem_tod{L}"] = w[:, max(0, t0 - MEM_TOD):t0 + MEM_TOD + 1].any(1).mean(0).astype(np.float32)
    w = R[max(0, g - 14):g].any(1)
    n = R.shape[2]
    if len(w):
        hit = w[::-1].argmax(0)
        cols["mem_last"] = np.where(w.any(0), hit + 1, 15).astype(np.float32)
        cols["mem_cnt14"] = w.mean(0).astype(np.float32)
    else:
        cols["mem_last"], cols["mem_cnt14"] = np.full(n, 15, np.float32), np.zeros(n, np.float32)
    for L in (3, 7):
        h = R[max(0, g - L):g, t0 + 3:t0 + 10]
        cols[f"mem_h30_{L}"] = h.any(1).mean(0).astype(np.float32) if len(h) else np.zeros(n, np.float32)
    for L in (1, 3):
        m = cols[f"mem_tod{L}"]
        cols[f"mem_nb{L}"] = maximum_filter1d(m, 2 * NB_SPAN + 1, mode="constant").astype(np.float32)
    return cols


def week(date):
    return max(i for i, s in enumerate(WEEKS) if int(date[8:10]) >= s)


def eval_rows(A, Q, dates, gidx, pid, m, R, rng, og_per_day, test):
    on, og = [], []
    for i, date in enumerate(dates):
        g, dow = gidx[date], pd.Timestamp(date).weekday()
        Qd = Q[i]
        obs = ~np.isnan(Qd)
        for t0 in tm.onset_origins(Qd):
            if t0 + tm.H >= tm.T or obs[t0].mean() < 0.3 or obs[t0 + tm.H].mean() < MIN_OBS:
                continue
            X = tm.features(A[i], t0, m["vf"], m["cap"], m["lanes"], m["freq"], dow, pid).assign(**memory(R, g, t0))
            on.append(X.assign(y=Qd[t0 + tm.H], win=f"M|{pid}|{date}|{t0}", day=g, src=1, wk=week(date)))
        cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(Qd[t0]) > 0 and (not test or obs[t0 + 1:t0 + tm.H + 1].mean() >= MIN_OBS)]
        for t0 in (rng.choice(cand, min(og_per_day, len(cand)), replace=False) if cand else []):
            X = tm.features(A[i], t0, m["vf"], m["cap"], m["lanes"], m["freq"], dow, pid).assign(**memory(R, g, int(t0)))
            for h in range(1, tm.H + 1):
                og.append(X.assign(h=h, y=Qd[t0 + h], win=f"M|{pid}|{date}|{t0}", day=g, src=1, wk=week(date)))
    return on, og


def build(rel, og_test_per_day):
    rng = np.random.default_rng(0)
    tr_on, tr_og, ev_on, ev_og = [], [], [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        full, by_m = td.month_freqs(P, days, True)
        rare = full < RARE
        vd, VA = tm.load_days(rel, panel, "validation", "mainline_states_masked", P["order"])
        VQ = np.stack([tm.queue_matrix(VA[i, 0], P["vf"]) for i in range(len(vd))])
        TQ = np.stack([tm.queue_matrix(P["M"][d, 0], P["vf"]) for d in days])
        R, gidx = calendar(list(P["dates"]) + list(vd), np.concatenate([TQ, VQ]), rare)
        m = {"vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full}
        o, g = td.gen(P, pid, "hybrid", rng, 2, days, lambda dt: by_m[dt[:7]])
        for x in o + g:
            d = int(x.day.iloc[0])
            t0 = int(x.tod.iloc[0])
            for c, v in memory(R, gidx[P["dates"][d]], t0).items():
                x[c] = v
            x["src"] = 0
            x["wk"] = -1
        tr_on += o
        tr_og += g
        o, g = eval_rows(VA, VQ, vd, gidx, pid, m, R, rng, 2, False)
        to, tg = eval_rows(VA, VQ, vd, gidx, pid, m, R, np.random.default_rng(1 + pid), og_test_per_day, True)
        tr_on += o
        tr_og += g
        ev_on += [x.assign(test=1) for x in to]
        ev_og += [x.assign(test=1) for x in tg]
        print(panel, f"{time.time()-s:.0f}s", "march on/og", len(o), len(g) // tm.H, "test", len(to), len(tg) // tm.H, flush=True)
        del P, VA, VQ, TQ, R
        gc.collect()
    cat = lambda x: pd.concat(x, ignore_index=True)
    return cat(tr_on), cat(tr_og), cat(ev_on), cat(ev_og)


def fit(df, rounds, w_eval, mem):
    d = df[df.y.notna()]
    drop = ["y", "win", "day", "month", "src", "wk", "test", "split"] + td.FREQ_COLS + ([] if mem == "ext" else EXT_COLS) + ([] if mem else MEM_COLS)
    X = d.drop(columns=[c for c in drop if c in d.columns])
    w = np.where(d.src.to_numpy() == 1, w_eval, 1.0)
    return lgb.train(dict(tm.PARAMS, num_threads=8), lgb.Dataset(X, d.y, weight=w), rounds), list(X.columns)


def score(name, k, te_on, te_og, m_on, m_og):
    res = []
    if len(te_on):
        p = m_on[0].predict(te_on[m_on[1]])
        for win, idx in te_on.groupby("win").indices.items():
            res.append({"variant": name, "wk": k, "cond": "onset", "win": win, "iou": iou(eiou_select(p[idx]), te_on.y.to_numpy()[idx])})
    if len(te_og):
        p = m_og[0].predict(te_og[m_og[1]])
        for win, idx in te_og.groupby("win").indices.items():
            idx = idx[np.argsort(te_og.h.to_numpy()[idx], kind="stable")]
            Pm = p[idx].reshape(tm.H, -1)
            res.append({"variant": name, "wk": k, "cond": "ongoing", "win": win, "iou": iou(eiou_select(Pm.ravel()).reshape(Pm.shape), te_og.y.to_numpy()[idx].reshape(tm.H, -1))})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--weights", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--og-test-per-day", type=int, default=4)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_on, tr_og, ev_on, ev_og = build(rel, a.og_test_per_day)
    base = tr_on[tr_on.src == 0], tr_og[tr_og.src == 0]
    s = time.time()
    b_on, b_og = fit(base[0], a.rounds, 1.0, False), fit(base[1], a.rounds, 1.0, False)
    print(f"base fit {time.time()-s:.0f}s", flush=True)
    res = []
    for k in range(1, len(WEEKS)):
        te_on, te_og = ev_on[ev_on.wk == k], ev_og[ev_og.wk == k]
        print(f"week {k}: test onset {te_on.win.nunique()} ongoing {te_og.win.nunique()}", flush=True)
        res += score("base", k, te_on, te_og, b_on, b_og)
        tr_on_k = tr_on[(tr_on.src == 0) | (tr_on.wk < k)]
        tr_og_k = tr_og[(tr_og.src == 0) | (tr_og.wk < k)]
        for w in a.weights:
            s = time.time()
            if "A" in a.variants:
                a_og = fit(tr_og_k, a.rounds, w, False)
                res += score(f"A_w{w:g}", k, te_on, te_og, b_on, a_og)
            for v, mem in (("B", True), ("C", "ext")):
                if v in a.variants:
                    m_on, m_og = fit(tr_on_k, a.rounds, w, mem), fit(tr_og_k, a.rounds, w, mem)
                    res += score(f"{v}_w{w:g}", k, te_on, te_og, m_on, m_og)
            print(f"  week {k} w{w:g} {time.time()-s:.0f}s", flush=True)
            if k == len(WEEKS) - 1 and w == a.weights[-1]:
                for nm, mm in (("og", m_og), ("on", m_on)):
                    pd.Series(mm[0].feature_importance("gain"), index=mm[1]).sort_values(ascending=False).to_csv(out / f"imp_{a.variants[-1]}_{nm}.csv")
        pd.DataFrame(res).to_csv(out / "refit_backtest.csv", index=False)
    r = pd.DataFrame(res)
    t = r.groupby(["variant", "cond"]).iou.mean().unstack()
    t["queue"] = t.mean(1)
    t["d_queue"] = t.queue - t.loc["base", "queue"]
    t["wk_min_d"] = (r.groupby(["variant", "wk"]).iou.mean().unstack().sub(r[r.variant == "base"].groupby("wk").iou.mean(), axis=1)).min(1)
    print(t.sort_values("queue", ascending=False).round(4).to_string(), flush=True)
    (out / "summary.json").write_text(json.dumps(t.round(5).reset_index().to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
