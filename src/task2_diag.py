import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

FREQ_COLS = ["freq_t0", "freq_max_h", "freq_h"]
DAY_COLS = ["day_min_r", "corr_q_frac", "corr_fc_mean", "corr_r_mean"]


def hybrid(full, mask, t0):
    day = mask.copy()
    day[:, t0 - tm.HIST:t0] = full[:, t0 - tm.HIST:t0]
    return day


def load_panel(rel, pid, panel):
    order, vf, cap, lanes = tm.network(rel, panel)
    dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
    dm, M = tm.load_days(rel, panel, "train", "mainline_states_masked", order)
    pos = {d: i for i, d in enumerate(dm)}
    M = M[[pos[d] for d in dates]]
    Q = np.stack([tm.queue_matrix(F[d, 0], vf) for d in range(len(dates))])
    return {"order": order, "vf": vf, "cap": cap, "lanes": lanes, "dates": dates, "F": F, "M": M, "Q": Q}


def month_freqs(P, days, lomo):
    months = np.array([P["dates"][d][:7] for d in days])
    Qd = P["Q"][days]
    full = np.nan_to_num(np.nanmean(Qd, 0)).astype(np.float32)
    if not lomo:
        return full, {m: full for m in set(months)}
    return full, {m: np.nan_to_num(np.nanmean(Qd[months != m], 0)).astype(np.float32) for m in set(months)}


def gen(P, pid, mode, rng, ongoing_per_day, days, freq_of):
    on, og = [], []
    for d in days:
        date = P["dates"][d]
        dow = pd.Timestamp(date).weekday()
        freq = freq_of(date)

        def inp(t0):
            if mode == "full":
                return P["F"][d], 0.35
            return hybrid(P["F"][d], P["M"][d], t0), 0.0

        for t0 in tm.onset_origins(P["Q"][d]):
            if t0 + tm.H >= tm.T:
                continue
            day, drop = inp(t0)
            X = tm.features(day, t0, P["vf"], P["cap"], P["lanes"], freq, dow, pid, drop, rng)
            X["y"] = P["Q"][d, t0 + tm.H]
            X["win"] = f"{pid}|{date}|{t0}"
            X["day"] = d
            on.append(X)
        cand = [t0 for t0 in range(tm.HIST, tm.T - tm.H) if np.nansum(P["Q"][d, t0]) > 0]
        for t0 in (rng.choice(cand, min(ongoing_per_day, len(cand)), replace=False) if cand else []):
            day, drop = inp(t0)
            X = tm.features(day, t0, P["vf"], P["cap"], P["lanes"], freq, dow, pid, drop, rng)
            for h in range(1, tm.H + 1):
                og.append(X.assign(h=h, freq_h=freq[t0 + h], y=P["Q"][d, t0 + h], win=f"{pid}|{date}|{t0}", day=d))
    return on, og


def official(rel, pid, panel, P, freq):
    w = pd.read_csv(rel / "task2" / panel / "train" / "window_index.csv")
    di = {d: i for i, d in enumerate(P["dates"])}
    out = []
    for r in w.itertuples():
        T0 = pd.Timestamp(r.forecast_origin)
        t0 = T0.hour * 12 + T0.minute // 5
        d = di[r.date]
        X = tm.features(P["M"][d], t0, P["vf"], P["cap"], P["lanes"], freq, T0.weekday(), pid)
        out.append((r.condition, X, t0, P["Q"][d, t0 + 1:t0 + tm.H + 1], freq))
    return out


def fit(df, drop, rounds):
    d = df[df.y.notna()]
    X = d.drop(columns=["y", "win", "day"] + [c for c in drop if c in d.columns])
    import lightgbm as lgb
    return lgb.train(tm.PARAMS, lgb.Dataset(X, d.y), rounds), list(X.columns)


def eval_official(wins, m_on, c_on, m_og, c_og, thr_on, thr_og):
    res = []
    for cond, X, t0, Y, freq in wins:
        P = np.zeros((tm.H, len(X)), bool)
        if cond == "queue_onset":
            p = m_on.predict(X[c_on])
            sel = p >= thr_on
            if not sel.any():
                sel[np.argmax(p)] = True
            P[-1] = sel
        else:
            for h in range(1, tm.H + 1):
                P[h - 1] = m_og.predict(X.assign(h=h, freq_h=freq[min(t0 + h, tm.T - 1)])[c_og]) >= thr_og
        res.append((cond, tm.iou(P.ravel(), Y.ravel()), tm.iou(np.repeat(X.q_now.to_numpy()[None] == 1, tm.H, 0).ravel(), Y.ravel())))
    r = pd.DataFrame(res, columns=["cond", "iou", "pers"])
    return r.groupby("cond").iou.mean().round(4).to_dict(), round(r.iou.mean(), 4), round(r.pers.mean(), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    rel = Path(a.release_root)
    rng = np.random.default_rng(0)
    panels = {}
    for pid, p in enumerate(tm.PANELS):
        s = time.time()
        panels[pid] = load_panel(rel, pid, p)
        print(p, f"{time.time()-s:.0f}s", flush=True)
    EVAL_M, HOLD_M = "2030-06", "2031-02"
    rows = []
    configs = [("hybrid", False, "all", []), ("hybrid", True, "all", []), ("hybrid", True, "no_freq", FREQ_COLS),
               ("hybrid", True, "no_day", DAY_COLS), ("hybrid", True, "no_freq_no_day", FREQ_COLS + DAY_COLS)]
    for mode, lomo, vname, drop in configs:
        s = time.time()
        r = {"mode": mode, "lomo_freq": lomo, "variant": vname}
        A_on, A_og, H_on, H_og, B_on, B_og, wins = [], [], [], [], [], [], []
        for pid, P in panels.items():
            mon = [d[:7] for d in P["dates"]]
            a_days = [i for i, m in enumerate(mon) if m not in (EVAL_M, HOLD_M)]
            h_days = [i for i, m in enumerate(mon) if m == HOLD_M]
            b_days = [i for i, m in enumerate(mon) if m != EVAL_M]
            fa, fa_m = month_freqs(P, a_days, lomo)
            fb, fb_m = month_freqs(P, b_days, lomo)
            o, g = gen(P, pid, mode, rng, 2, a_days, lambda dt: fa_m[dt[:7]])
            A_on += o; A_og += g
            o, g = gen(P, pid, mode, rng, 2, h_days, lambda dt: fa)
            H_on += o; H_og += g
            o, g = gen(P, pid, mode, rng, 2, b_days, lambda dt: fb_m[dt[:7]])
            B_on += o; B_og += g
            wins += official(rel, pid, tm.PANELS[pid], P, fb)
        cat = lambda x: pd.concat(x, ignore_index=True)
        A_on, A_og, H_on, H_og, B_on, B_og = map(cat, [A_on, A_og, H_on, H_og, B_on, B_og])
        models = {}
        for name, tr, te, full, top1 in [("onset", A_on, H_on, B_on, True), ("ongoing", A_og, H_og, B_og, False)]:
            m, cols = fit(tr, drop, a.rounds)
            p = m.predict(te[cols])
            grid = {thr: tm.win_iou(te[["win", "y"]], p, thr, top1) for thr in [0.2, 0.3, 0.35, 0.4, 0.5, 0.6]}
            best = max(grid, key=grid.get)
            r[f"{name}_holdFeb"] = round(grid[best], 4)
            r[f"{name}_thr"] = best
            models[name] = fit(full, drop, a.rounds)
        oc, oall, pers = eval_official(wins, *models["onset"], *models["ongoing"], r["onset_thr"], r["ongoing_thr"])
        r.update({"off_onset": oc.get("queue_onset"), "off_ongoing": oc.get("queue_ongoing"), "off_all": oall, "off_persistence": pers})
        rows.append(r)
        print(json.dumps(r), f"{time.time()-s:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
