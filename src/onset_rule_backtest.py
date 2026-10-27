import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm
from online_refit import WEEKS, build, fit, iou
from task2_predict_sel import eiou_select

TOD_LO, TOD_HI = 12, 18


def month_rates(rel):
    out = {}
    for pid, panel in enumerate(tm.PANELS):
        order, vf, _, _ = tm.network(rel, panel)
        vd, VA = tm.load_days(rel, panel, "validation", "mainline_states_masked", order)
        out[pid] = (list(vd), np.stack([tm.queue_matrix(VA[i, 0], vf) for i in range(len(vd))]))
    return out


def prior_rate(rates, pid, date, t0):
    dates, Q = rates[pid]
    prior = [i for i, d in enumerate(dates) if d < date]
    if not prior:
        return 0, np.zeros(Q.shape[2])
    w = Q[prior, max(0, t0 - TOD_LO):t0 + TOD_HI]
    seen = (~np.isnan(w)).any(1)
    hit = (np.nan_to_num(w) > 0).any(1)
    n = seen.sum(0)
    return len(prior), np.where(n > 0, hit.sum(0) / np.maximum(n, 1), 0.0)


def rules(p, freq, n, rate):
    out = {"B": p}
    for fthr in (0.05, 0.15):
        for a in (0.0, 0.3):
            cl = (n >= 10) & (freq >= fthr) & (rate < 0.05)
            out[f"close_f{int(fthr*100)}_a{int(a*10)}"] = np.where(cl, p * a, p)
    for rthr in (0.3, 0.5):
        for g in (0.5, 1.0):
            op = (n >= 10) & (freq < 0.05) & (rate >= rthr)
            out[f"open_r{int(rthr*10)}_g{int(g*10)}"] = np.where(op, np.maximum(p, g * rate), p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--out", default="reports/queue/onset_rule")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--weight", type=float, default=3.0)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr_on, _, ev_on, _ = build(rel, 1)
    rates = month_rates(rel)
    res = []
    for k in range(1, len(WEEKS)):
        te = ev_on[ev_on.wk == k]
        if te.empty:
            continue
        s = time.time()
        m, cols = fit(tr_on[(tr_on.src == 0) | (tr_on.wk < k)], a.rounds, a.weight, True)
        p = m.predict(te[cols])
        for win, idx in te.groupby("win").indices.items():
            _, pid, date, t0 = win.split("|")
            n, rate = prior_rate(rates, int(pid), date, int(t0))
            y = te.y.to_numpy()[idx]
            freq = te.freq_max_h.to_numpy()[idx]
            for name, pp in rules(p[idx], freq, n, rate).items():
                res.append({"wk": k, "win": win, "n_prior": n, "variant": name, "iou": iou(eiou_select(pp), y), "rare": float(np.nanmean(freq[np.nan_to_num(y) > 0] < 0.02)) if (np.nan_to_num(y) > 0).any() else np.nan})
        print(f"week {k}: {te.win.nunique()} windows {time.time()-s:.0f}s", flush=True)
    r = pd.DataFrame(res)
    r.to_csv(out / "onset_rule_backtest.csv", index=False)
    t = r.groupby("variant").iou.mean().to_frame("all")
    t["n10"] = r[r.n_prior >= 10].groupby("variant").iou.mean()
    t["rare"] = r[r.rare > 0.5].groupby("variant").iou.mean()
    t["normal"] = r[r.rare <= 0.5].groupby("variant").iou.mean()
    t = t.sub(t.loc["B"], axis=1).assign(B_all=t.loc["B", "all"])
    print(t.sort_values("n10", ascending=False).round(4).to_string(), flush=True)
    print("windows", r.win.nunique(), "n>=10", r[r.n_prior >= 10].win.nunique(), flush=True)
    (out / "summary.json").write_text(t.round(5).reset_index().to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
