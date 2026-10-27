import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm
from online_refit import RARE, calendar, eval_rows, fit, memory, week


def build(rel, og_per_day, seed=0):
    rng = np.random.default_rng(seed)
    rows, wins = [], []
    for pid, panel in enumerate(tm.PANELS):
        s = time.time()
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        full, by_m = td.month_freqs(P, days, True)
        rare = full < RARE
        split_data = {}
        for split in ["validation", "private"]:
            sd, SA = tm.load_days(rel, panel, split, "mainline_states_masked", P["order"])
            split_data[split] = (sd, SA, np.stack([tm.queue_matrix(SA[i, 0], P["vf"]) for i in range(len(sd))]))
        TQ = np.stack([tm.queue_matrix(P["M"][d, 0], P["vf"]) for d in days])
        R, gidx = calendar(list(P["dates"]) + list(split_data["validation"][0]) + list(split_data["private"][0]),
                           np.concatenate([TQ, split_data["validation"][2], split_data["private"][2]]), rare)
        m = {"vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full}
        _, g = td.gen(P, pid, "hybrid", rng, og_per_day, days, lambda dt: by_m[dt[:7]])
        for x in g:
            for c, v in memory(R, gidx[P["dates"][int(x.day.iloc[0])]], int(x.tod.iloc[0])).items():
                x[c] = v
        rows += [x.assign(src=0, wk=-1, split="train") for x in g]
        for split, (sd, SA, SQ) in split_data.items():
            _, g = eval_rows(SA, SQ, sd, gidx, pid, m, R, rng, og_per_day, False)
            rows += [x.assign(split=split) for x in g]
            di = {d: i for i, d in enumerate(sd)}
            w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
            for r in w[w.condition == "queue_ongoing"].itertuples():
                T0 = pd.Timestamp(r.forecast_origin)
                t0 = T0.hour * 12 + T0.minute // 5
                X = tm.features(SA[di[r.date]], t0, m["vf"], m["cap"], m["lanes"], m["freq"], T0.weekday(), pid)
                X = X.assign(**memory(R, gidx[r.date], t0))
                wins.append(X.assign(link_id=P["order"], window_id=r.window_id, T0=T0, split=split, wk=week(r.date), t0=t0))
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P, split_data, TQ, R
        gc.collect()
    return pd.concat(rows, ignore_index=True), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--out", default="submissions/queue_variants/og_refit")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--og-per-day", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, wins = build(rel, a.og_per_day, a.seed)
    edate = rows.win.str.split("|").str[2].where(rows.split != "train", "")
    plan = [("validation", k, (rows.split == "train") | ((rows.split == "validation") & (rows.wk < k))) for k in range(0, 4)]
    plan += [("private", k, (rows.split != "private") | ((rows.split == "private") & (rows.wk < k))) for k in range(0, 4)]
    preds, used = [], {}
    for split, k, sel in plan:
        targets = [X for X in wins if X.split.iloc[0] == split and X.wk.iloc[0] == k]
        if not targets:
            continue
        s = time.time()
        tr = rows[sel]
        m, cols = fit(tr, a.rounds, a.weight, True)
        last = sorted(set(edate[sel & (rows.split != "train")]))
        first = min(str(X.T0.iloc[0].date()) for X in targets)
        used[f"{split}_wk{k}"] = {"rows": int(len(tr)), "eval_rows": int((tr.src == 1).sum()), "last_eval_day": last[-1] if last else None, "first_target_day": first}
        assert not last or last[-1] < first
        for X in targets:
            t0 = int(X.t0.iloc[0])
            for h in range(1, tm.H + 1):
                Xh = X.assign(h=h)
                preds.append(pd.DataFrame({"window_id": X.window_id.iloc[0], "h": h, "link_id": X.link_id, "p_ref": m.predict(Xh[cols])}))
        print(split, k, len(targets), f"{time.time()-s:.0f}s", flush=True)
    pr = pd.concat(preds, ignore_index=True)
    pr.to_parquet(out / "probs.parquet", index=False)
    used["windows"] = int(pr.window_id.nunique())
    (out / "config.json").write_text(json.dumps(used, indent=2))
    print(json.dumps(used), flush=True)


if __name__ == "__main__":
    main()
