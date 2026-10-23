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
from task2_predict_sel import STEP, eiou_select


def build(rel):
    rng = np.random.default_rng(0)
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
        o, _ = td.gen(P, pid, "hybrid", rng, 2, days, lambda dt: by_m[dt[:7]])
        for x in o:
            for c, v in memory(R, gidx[P["dates"][int(x.day.iloc[0])]], int(x.tod.iloc[0])).items():
                x[c] = v
        rows += [x.assign(src=0, wk=-1, split="train") for x in o]
        for split, (sd, SA, SQ) in split_data.items():
            o, _ = eval_rows(SA, SQ, sd, gidx, pid, m, R, rng, 0, False)
            rows += [x.assign(split=split) for x in o]
            di = {d: i for i, d in enumerate(sd)}
            w = pd.read_csv(rel / "task2" / panel / split / "window_index.csv")
            for r in w[w.condition == "queue_onset"].itertuples():
                T0 = pd.Timestamp(r.forecast_origin)
                t0 = T0.hour * 12 + T0.minute // 5
                X = tm.features(SA[di[r.date]], t0, m["vf"], m["cap"], m["lanes"], m["freq"], T0.weekday(), pid)
                X = X.assign(**memory(R, gidx[r.date], t0))
                wins.append(X.assign(link_id=P["order"], window_id=r.window_id, T0=T0, split=split, wk=week(r.date)))
        print(panel, f"{time.time()-s:.0f}s", flush=True)
        del P, split_data, TQ, R
        gc.collect()
    return pd.concat(rows, ignore_index=True), wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--base-queue", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--weight", type=float, default=3.0)
    ap.add_argument("--daily", action="store_true")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, wins = build(rel)
    edate = rows.win.str.split("|").str[2].where(rows.split != "train", "")
    if a.daily:
        tdays = sorted({(X.split.iloc[0], str(X.T0.iloc[0].date())) for X in wins})
        plan = [(sp, dd, (rows.split == "train") | ((rows.split != "train") & (edate < dd))) for sp, dd in tdays if sp == "private" or edate[rows.split == "validation"].lt(dd).any()]
        key = lambda X: str(X.T0.iloc[0].date())
    else:
        plan = [("validation", k, (rows.split == "train") | ((rows.split == "validation") & (rows.wk < k))) for k in range(1, 4)]
        plan += [("private", k, (rows.split != "private") | ((rows.split == "private") & (rows.wk < k))) for k in range(0, 4)]
        key = lambda X: X.wk.iloc[0]
    preds, used = [], {}
    for split, k, sel in plan:
        targets = [X for X in wins if X.split.iloc[0] == split and key(X) == k]
        if not targets:
            continue
        s = time.time()
        tr = rows[sel]
        m, cols = fit(tr, a.rounds, a.weight, True)
        last = sorted(set(edate[sel & (rows.split != "train")]))
        used[f"{split}_wk{k}"] = {"rows": int(len(tr)), "eval_rows": int((tr.src == 1).sum()), "last_eval_day": last[-1] if last else None,
                                  "first_target_day": min(str(X.T0.iloc[0].date()) for X in targets)}
        assert used[f"{split}_wk{k}"]["last_eval_day"] is None or used[f"{split}_wk{k}"]["last_eval_day"] < used[f"{split}_wk{k}"]["first_target_day"]
        for X in targets:
            sel_l = eiou_select(m.predict(X[cols]))
            ts = (X.T0.iloc[0] + STEP * tm.H).strftime("%Y-%m-%dT%H:%M:%SZ")
            preds.append(pd.DataFrame({"window_id": X.window_id.iloc[0], "timestamp": ts, "link_id": X.link_id, "new": sel_l.astype(int)}))
        print(split, k, len(targets), f"{time.time()-s:.0f}s", flush=True)
    new = pd.concat(preds, ignore_index=True)
    q = pd.read_csv(a.base_queue)
    redo = q.window_id.isin(new.window_id.unique())
    q.loc[redo, "queue_pred"] = 0
    q = q.merge(new, on=["window_id", "timestamp", "link_id"], how="left")
    assert q.loc[q.new.notna()].shape[0] == len(new)
    q["queue_pred"] = np.where(q.new.notna(), q.new, q.queue_pred).astype(int)
    q.drop(columns="new").to_csv(out / "queue_vp.csv", index=False)
    used["changed_windows"] = int(new.window_id.nunique())
    used["pred_rate"] = float(q.queue_pred.mean())
    (out / "config.json").write_text(json.dumps(used, indent=2))
    print(json.dumps(used), flush=True)


if __name__ == "__main__":
    main()
