import argparse
import glob
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

T = 288
H = 6
HIST = 12
SPACING = 72
MIN_COV = 0.7
MAX_PERS = 0.9
N_PER = 5
FAMILY = {p: p.rsplit("_", 1)[0] for p in tm.PANELS}
PREDS = ["persistence", "shift1"]
TRUTHS = ["fill", "obs"]


def load_panel(rel, panel):
    order, vf, _, _ = tm.network(rel, panel)
    vc = (0.6 * vf).astype(np.float32)
    idx = {l: i for i, l in enumerate(order)}
    files = sorted(glob.glob(str(rel / "corridors" / panel / "train" / "mainline_states" / "**" / "*.parquet"), recursive=True))
    dates, S, E = [], [], []
    for f in files:
        d = pq.ParquetFile(f).read(columns=["date", "timestamp", "link_id", "speed_kmh", "is_score_eligible"]).to_pandas()
        li = d.link_id.map(idx)
        d, li = d[li.notna()], li[li.notna()].astype(int).to_numpy()
        ts = pd.to_datetime(d.timestamp, utc=True)
        t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        s = np.full((T, len(order)), np.nan, np.float32)
        e = np.zeros((T, len(order)), bool)
        s[t, li] = d.speed_kmh.to_numpy(np.float32)
        e[t, li] = d.is_score_eligible.to_numpy() == 1
        dates.append(str(d.date.iloc[0]))
        S.append(s)
        E.append(e)
    S, E = np.stack(S), np.stack(E)
    U = np.stack([pd.DataFrame(s).ffill().bfill().to_numpy() <= vc for s in S])
    Qo = S <= vc
    return {"order": order, "vc": vc, "dates": dates, "S": S, "E": E, "U": U, "Qv": Qo & E, "Qo": Qo}


def csum(x):
    return np.concatenate([np.zeros((1,) + x.shape[1:], np.int32), np.cumsum(x, 0, dtype=np.int32)])


def candidates(u, qv, e, pers_row):
    ts = np.arange(HIST, T - H)
    Cu, Cq, Ce = csum(u), csum(qv), csum(e)
    hz = Cu[ts + H + 1] - Cu[ts + 1]
    hist = Cq[ts] - Cq[ts - HIST]
    cov = (Ce[ts] - Ce[ts - HIST]).sum(1) / (HIST * u.shape[1])
    p = u[ts if pers_row == "T" else ts - 1]
    inter = (p * hz).sum(1)
    union = np.where(p, H, hz).sum(1)
    pers = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    ok = (hz.sum(1) > 0) & (cov >= MIN_COV) & (pers <= MAX_PERS)
    return ts[ok], (hist.max(1) >= 2)[ok]


def panel_candidates(P, pers_row):
    rows = []
    for d in range(len(P["dates"])):
        t, og = candidates(P["U"][d], P["Qv"][d], P["E"][d], pers_row)
        rows.append(pd.DataFrame({"day": d, "t": t, "og": og}))
    c = pd.concat(rows, ignore_index=True)
    c["abs"] = c.day * T + c.t
    c["month"] = [P["dates"][d][:7] for d in c.day]
    return c.sort_values("abs").reset_index(drop=True)


def select(c, start, end, rule):
    x = c[(c["abs"] >= start) & (c["abs"] < end)]
    sel = []
    if rule == "joint":
        k = {False: 0, True: 0}
        for a, og in zip(x["abs"].to_numpy(), x.og.to_numpy()):
            if k[og] < N_PER and all(abs(a - s) >= SPACING for s, _ in sel):
                sel.append((a, og))
                k[og] += 1
                if k[False] == N_PER and k[True] == N_PER:
                    break
        return sel
    for cond in [False, True]:
        n = 0
        for a in x["abs"].to_numpy()[x.og.to_numpy() == cond]:
            if all(abs(a - s) >= SPACING for s, _ in sel):
                sel.append((a, cond))
                n += 1
                if n == N_PER:
                    break
    return sel


def chain(c, P, rule):
    out = []
    months = sorted(set(d[:7] for d in P["dates"]))
    for m in months:
        days = [i for i, d in enumerate(P["dates"]) if d[:7] == m]
        start, end, k = days[0] * T, (days[-1] + 1) * T, 0
        while start < end:
            sel = select(c, start, end, rule)
            if sum(1 for _, og in sel if not og) < N_PER or sum(1 for _, og in sel if og) < N_PER:
                break
            for a, og in sel:
                out.append((m, k, a, og))
            start = (max(a for a, _ in sel) // T + 1) * T
            k += 1
    return pd.DataFrame(out, columns=["month", "sample_id", "abs", "og"])


def iou(p, y, m=None):
    if m is not None:
        p, y = p[m], y[m]
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def window_metrics(P, d, t):
    S, U, Qv, Qo = P["S"][d], P["U"][d], P["Qv"][d], P["Qo"][d]
    yf = U[t + 1:t + H + 1]
    m = ~np.isnan(S[t + 1:t + H + 1])
    yo = Qo[t + 1:t + H + 1] & m
    preds = {"persistence": np.repeat(Qv[t - 1][None], H, 0), "shift1": U[t:t + H]}
    r = {}
    for pn, p in preds.items():
        r[f"{pn}_fill"] = iou(p, yf)
        r[f"{pn}_obs"] = iou(p, yo, m)
    hq = Qv[t - HIST:t]
    first = np.where(yf.any(1))[0]
    r.update({"tod": t, "hist_q_cells": int(hq.sum()), "hist_q_maxlink": int(hq.sum(0).max()), "q_links_T5": int(Qv[t - 1].sum()),
              "hist_cov": float(P["E"][d][t - HIST:t].mean()), "hz_cells": int(yf.sum()), "hz_links": int(yf.any(0).sum()),
              "first_q_step": int(first[0]) + 1 if len(first) else 0, "L": S.shape[1]})
    return r


def official_index(rel, split):
    w = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(rel / "task2" / "*" / split / "window_index.csv")))], ignore_index=True)
    T0 = pd.to_datetime(w.forecast_origin, utc=True)
    w["t"] = (T0.dt.hour * 12 + T0.dt.minute // 5).to_numpy()
    w["tod"] = w["t"]
    return w


def visible_descriptors(rel, split, vcs):
    w = official_index(rel, split)
    rows = []
    for panel in tm.PANELS:
        h = pd.read_parquet(rel / "task2" / panel / split / "window_history.parquet", columns=["window_id", "timestamp", "link_id", "speed_kmh", "is_score_eligible"])
        h["q"] = (h.speed_kmh <= h.link_id.map(vcs[panel])) & h.is_score_eligible.astype(bool)
        h["qa"] = h.speed_kmh <= h.link_id.map(vcs[panel])
        last = h.timestamp == h.groupby("window_id").timestamp.transform("max")
        g = h.groupby("window_id")
        mx = h.groupby(["window_id", "link_id"])[["q", "qa"]].sum().groupby("window_id").max()
        d = pd.DataFrame({"hist_q_cells": g.q.sum(), "hist_q_maxlink": mx.q, "hist_qa_maxlink": mx.qa,
                          "q_links_T5": h[last].groupby("window_id").q.sum(), "hist_cov": g.is_score_eligible.mean()})
        rows.append(d)
    d = pd.concat(rows).reset_index()
    return w.merge(d, on="window_id")


def aggregate(df, col):
    pc = df.groupby(["panel", "condition"])[col].mean().groupby("panel").mean()
    return float(pc.groupby(pc.index.map(FAMILY)).mean().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", default="joint", choices=["joint", "onset_first"])
    ap.add_argument("--pers-row", default="T", choices=["T", "T5"])
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    t_start = time.time()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    off_tr = official_index(rel, "train")
    wins, offrows, repro, vcs = [], [], [], {}
    for panel in tm.PANELS:
        s = time.time()
        P = load_panel(rel, panel)
        vcs[panel] = dict(zip(P["order"], P["vc"]))
        di = {d: i for i, d in enumerate(P["dates"])}
        wp = off_tr[off_tr.panel == panel]
        off = {(di[r.date] * T + r.t, r.condition == "queue_ongoing") for r in wp.itertuples()}
        june = [i for i, d in enumerate(P["dates"]) if d[:7] == "2030-06"]
        for pr in ["T", "T5"]:
            c = panel_candidates(P, pr)
            for rule in ["joint", "onset_first"]:
                g = select(c, june[0] * T, (june[-1] + 1) * T, rule)
                near = sum(any(og == oc and a0 // T == o // T and abs(a0 - o) <= 2 for a0, og in g) for o, oc in off)
                repro.append({"panel": panel, "pers_row": pr, "rule": rule, "exact": len(set(g) & off), "within_10min": near, "n": len(off)})
                if pr == a.pers_row and rule == a.rule:
                    ch = chain(c, P, rule)
        for r in ch.itertuples():
            d, t = int(r.abs // T), int(r.abs % T)
            wins.append({"panel": panel, "month": r.month, "sample_id": r.sample_id, "condition": "queue_ongoing" if r.og else "queue_onset",
                         "date": P["dates"][d], "origin": pd.Timestamp(P["dates"][d], tz="UTC") + pd.Timedelta(minutes=5 * t),
                         "official": (r.abs, r.og) in off, **window_metrics(P, d, t)})
        for r in wp.itertuples():
            offrows.append({"panel": panel, "condition": r.condition, "date": r.date, "window_id": r.window_id, **window_metrics(P, di[r.date], r.t)})
        print(panel, f"{len(ch)} pencere, {time.time()-s:.0f}s", flush=True)
        del P
    W = pd.DataFrame(wins)
    W["window_id"] = [f"G_{r.panel}_{r.month}_{r.sample_id:02d}_{i:05d}" for i, r in enumerate(W.itertuples())]
    O = pd.DataFrame(offrows)
    W[["window_id", "panel", "month", "sample_id", "condition", "origin", "date"]].to_parquet(out / "devset_windows.parquet", index=False)
    W.to_parquet(out / "devset_windows_metrics.parquet", index=False)
    R = pd.DataFrame(repro)
    rs = R.groupby(["pers_row", "rule"])[["exact", "within_10min", "n"]].sum().reset_index()
    rs.to_csv(out / "reproduction.csv", index=False)
    print(rs.to_string(index=False), flush=True)

    cols = [f"{p}_{t}" for p in PREDS for t in TRUTHS]
    cal = []
    for src, df in [("generated", W), ("official_train", O)]:
        for col in cols:
            cal.append({"source": src, "level": "overall", "group": "all", "metric": col, "n": len(df), "iou": df[col].mean(), "family_agg": aggregate(df, col) if src == "official_train" else np.nan})
            for lv in ["condition", "panel"] + (["month"] if src == "generated" else []):
                for k, g in df.groupby(lv):
                    cal.append({"source": src, "level": lv, "group": k, "metric": col, "n": len(g), "iou": g[col].mean(), "family_agg": np.nan})
            if src == "generated":
                for (m, cnd), g in df.groupby(["month", "condition"]):
                    cal.append({"source": src, "level": "month_condition", "group": f"{m}|{cnd}", "metric": col, "n": len(g), "iou": g[col].mean(), "family_agg": np.nan})
    C = pd.DataFrame(cal)
    rel_scores = []
    for (m, k), g in W.groupby(["month", "sample_id"]):
        if g.panel.nunique() == len(tm.PANELS):
            rel_scores.append({"month": m, "sample_id": k, **{c: aggregate(g, c) for c in cols}})
    RS = pd.DataFrame(rel_scores)
    for c in cols:
        C.loc[len(C)] = {"source": "generated", "level": "pseudo_release", "group": f"n={len(RS)}", "metric": c, "n": len(RS), "iou": RS[c].mean(), "family_agg": RS[c].mean()}
    C.to_csv(out / "calibration.csv", index=False)

    pools = {p: [g for _, g in W[W.panel == p].groupby(["month", "sample_id"])] for p in tm.PANELS}
    noise = []
    for c in cols + ["diff_fill", "diff_obs"]:
        x = W.assign(diff_fill=W.shift1_fill - W.persistence_fill, diff_obs=W.shift1_obs - W.persistence_obs)
        v = x.groupby(["panel", "condition"])[c].var()
        se_an = float(np.sqrt((v * N_PER).sum()) / (N_PER * 2 * len(tm.PANELS)))
        bs = []
        for _ in range(a.boot):
            g = pd.concat([pools[p][rng.integers(len(pools[p]))] for p in tm.PANELS])
            g = g.assign(diff_fill=g.shift1_fill - g.persistence_fill, diff_obs=g.shift1_obs - g.persistence_obs)
            bs.append(aggregate(g, c))
        r = {"metric": c, "window_mean": float(x[c].mean()), "window_std": float(x[c].std()), "std_onset": float(x[x.condition == "queue_onset"][c].std()),
             "std_ongoing": float(x[x.condition == "queue_ongoing"][c].std()), "se80_stratified": se_an, "se80_naive": float(x[c].std() / np.sqrt(80)),
             "se80_bootstrap_release": float(np.std(bs)), "boot_mean": float(np.mean(bs))}
        if c in cols and len(RS) > 1:
            r["pseudo_release_std"] = float(RS[c].std())
        noise.append(r)
    N = pd.DataFrame(noise)
    N.to_csv(out / "noise.csv", index=False)

    V = visible_descriptors(rel, "validation", vcs)
    Pv = visible_descriptors(rel, "private", vcs)
    Tv = visible_descriptors(rel, "train", vcs)
    rule_chk = {}
    for nm, df in [("train", Tv), ("validation", V), ("private", Pv)]:
        og = df.condition == "queue_ongoing"
        rule_chk[nm] = {"elig_rule_agree": float(((df.hist_q_maxlink >= 2) == og).mean()), "all_rule_agree": float(((df.hist_qa_maxlink >= 2) == og).mean())}
    print("kosul kurali (gorunur gecmis):", json.dumps(rule_chk), flush=True)
    desc = []
    dcols = ["tod", "hist_q_cells", "q_links_T5", "hist_cov"]
    for src, df in [("generated", W), ("official_train", O), ("official_train_vis", Tv), ("validation_vis", V), ("private_vis", Pv)]:
        for cnd, g in df.groupby("condition"):
            r = {"source": src, "condition": cnd, "n": len(g), **{c: float(g[c].mean()) for c in dcols}, "am_share": float((g.tod < 144).mean())}
            for c in ["hz_cells", "hz_links", "first_q_step"]:
                if c in g:
                    r[c] = float(g[c].mean())
            desc.append(r)
    for (m, cnd), g in W.groupby(["month", "condition"]):
        desc.append({"source": f"generated_{m}", "condition": cnd, "n": len(g), **{c: float(g[c].mean()) for c in dcols}, "am_share": float((g.tod < 144).mean()),
                     "hz_cells": float(g.hz_cells.mean()), "hz_links": float(g.hz_links.mean()), "first_q_step": float(g.first_q_step.mean())})
    D = pd.DataFrame(desc)
    D.to_csv(out / "descriptors.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, cnd in enumerate(["queue_onset", "queue_ongoing"]):
        for nm, df, col in [("uretilen persistence", W, "persistence_fill"), ("uretilen shift1", W, "shift1_fill"), ("resmi train persistence", O, "persistence_fill")]:
            ax[i].hist(df[df.condition == cnd][col], bins=np.linspace(0, 1, 21), alpha=0.5, density=True, label=nm)
        ax[i].set_title(f"{cnd} IoU dagilimi (fill truth)")
        ax[i].set_xlabel("pencere IoU")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "iou_dist_condition.png", dpi=110)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(0, 25, 1)
    for i, cnd in enumerate(["queue_onset", "queue_ongoing"]):
        for nm, df in [("uretilen", W), ("resmi train", O), ("validation", V), ("private", Pv)]:
            ax[i].hist(df[df.condition == cnd].tod / 12, bins=bins, alpha=0.4, density=True, label=nm)
        ax[i].set_title(f"{cnd} origin saati (UTC)")
        ax[i].set_xlabel("saat")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "tod_hist.png", dpi=110)

    summ = {"n_windows": len(W), "n_samples": int(W.groupby(["panel", "month", "sample_id"]).ngroups), "n_pseudo_releases": len(RS),
            "official_in_generated": int(W.official.sum()), "onset_first_q_at_T30": float((W[W.condition == "queue_onset"].first_q_step == H).mean()),
            "runtime_s": round(time.time() - t_start, 1), "rule": a.rule, "pers_row": a.pers_row, "condition_rule_check": rule_chk}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ), flush=True)
    print(C[C.level.isin(["overall", "condition", "pseudo_release"])].to_string(index=False), flush=True)
    print(N.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
