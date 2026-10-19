import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T = 288
COLS = ["date", "timestamp", "link_id", "speed_kmh", "flow_vph"]


def masked_frame(rel, panel, split):
    files = sorted((rel / "corridors" / panel / split / "mainline_states_masked").glob("**/*.parquet"))
    parts = [pq.ParquetFile(f).read(columns=COLS).to_pandas() for f in files]
    df = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["tod"] = (ts.dt.hour * 12 + ts.dt.minute // 5).astype(np.int16)
    df["dow"] = ts.dt.weekday.astype(np.int8)
    df["date"] = df["date"].astype(str)
    df["k"] = df["flow_vph"] / df["speed_kmh"].clip(lower=1.0)
    return df


def link_order(rel, panel):
    topo = pd.read_csv(rel / "corridors" / panel / "network" / "lwr_mainline_topology.csv")
    order = topo.sort_values("order_index").mainline_link_id.tolist()
    lens = topo.set_index("mainline_link_id").loc[order, "length_km"].to_numpy()
    pos = np.concatenate([[0.0], np.cumsum((lens[:-1] + lens[1:]) / 2)])
    return order, pos


def profiles(train, order):
    idx = {l: i for i, l in enumerate(order)}
    L = len(order)
    out = {}
    for col in ["speed_kmh", "flow_vph", "k"]:
        g = train.dropna(subset=[col]).groupby(["dow", "tod", "link_id"])[col].mean()
        arr = np.full((7, T, L), np.nan)
        d, t, l = g.index.get_level_values(0), g.index.get_level_values(1), g.index.get_level_values(2).map(idx)
        ok = ~pd.isna(l)
        arr[d[ok], t[ok], l[ok].astype(int)] = g.to_numpy()[ok]
        tod_mean = np.nanmean(arr, axis=0)
        for k in range(7):
            m = np.isnan(arr[k])
            arr[k][m] = tod_mean[m]
        out[col] = arr
    return out


def interp_axis(R, coord):
    n, m = R.shape
    est = np.full_like(R, np.nan)
    dist = np.full_like(R, np.inf)
    for j in range(m):
        y = R[:, j]
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            continue
        xo = coord[ok]
        est[:, j] = np.interp(coord, xo, y[ok])
        k = np.searchsorted(xo, coord)
        left = np.where(k > 0, coord - xo[np.clip(k - 1, 0, len(xo) - 1)], np.inf)
        right = np.where(k < len(xo), xo[np.clip(k, 0, len(xo) - 1)] - coord, np.inf)
        dist[:, j] = np.minimum(left, right)
    return est, dist


def reconstruct_day(obs, prof, pos, a):
    R = obs - prof
    et, dt = interp_axis(R, np.arange(T, dtype=float))
    es, ds = interp_axis(R.T, pos)
    es, ds = es.T, ds.T
    wt = 1.0 / (dt + a["t0"]) ** a["p"]
    ws = 1.0 / (ds / a["s_km"] + a["t0"]) ** a["p"]
    et = np.nan_to_num(et)
    es = np.nan_to_num(es)
    wt[~np.isfinite(dt)] = 0
    ws[~np.isfinite(ds)] = 0
    wp = a["w_prof"]
    den = wt + ws + wp
    res = np.divide(wt * et + ws * es, den, out=np.zeros_like(den), where=den > 0)
    out = prof + res
    keep = ~np.isnan(obs)
    out[keep] = obs[keep]
    return out


def build(rel, panel, split, a, train_prof=None):
    order, pos = link_order(rel, panel)
    idx = {l: i for i, l in enumerate(order)}
    tr = train_prof or profiles(masked_frame(rel, panel, "train"), order)
    df = masked_frame(rel, panel, split)
    df["li"] = df.link_id.map(idx)
    df = df.dropna(subset=["li"])
    df["li"] = df.li.astype(int)
    tpl = pd.read_csv(rel / "task1" / panel / split / "sample_submission_state.csv", dtype={"timestamp": str})
    tpl["date"] = tpl.timestamp.str[:10]
    ts = pd.to_datetime(tpl.timestamp, utc=True)
    tpl["tod"] = (ts.dt.hour * 12 + ts.dt.minute // 5).astype(int)
    tpl["li"] = tpl.link_id.map(idx)
    sp = np.full(len(tpl), np.nan)
    fl = np.full(len(tpl), np.nan)
    kk = np.full(len(tpl), np.nan)
    tgroups = tpl.groupby("date").indices
    for date, g in df.groupby("date"):
        if date not in tgroups:
            continue
        dow = pd.Timestamp(date).weekday()
        rows = tgroups[date]
        for col, store in [("speed_kmh", sp), ("flow_vph", fl), ("k", kk)]:
            obs = np.full((T, len(order)), np.nan)
            obs[g.tod.to_numpy(), g.li.to_numpy()] = g[col].to_numpy()
            rec = reconstruct_day(obs, tr[col][dow], pos, a)
            store[rows] = rec[tpl.tod.to_numpy()[rows], tpl.li.to_numpy()[rows]]
    vf = pd.read_csv(rel / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").free_speed_kmh
    rec = a.get("reconcile", "none")
    if rec != "none":
        ratio = np.clip(kk * sp / np.clip(fl, 1.0, None), 0.2, 5.0)
        mode, _, thr = rec.partition(":")
        w = {"q": 1.0, "v": 0.0, "geo": 0.5}[mode]
        if thr:
            vfl = tpl.link_id.map(vf).fillna(117.0).to_numpy()
            ratio = np.where(sp < float(thr) * vfl, ratio, 1.0)
        fl = fl * ratio ** w
        sp = sp / ratio ** (1.0 - w)
    tpl["speed_kmh"] = np.clip(sp, 5.0, np.minimum(tpl.link_id.map(vf).fillna(120.75).to_numpy() + 4.5, 120.75))
    tpl["flow_vph"] = np.clip(fl, 0.0, None)
    return tpl[["panel", "timestamp", "station_id", "link_id", "mask_regime", "speed_kmh", "flow_vph"]], tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--split", default="train", choices=["train", "validation", "private"])
    ap.add_argument("--panel", action="append")
    ap.add_argument("--output", required=True)
    ap.add_argument("--t0", type=float, default=2.0)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--s-km", type=float, default=0.25)
    ap.add_argument("--w-prof", type=float, default=0.0)
    ap.add_argument("--reconcile", default="none")
    args = ap.parse_args()
    rel = Path(args.release_root)
    a = {"t0": args.t0, "p": args.p, "s_km": args.s_km, "w_prof": args.w_prof, "reconcile": args.reconcile}
    panels = args.panel or sorted(p.name for p in (rel / "corridors").iterdir())
    outs = []
    for p in panels:
        s = time.time()
        o, _ = build(rel, p, args.split, a)
        print(f"{p}: {len(o):,} satır, NaN {int(o[['speed_kmh','flow_vph']].isna().sum().sum())}, {time.time()-s:.0f}s", flush=True)
        outs.append(o)
    pd.concat(outs).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
