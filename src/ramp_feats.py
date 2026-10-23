import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T = 288
HIST = 12
RAMP_COLS = ["on_last", "on_slope", "off_last", "off_slope", "on_up3", "off_dn3", "corr_on", "corr_off"]


def load(rel, panel, split, order):
    net = Path(rel) / "corridors" / panel / "network"
    am = pd.read_csv(net / "ramp_attachment_map.csv")
    lix = {l: i for i, l in enumerate(order)}
    am = am[am.nearest_mainline_link_id.isin(lix)].reset_index(drop=True)
    rix = {r: i for i, r in enumerate(am.ramp_link_id)}
    files = sorted(glob.glob(str(Path(rel) / "corridors" / panel / split / "ramp_states" / "**" / "*.parquet"), recursive=True))
    dates, arrs = [], []
    for f in files:
        d = pq.ParquetFile(f).read(columns=["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed"]).to_pandas()
        ri = d.ramp_link_id.map(rix)
        d = d[ri.notna()]
        ri = ri[ri.notna()].astype(int).to_numpy()
        ts = pd.to_datetime(d.timestamp, utc=True)
        t = (ts.dt.hour * 12 + ts.dt.minute // 5).to_numpy()
        a = np.full((T, len(am)), np.nan, np.float32)
        v = np.where(d.pct_observed.to_numpy() >= 75, d.flow_vph.to_numpy(np.float32), np.nan)
        a[t, ri] = v
        dates.append(str(d.date.iloc[0]))
        arrs.append(a)
    meta = {"link": am.nearest_mainline_link_id.map(lix).to_numpy(), "is_on": (am.ramp_type == "OR").to_numpy(), "L": len(order)}
    return dates, np.stack(arrs), meta


def scale(R):
    with np.errstate(all="ignore"):
        s = np.nanmean(R, axis=(0, 1))
    return np.where(np.isfinite(s) & (s > 0), s, 1.0).astype(np.float32)


def last_valid(x):
    out = np.full(x.shape[1:], np.nan, np.float32)
    for k in range(x.shape[0]):
        out = np.where(np.isnan(x[k]), out, x[k])
    return out


def features(Rday, t0, meta, sc):
    L = meta["L"]
    h = Rday[t0 - HIST:t0 + 1] / sc[None]
    last = last_valid(h)
    with np.errstate(all="ignore"):
        slope = np.nanmean(h[-3:], 0) - np.nanmean(h[:3], 0)
    cols = {}
    for name, m in [("on", meta["is_on"]), ("off", ~meta["is_on"])]:
        for k, x in [("last", last), ("slope", slope)]:
            s = np.zeros(L, np.float32)
            n = np.zeros(L, np.float32)
            ok = m & np.isfinite(x)
            np.add.at(s, meta["link"][ok], x[ok])
            np.add.at(n, meta["link"][ok], 1)
            cols[f"{name}_{k}"] = np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)
        with np.errstate(all="ignore"):
            cols[f"corr_{name}"] = np.full(L, np.nanmean(last[m]) if (m & np.isfinite(last)).any() else np.nan, np.float32)
    on = np.nan_to_num(cols["on_last"])
    off = np.nan_to_num(cols["off_last"])
    c = np.concatenate([[0], np.cumsum(on)])
    idx = np.arange(L)
    cols["on_up3"] = (c[idx + 1] - c[np.maximum(idx - 3, 0)]).astype(np.float32)
    c = np.concatenate([[0], np.cumsum(off)])
    cols["off_dn3"] = (c[np.minimum(idx + 4, L)] - c[idx]).astype(np.float32)
    return cols
