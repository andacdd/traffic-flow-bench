import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm


def nanmean_stack(arrs):
    s = np.nansum(arrs, 0)
    n = np.sum(~np.isnan(arrs), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def shifted(S, dt, dl):
    out = np.full_like(S, np.nan)
    T, L = S.shape
    t0, t1 = max(0, -dt), min(T, T - dt)
    l0, l1 = max(0, -dl), min(L, L - dl)
    out[t0:t1, l0:l1] = S[t0 + dt:t1 + dt, l0 + dl:l1 + dl]
    return out


def smooth(S, kind):
    if kind == "obs":
        return S
    if kind == "t3":
        return nanmean_stack(np.stack([shifted(S, d, 0) for d in (-1, 0, 1)]))
    if kind == "t5":
        return nanmean_stack(np.stack([shifted(S, d, 0) for d in (-2, -1, 0, 1, 2)]))
    if kind == "st3":
        return nanmean_stack(np.stack([shifted(S, a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]))
    if kind == "med3":
        st = np.stack([shifted(S, d, 0) for d in (-1, 0, 1)])
        return np.nanmedian(st, 0)
    if kind == "medst3":
        st = np.stack([shifted(S, a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
        return np.nanmedian(st, 0)
    if kind == "s3":
        return nanmean_stack(np.stack([shifted(S, 0, d) for d in (-1, 0, 1)]))
    raise ValueError(kind)


KINDS = ["obs", "t3", "t5", "med3", "s3", "st3", "medst3"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, noise = [], []
    for panel in tm.PANELS:
        order, vf, cap, lanes = tm.network(rel, panel)
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        di = {d: i for i, d in enumerate(dates)}
        S = F[:, 0]
        d2 = S[:, 1:-1] - 0.5 * (S[:, :-2] + S[:, 2:])
        free = S[:, 1:-1] > 0.9 * vf
        cong = S[:, 1:-1] < 0.6 * vf
        noise.append({"panel": panel, "sigma_free": float(np.nanstd(d2[free]) / np.sqrt(1.5)), "sigma_cong": float(np.nanstd(d2[cong]) / np.sqrt(1.5)),
                      "frac_cong": float(np.nanmean(S < 0.6 * vf))})
        w = pd.read_csv(rel / "task2" / panel / "train" / "window_index.csv")
        cache = {}
        for r in w.itertuples():
            T0 = pd.Timestamp(r.forecast_origin)
            t0 = T0.hour * 12 + T0.minute // 5
            d = di[r.date]
            rec = {"panel": panel, "cond": r.condition, "date": r.date, "t0": t0}
            hz = {}
            for k in KINDS:
                if (d, k) not in cache:
                    cache[(d, k)] = tm.queue_matrix(smooth(S[d], k), vf)
                Q = cache[(d, k)]
                Y = Q[t0 + 1:t0 + tm.H + 1]
                hz[k] = Y
                rec[f"{k}_early"] = float(np.nansum(Y[:-1]) > 0)
                rec[f"{k}_last"] = float(np.nansum(Y[-1]) > 0)
                rec[f"{k}_hist"] = float(np.nansum(Q[t0 - tm.HIST:t0 + 1]) > 0)
                rec[f"{k}_ncell"] = float(np.nansum(Y))
                pers = np.repeat((Q[t0] == 1)[None], tm.H, 0)
                rec[f"{k}_pers"] = tm.iou(pers.ravel(), Y.ravel())
            for k in KINDS:
                rec[f"{k}_iou_obs"] = tm.iou(hz["obs"].ravel() == 1, hz[k].ravel())
            rows.append(rec)
        print(panel, json.dumps(noise[-1]), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "latent_windows.csv", index=False)
    pd.DataFrame(noise).to_csv(out / "latent_noise.csv", index=False)
    summ = df.groupby("cond")[[c for c in df.columns if any(c.startswith(k + "_") for k in KINDS)]].mean().T.round(3)
    summ.to_csv(out / "latent_summary.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
