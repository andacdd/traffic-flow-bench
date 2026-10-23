import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_model as tm

STEP_H = 5.0 / 60.0

CONFIGS = {
    "t12": dict(src="topo", up="obs", head=6, n_src=12),
    "t6": dict(src="topo", up="obs", head=6, n_src=6),
    "t24": dict(src="topo", up="obs", head=6, n_src=24),
    "tp": dict(src="topo_prof", up="obs", head=6, n_src=12),
    "tb": dict(src="topo_blend", up="obs", head=6, n_src=12),
    "t12h12": dict(src="topo", up="obs", head=12, n_src=12),
    "zsh6": dict(src="zero", up="obs", head=6),
    "ph6": dict(src="prof", up="obs", head=6),
    "zh6u": dict(src="zero", up="prof", head=6),
    "ph6u": dict(src="prof", up="prof", head=6),
    "ph6s": dict(src="prof_scaled", up="prof", head=6),
    "ph3u": dict(src="prof", up="prof", head=3),
    "ph12u": dict(src="prof", up="prof", head=12),
}


def params(rel, panel, order):
    fd = pd.read_csv(rel / "corridors" / panel / "network" / "fd_parameters.csv").set_index("link_id").loc[order]
    p = {c: fd[c].to_numpy(np.float64) for c in ["length_km", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam", "lanes"]}
    p["w"] = p["capacity_vph"] / (p["k_jam"] - p["critical_density"])
    return p


def reversed_order(rel, panel, order):
    topo = pd.read_csv(rel / "corridors" / panel / "network" / "lwr_mainline_topology.csv").set_index("mainline_link_id")
    prv = {l: set(str(x).split(";")) for l, x in topo.incoming_link_ids.items()}
    fwd = np.mean([order[i] in prv[order[i + 1]] for i in range(len(order) - 1)])
    rev = np.mean([order[i + 1] in prv[order[i]] for i in range(len(order) - 1)])
    return rev > fwd


def source_nodes(rel, panel, order):
    topo = pd.read_csv(rel / "corridors" / panel / "network" / "lwr_mainline_topology.csv").set_index("mainline_link_id").loc[order]
    inc = [set(str(x).split(";")) for x in topo.incoming_link_ids]
    out = [set(str(x).split(";")) for x in topo.outgoing_link_ids]
    ok = np.zeros(len(order), bool)
    for i in range(1, len(order)):
        conn = any(x.startswith("CONN") for x in inc[i]) or any(x.startswith("CONN") for x in out[i - 1])
        gap = order[i - 1] not in inc[i]
        ok[i] = conn or gap
    return ok


def flip(p):
    return {k: v[::-1].copy() for k, v in p.items()}


def profile(F):
    with np.errstate(invalid="ignore"):
        q = np.nanmean(F[:, 1], 0)
    q = pd.DataFrame(q).interpolate(axis=1, limit_direction="both").to_numpy()
    return q


def fill_last(X, t0, back):
    out = X[:, t0].copy()
    for k in range(1, back + 1):
        out = np.where(np.isnan(out), X[:, t0 - k], out)
    return out


def interp_links(x):
    out = x.copy()
    idx = np.arange(x.shape[1])
    for b in range(x.shape[0]):
        ok = ~np.isnan(x[b])
        if ok.any():
            out[b] = np.interp(idx, idx[ok], x[b, ok])
    return out


def recent_mean(X, t0s, n):
    with np.errstate(invalid="ignore"):
        return interp_links(np.stack([np.nanmean(X[b, t - n + 1:t + 1], 0) for b, t in enumerate(t0s)]))


def init_state(Sv, Sq, t0s, p, free_thr=0.93):
    v = interp_links(np.stack([fill_last(Sv[b:b + 1], t, 3)[0] for b, t in enumerate(t0s)]))
    q = interp_links(np.stack([fill_last(Sq[b:b + 1], t, 3)[0] for b, t in enumerate(t0s)]))
    vf, w, kj = p["free_speed_kmh"], p["w"], p["k_jam"]
    k_cong = w * kj / (np.maximum(v, 1.0) + w)
    k_free = np.minimum(q / vf, p["critical_density"])
    cong = v < free_thr * vf
    return np.where(cong, k_cong, k_free), v, cong


def simulate(k0, src, qprof, qup, qdn, node_cap, p, dt_s=0.6):
    L, vf, w, kj, C = p["length_km"], p["free_speed_kmh"], p["w"], p["k_jam"], p["capacity_vph"]
    kc = C / vf
    n = int(round(STEP_H * 3600 / dt_s))
    dt = STEP_H / n
    k = k0.copy()
    out = []
    for h in range(src.shape[1]):
        on = np.maximum(src[:, h, 1:], 0.0)
        beta = np.clip(-np.minimum(src[:, h, 1:], 0.0) / np.maximum(qprof[:, h, :-1], 1.0), 0.0, 0.5)
        acc = np.zeros_like(k)
        for _ in range(n):
            D = np.minimum(vf * k, C)
            S = np.minimum(C, w * np.maximum(kj - k, 0.0))
            Dup = D[:, :-1]
            want = Dup * (1 - beta) + on
            scale = np.where(want > S[:, 1:], S[:, 1:] / np.maximum(want, 1e-9), 1.0)
            f_ml = np.minimum(Dup * scale, node_cap)
            f_in = f_ml * (1 - beta) + on * scale
            inflow = np.empty_like(k)
            outflow = np.empty_like(k)
            inflow[:, 0] = np.minimum(qup[:, h], S[:, 0])
            inflow[:, 1:] = f_in
            outflow[:, :-1] = f_ml
            outflow[:, -1] = np.minimum(D[:, -1], qdn)
            k = np.maximum(k + dt / L * (inflow - outflow), 0.0)
            acc += k
        km = acc / n
        out.append(np.where(km <= kc, vf, w * (kj / np.maximum(km, 1e-6) - 1.0)))
    return np.stack(out, 1)


def forecast(Sv, Sq, t0s, p, prof, cfg, H=tm.H, rev=False):
    if rev:
        V = forecast(Sv[:, :, ::-1], Sq[:, :, ::-1], t0s, flip(p), prof[:, ::-1], cfg, H)
        return V[:, :, ::-1]
    B, Lk = len(t0s), Sv.shape[2]
    k0, v, cong = init_state(Sv, Sq, t0s, p)
    qrec = recent_mean(Sq, t0s, cfg["head"])
    node_cap = np.where(cong[:, :-1] & ~cong[:, 1:], qrec[:, :-1], np.inf)
    hz = np.clip(np.array(t0s)[:, None] + np.arange(1, H + 1)[None], 0, tm.T - 1)
    qprof = prof[hz]
    src = np.zeros((B, H, Lk))
    if cfg["src"] in ("prof", "prof_scaled"):
        src[:, :, 1:] = qprof[:, :, 1:] - qprof[:, :, :-1]
    if cfg["src"].startswith("topo"):
        qn = recent_mean(Sq, t0s, cfg["n_src"])
        today = np.zeros((B, Lk))
        today[:, 1:] = qn[:, 1:] - qn[:, :-1]
        pr = np.zeros((B, H, Lk))
        pr[:, :, 1:] = qprof[:, :, 1:] - qprof[:, :, :-1]
        if cfg["src"] == "topo":
            src = np.repeat(today[:, None], H, 1)
        elif cfg["src"] == "topo_prof":
            src = pr
        else:
            src = 0.5 * np.repeat(today[:, None], H, 1) + 0.5 * pr
        if cfg.get("mask") is not None:
            src = src * cfg["mask"][None, None, :]
    q3 = recent_mean(Sq, t0s, 3)
    p3 = np.stack([prof[t - 2:t + 1].mean(0) for t in t0s])
    if cfg["src"] == "prof_scaled":
        src = src * np.clip(q3.mean(1) / p3.mean(1), 0.7, 1.3)[:, None, None]
    if cfg["up"] == "obs":
        qup = np.repeat(q3[:, :1], H, 1)
    else:
        ratio = np.where(cong[:, 0], 1.0, np.clip(q3[:, 0] / np.maximum(p3[:, 0], 1.0), 0.5, 1.5))
        qup = qprof[:, :, 0] * ratio[:, None]
    qdn = np.where(cong[:, -1], q3[:, -1], 1e9)
    return simulate(k0, src, qprof, qup, qdn, node_cap, p)


def iou(p, y):
    m = ~np.isnan(y)
    p, y = p[m], y[m].astype(bool)
    u = (p | y).sum()
    return 1.0 if u == 0 else (p & y).sum() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", default=",".join(tm.PANELS))
    ap.add_argument("--max-per-panel", type=int, default=120)
    ap.add_argument("--configs", default="zsh6")
    ap.add_argument("--condition", default="queue_ongoing")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(a.windows)
    W = W[W.condition == a.condition]
    rows = []
    for panel in a.panels.split(","):
        s = time.time()
        order, vf, _, _ = tm.network(rel, panel)
        p = params(rel, panel, order)
        rev = False
        dates, F = tm.load_days(rel, panel, "train", "mainline_states", order)
        prof = profile(F)
        smask = source_nodes(rel, panel, order)
        di = {d: i for i, d in enumerate(dates)}
        w = W[W.panel == panel].head(a.max_per_panel).copy()
        t0s = np.array([o.hour * 12 + o.minute // 5 for o in pd.to_datetime(w.origin)])
        ds = np.array([di[str(x)[:10]] for x in w.date])
        Sv, Sq = F[ds, 0], F[ds, 1]
        cut = 0.6 * vf
        Y = np.stack([Sv[b, t + 1:t + tm.H + 1] for b, t in enumerate(t0s)])
        Yt = np.where(np.isnan(Y), np.nan, (Y <= cut).astype(float))
        vT = np.stack([Sv[b, t] for b, t in enumerate(t0s)])
        res = {"persistence": [iou(np.repeat((vT[b] <= cut)[None], tm.H, 0), Yt[b]) for b in range(len(w))]}
        for c in a.configs.split(","):
            V = forecast(Sv, Sq, t0s, p, prof, dict(CONFIGS[c], mask=smask), rev=rev)
            res[c] = [iou(V[b] <= cut, Yt[b]) for b in range(len(w))]
        for b in range(len(w)):
            rows.append({"panel": panel, "window_id": w.window_id.iloc[b], **{k: v[b] for k, v in res.items()}})
        print(panel, int(smask.sum()), "src nodes", f"{time.time()-s:.0f}s", {k: round(float(np.mean(v)), 3) for k, v in res.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"ctm_eval_{a.condition}.csv", index=False)
    cols = [c for c in df.columns if c not in ("panel", "window_id")]
    print(df.groupby("panel")[cols].mean().round(3).to_string())
    print("ALL", df[cols].mean().round(4).to_dict())


if __name__ == "__main__":
    main()
