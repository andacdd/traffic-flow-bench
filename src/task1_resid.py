import argparse
import gc
import resource
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task1_interp import T, interp_axis, link_order
from task3_fast import Proxy
from task1.baseline_task1_historical_mean import lane_vector, station_lanes

FOLDS = {
    1: ["2030-06", "2030-07", "2030-08"],
    2: ["2030-09", "2030-10", "2030-11"],
    3: ["2030-12", "2031-01", "2031-02"],
}
A = {"t0": 2.0, "p": 1.0, "s_km": 0.25}
VARS = ["v", "q", "k"]
REG = {"R1": 1, "R2": 2, "R3": 3}
GEO = ["length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]
STAGES = {}


def tick(name, s):
    STAGES[name] = STAGES.get(name, 0.0) + time.time() - s


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def feature_names():
    f = ["v0", "q0", "k0", "v06", "queued0", "vf_ratio0"]
    for c in VARS:
        f += [f"et_{c}", f"es_{c}", f"prof_{c}"]
    f += ["dt", "ds_km", "gap_len", "gap_from_start", "gap_from_end"]
    for c in VARS:
        for o in [-2, -1, 1, 2]:
            f.append(f"lag{o:+d}_{c}")
    for c in VARS:
        for o in [-3, -2, -1, 1, 2, 3]:
            f.append(f"nb{o:+d}_{c}")
    for c in ["v", "q"]:
        for o in [-3, -2, -1, 1, 2, 3]:
            f.append(f"nbr{o:+d}_{c}")
    f += ["tod", "dow", "regime", "ramp_on", "ramp_off"]
    f += GEO + ["pos_frac", "has_sensor"]
    return f


FEATS = feature_names()


def grid_index(t, days):
    secs = pc.cast(pc.strptime(t["timestamp"], format="%Y-%m-%dT%H:%M:%SZ", unit="s"), "int64").to_numpy()
    tod = ((secs % 86400) // 300).astype(np.int64)
    di = days.get_indexer(pd.Series(t["date"].to_numpy(zero_copy_only=False)).astype(str))
    return di, tod


class Panel:
    def __init__(self, rel, panel):
        s = time.time()
        self.name = panel
        pdir = rel / "corridors" / panel
        self.order, self.pos = link_order(rel, panel)
        self.L = len(self.order)
        lix = pd.Index(self.order)
        tfiles = sorted((pdir / "train" / "mainline_states").glob("**/*.parquet"))
        mfiles = sorted((pdir / "train" / "mainline_states_masked").glob("**/*.parquet"))
        dates = sorted({pq.ParquetFile(f).read(columns=["date"])["date"][0].as_py().__str__()[:10] for f in tfiles})
        self.days = pd.Index(dates)
        self.D = len(dates)
        self.dow = np.array([pd.Timestamp(d).weekday() for d in dates])
        self.month = np.array([d[:7] for d in dates])
        shape = (self.D, T, self.L)
        self.obs = {c: np.full(shape, np.nan) for c in VARS}
        for f in mfiles:
            t = pq.ParquetFile(f).read(columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph"])
            di, tod = grid_index(t, self.days)
            li = lix.get_indexer(t["link_id"].to_numpy(zero_copy_only=False))
            ok = (li >= 0) & (di >= 0)
            sp = t["speed_kmh"].to_numpy(zero_copy_only=False).astype(float)
            fl = t["flow_vph"].to_numpy(zero_copy_only=False).astype(float)
            self.obs["v"][di[ok], tod[ok], li[ok]] = sp[ok]
            self.obs["q"][di[ok], tod[ok], li[ok]] = fl[ok]
            self.obs["k"][di[ok], tod[ok], li[ok]] = (fl / np.clip(sp, 1.0, None))[ok]
        self.tv = np.full(shape, np.nan, dtype=np.float32)
        self.tq = np.full(shape, np.nan, dtype=np.float32)
        for f in tfiles:
            t = pq.ParquetFile(f).read(columns=["date", "timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
            di, tod = grid_index(t, self.days)
            li = lix.get_indexer(t["link_id"].to_numpy(zero_copy_only=False))
            ok = (li >= 0) & (di >= 0) & (t["is_score_eligible"].to_numpy(zero_copy_only=False) == 1)
            self.tv[di[ok], tod[ok], li[ok]] = t["speed_kmh"].to_numpy(zero_copy_only=False)[ok]
            self.tq[di[ok], tod[ok], li[ok]] = t["flow_vph"].to_numpy(zero_copy_only=False)[ok]
        tpl = pcsv.read_csv(
            rel / "task1" / panel / "train" / "sample_submission_state.csv",
            convert_options=pcsv.ConvertOptions(
                include_columns=["timestamp", "station_id", "link_id", "mask_regime"],
                column_types={"timestamp": "string", "station_id": "string", "link_id": "string", "mask_regime": "string"},
            ),
        )
        tpl = tpl.append_column("date", pc.utf8_slice_codeunits(tpl["timestamp"], 0, 10))
        di, tod = grid_index(tpl, self.days)
        links = pd.Series(tpl["link_id"].to_numpy(zero_copy_only=False))
        stations = pd.Series(tpl["station_id"].to_numpy(zero_copy_only=False))
        li = lix.get_indexer(links)
        self.n_tpl_drop = int(((li < 0) | (di < 0)).sum())
        ok = (li >= 0) & (di >= 0)
        self.td, self.tt, self.tl = di[ok], tod[ok], li[ok]
        self.treg = pd.Series(tpl["mask_regime"].to_numpy(zero_copy_only=False)[ok]).map(REG).to_numpy()
        self.tlanes = lane_vector(stations[ok], links[ok], station_lanes(pdir)).astype(np.float32)
        self.tpl_keys = pd.DataFrame({"timestamp": tpl["timestamp"].to_numpy(zero_copy_only=False)[ok], "link_id": links[ok].to_numpy()})
        fd = pd.read_csv(pdir / "network" / "fd_parameters.csv").set_index("link_id")
        topo = pd.read_csv(pdir / "network" / "lwr_mainline_topology.csv").set_index("mainline_link_id").loc[self.order]
        self.geo = {c: fd[c].reindex(self.order).to_numpy(dtype=float) for c in GEO}
        self.vf = fd.free_speed_kmh.reindex(self.order).fillna(117.0).to_numpy()
        self.vclip = np.minimum(fd.free_speed_kmh.reindex(self.order).fillna(120.75).to_numpy() + 4.5, 120.75)
        self.geo["pos_frac"] = self.pos / max(self.pos[-1], 1e-9)
        self.geo["has_sensor"] = topo.has_sensor.to_numpy(dtype=float)
        self.ramp_on, self.ramp_off = self.ramps(pdir, topo)
        self.day_rows = pd.Series(np.arange(len(self.td))).groupby(self.td).indices
        tick("load", s)

    def ramps(self, pdir, topo):
        files = sorted((pdir / "train" / "ramp_states").glob("**/*.parquet"))
        rid = sorted(set(";".join(topo.on_ramp_link_ids.fillna("").astype(str).tolist() + topo.off_ramp_link_ids.fillna("").astype(str).tolist()).split(";")) - {""})
        rix = pd.Index(rid)
        R = np.full((self.D, T, len(rid)), np.nan, dtype=np.float32)
        for f in files:
            t = pq.ParquetFile(f).read(columns=["date", "timestamp", "ramp_link_id", "flow_vph", "pct_observed", "is_score_eligible"])
            di, tod = grid_index(t, self.days)
            ri = rix.get_indexer(t["ramp_link_id"].to_numpy(zero_copy_only=False))
            ok = (ri >= 0) & (di >= 0) & (t["pct_observed"].to_numpy() >= 75) & (t["is_score_eligible"].to_numpy() == 1)
            R[di[ok], tod[ok], ri[ok]] = t["flow_vph"].to_numpy(zero_copy_only=False)[ok]
        out = []
        for col in ["on_ramp_link_ids", "off_ramp_link_ids"]:
            arr = np.zeros((self.D, T, self.L), dtype=np.float32)
            for j, ids in enumerate(topo[col].fillna("").astype(str)):
                k = [rix.get_loc(x) for x in ids.split(";") if x in rix]
                if k:
                    arr[:, :, j] = R[:, :, k].sum(axis=2) if len(k) > 1 else R[:, :, k[0]]
            out.append(arr)
        return out

    def profiles(self, keep):
        out = {}
        dow = self.dow[keep]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for c in VARS:
                x = self.obs[c][keep]
                arr = np.full((7, T, self.L), np.nan)
                for d in range(7):
                    s = x[dow == d]
                    if len(s) == 0:
                        continue
                    n = (~np.isnan(s)).sum(axis=0)
                    arr[d] = np.where(n > 0, np.nansum(s, axis=0) / np.maximum(n, 1), np.nan)
                tm = np.nanmean(arr, axis=0)
                for d in range(7):
                    m = np.isnan(arr[d])
                    arr[d][m] = tm[m]
                out[c] = arr
        return out


def components(obs, prof, pos):
    R = obs - prof
    et, dt = interp_axis(R, np.arange(T, dtype=float))
    es, ds = interp_axis(R.T, pos)
    es, ds = es.T, ds.T
    wt = 1.0 / (dt + A["t0"]) ** A["p"]
    ws = 1.0 / (ds / A["s_km"] + A["t0"]) ** A["p"]
    wt[~np.isfinite(dt)] = 0
    ws[~np.isfinite(ds)] = 0
    den = wt + ws
    res = np.divide(wt * np.nan_to_num(et) + ws * np.nan_to_num(es), den, out=np.zeros_like(den), where=den > 0)
    out = prof + res
    keep = ~np.isnan(obs)
    out[keep] = obs[keep]
    dt = np.where(np.isfinite(dt), dt, np.nan)
    ds = np.where(np.isfinite(ds), ds, np.nan)
    return out, et, es, dt, ds


def reconcile(v0, q0, k0, vf, thr=0.6):
    ratio = np.clip(k0 * v0 / np.clip(q0, 1.0, None), 0.2, 5.0)
    ratio = np.where(v0 < thr * vf, ratio, 1.0)
    return v0 / ratio


def gap_info(miss):
    idx = np.arange(T)[:, None]
    last = np.maximum.accumulate(np.where(~miss, idx, -1), axis=0)
    nxt = np.flip(np.minimum.accumulate(np.flip(np.where(~miss, idx, T), 0), axis=0), 0)
    fs = np.where(miss, idx - last, 0)
    fe = np.where(miss, nxt - idx, 0)
    return fs + fe - 1, fs, fe


def day_features(P, di, rows, prof):
    t, l = P.tt[rows], P.tl[rows]
    dw = P.dow[di]
    obs = {c: P.obs[c][di] for c in VARS}
    rec, comp = {}, {}
    for c in VARS:
        pr = prof[c][dw]
        rec[c], et, es, dt, ds = components(obs[c], pr, P.pos)
        comp[c] = (et[t, l], es[t, l], pr[t, l])
        if c == "v":
            dtv, dsv = dt[t, l], ds[t, l]
    v0, q0, k0 = rec["v"][t, l], rec["q"][t, l], rec["k"][t, l]
    vf = P.vf[l]
    v06 = reconcile(v0, q0, k0, vf)
    cols = [v0, q0, k0, v06, (v0 < 0.6 * vf).astype(float), v0 / vf]
    for c in VARS:
        cols += list(comp[c])
    gl, fs, fe = gap_info(np.isnan(obs["v"]))
    cols += [dtv, dsv, gl[t, l], fs[t, l], fe[t, l]]
    for c in VARS:
        pad = np.pad(obs[c], ((2, 2), (0, 0)), constant_values=np.nan)
        for o in [-2, -1, 1, 2]:
            cols.append(pad[t + 2 + o, l])
    for c in VARS:
        pad = np.pad(obs[c], ((0, 0), (3, 3)), constant_values=np.nan)
        for o in [-3, -2, -1, 1, 2, 3]:
            cols.append(pad[t, l + 3 + o])
    for c in ["v", "q"]:
        r = obs[c] / np.where(prof[c][dw] > 0, prof[c][dw], np.nan)
        pad = np.pad(r, ((0, 0), (3, 3)), constant_values=np.nan)
        for o in [-3, -2, -1, 1, 2, 3]:
            cols.append(pad[t, l + 3 + o])
    cols += [t.astype(float), np.full(len(t), float(dw)), P.treg[rows].astype(float), P.ramp_on[di][t, l], P.ramp_off[di][t, l]]
    cols += [P.geo[c][l] for c in GEO + ["pos_frac", "has_sensor"]]
    X = np.column_stack(cols).astype(np.float32)
    return X, np.column_stack([v0, q0, k0, v06]).astype(np.float32)


def build_fold(P, fold, n_sample, seed, cache):
    s = time.time()
    hold = np.isin(P.month, FOLDS[fold])
    prof = P.profiles(~hold)
    rng = np.random.default_rng(seed + fold)
    cand = np.flatnonzero(~hold[P.td])
    pick = np.zeros(len(P.td), dtype=bool)
    pick[rng.choice(cand, size=min(n_sample, len(cand)), replace=False)] = True
    pick |= hold[P.td]
    Xs, Bs, Rs = [], [], []
    for di in range(P.D):
        rows = P.day_rows.get(di)
        if rows is None:
            continue
        rows = rows[pick[rows]]
        if len(rows) == 0:
            continue
        X, B = day_features(P, di, rows, prof)
        Xs.append(X)
        Bs.append(B)
        Rs.append(rows)
    X, B, R = np.concatenate(Xs), np.concatenate(Bs), np.concatenate(Rs)
    tr = ~hold[P.td[R]]
    yv, yq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
    good = tr & np.isfinite(yv) & np.isfinite(yq)
    np.savez(cache / f"{P.name}_f{fold}.npz", Xtr=X[good], Btr=B[good], yv=yv[good], yq=yq[good], Xte=X[~tr], Bte=B[~tr], Rte=R[~tr], vf_te=P.vf[P.tl[R[~tr]]], vc_te=P.vclip[P.tl[R[~tr]]])
    tick("features", s)
    return int(good.sum()), int((~tr).sum())


PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def train(X, y, threads, rounds, seed):
    p = dict(PARAMS, num_threads=threads, seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    return lgb.train(p, lgb.Dataset(X, y, feature_name=FEATS, free_raw_data=True), rounds)


def finalize(vc, v, q):
    return np.clip(v, 5.0, vc), np.clip(q, 0.0, None)


def variants(exp, d, models):
    B, X, vf, vc = d["Bte"].astype(float), d["Xte"], d["vf_te"], d["vc_te"]
    v0, q0, k0, v06 = B.T
    if exp == "base":
        return {"v06": finalize(vc, v06, q0), "none": finalize(vc, v0, q0)}
    if exp == "A":
        return {"A": finalize(vc, v06 + models["dv"].predict(X), q0 + models["dq"].predict(X))}
    kp = np.clip(k0 + models["dk"].predict(X), 0.5, None)
    ratio = np.clip(kp * v0 / np.clip(q0, 1.0, None), 0.2, 5.0)
    return {
        "B_cong": finalize(vc, np.where(v0 < 0.6 * vf, v0 / ratio, v0), q0),
        "B_all": finalize(vc, v0 / ratio, q0),
    }


def fit_fold(panels, fold, exps, args, cache, imp):
    s = time.time()
    need = [e for e in exps if e != "base"]
    models = {}
    if need:
        parts = [np.load(cache / f"{p}_f{fold}.npz") for p in panels]
        X = np.concatenate([d["Xtr"] for d in parts])
        Bt = np.concatenate([d["Btr"] for d in parts]).astype(float)
        yv = np.concatenate([d["yv"] for d in parts]).astype(float)
        yq = np.concatenate([d["yq"] for d in parts]).astype(float)
        del parts
        tasks = []
        if "A" in need:
            tasks += [("A", "dv", yv - Bt[:, 3]), ("A", "dq", yq - Bt[:, 1])]
        if "B" in need:
            tasks += [("B", "dk", yq / np.clip(yv, 1.0, None) - Bt[:, 2])]
        for e, name, y in tasks:
            m = train(X, y, args.threads, args.rounds, args.seed)
            models.setdefault(e, {})[name] = m
            g = m.feature_importance("gain")
            imp.append(pd.DataFrame({"exp": e, "target": name, "fold": fold, "feature": FEATS, "gain": g, "gain_frac": g / max(g.sum(), 1e-12)}))
        del X
        gc.collect()
    tick("train", s)
    return models


class FoldProxy(Proxy):
    def __init__(self, rel, panel, P):
        super().__init__(rel, panel)
        f = self.f
        self.di = P.days.get_indexer(f.date)
        self.ti = (f.ts.dt.hour * 12 + f.ts.dt.minute // 5).to_numpy()
        self.li = pd.Index(P.order).get_indexer(f.link_id)
        self.tgt = f.is_target.to_numpy()
        self.reg = f.regime.to_numpy()
        self.sp, self.fl = f.speed_kmh.to_numpy(), f.flow_vph.to_numpy()
        self.Ltab = f[["length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]].to_numpy()
        self.hn = f.has_next.to_numpy()
        self.Nt = f.N_true.to_numpy()
        self.keys = f[["timestamp", "link_id"]].copy()
        del self.f
        gc.collect()

    def score_values(self, v, q, day_sel):
        length, lanes, vf, cap, kc, kj = self.Ltab.T
        N = q / np.clip(v, 1.0, None) * length
        Nt = self.Nt
        nxt, nxt_t = np.r_[N[1:], np.nan], np.r_[Nt[1:], np.nan]
        ln = np.clip(lanes, 1, None)
        kl = q / np.clip(v, 1.0, None) / ln
        kcl, kjl, capl = kc / ln, kj / ln, cap / ln
        qfd = np.where(kl <= kcl, vf * kl, capl / np.maximum(kjl - kcl, 1e-9) * np.maximum(kjl - kl, 0))
        ql = q / ln
        out = {}
        for r in ["R1", "R2", "R3"]:
            m = (self.reg == r) & day_sel
            mm = m & self.hn
            e = np.abs((nxt - N) - (nxt_t - Nt))[mm].sum() / np.abs(nxt_t - Nt)[mm].sum()
            sfd = max(0.0, 1 - np.abs(ql[m] - qfd[m]).sum() / np.abs(ql[m]).sum())
            slwr = max(0.0, 1 - min(1.0, e))
            out[r] = {"S_FD": sfd, "E_LWR": e, "S_LWR": slwr, "S_physics": (sfd + 2 * slwr) / 3}
        return out

    def score_dense(self, V, Q, hold_days):
        sel = self.tgt & (self.di >= 0) & hold_days[np.clip(self.di, 0, None)]
        v, q = self.sp.copy(), self.fl.copy()
        v[sel] = V[self.di[sel], self.ti[sel], self.li[sel]]
        q[sel] = Q[self.di[sel], self.ti[sel], self.li[sel]]
        return self.score_values(v, q, (self.di >= 0) & hold_days[np.clip(self.di, 0, None)])


def state_metrics(P, R, v, q):
    d, t, l = P.td[R], P.tt[R], P.tl[R]
    tv, tq = P.tv[d, t, l].astype(float), P.tq[d, t, l].astype(float)
    ln = P.tlanes[R].astype(float)
    reg = P.treg[R]
    usable = np.isfinite(tv) & np.isfinite(tq)
    vp, qp = np.where(np.isfinite(v), v, 0.0), np.where(np.isfinite(q), q, 0.0)
    kp, kt = qp / np.clip(vp, 1.0, None), tq / np.clip(tv, 1.0, None)
    queued = tv <= 0.6 * P.vf[l]
    rows = []
    for r, ri in REG.items():
        m = usable & (reg == ri)
        ev, eq = vp[m] - tv[m], (qp[m] - tq[m]) / ln[m]
        rv, rq = np.sqrt(np.mean(ev ** 2)), np.sqrt(np.mean(eq ** 2))
        row = {"regime": r, "n": int(m.sum()), "n_nan_pred": int((~np.isfinite(v[m]) | ~np.isfinite(q[m])).sum()), "rmse_v": rv, "rmse_q_lane": rq,
               "S_state": 0.54 * max(0.0, 1 - rv / 25) + 0.46 * max(0.0, 1 - rq / 600)}
        for tag, qm in [("q", queued), ("nq", ~queued)]:
            mm = m & qm
            n = int(mm.sum())
            dk = kp[mm] - kt[mm]
            row.update({f"{tag}_n": n,
                        f"{tag}_rmse_v": np.sqrt(np.mean((vp[mm] - tv[mm]) ** 2)) if n else np.nan,
                        f"{tag}_rmse_q_lane": np.sqrt(np.mean(((qp[mm] - tq[mm]) / ln[mm]) ** 2)) if n else np.nan,
                        f"{tag}_mae_k": np.mean(np.abs(dk)) if n else np.nan,
                        f"{tag}_bias_k": np.mean(dk) if n else np.nan,
                        f"{tag}_sum_dk": float(dk.sum()), f"{tag}_sum_kt": float(kt[mm].sum())})
        rows.append(row)
    return rows


def score_panel(rel, P, folds, exps, cache):
    s = time.time()
    px = FoldProxy(rel, P.name, P)
    tick("proxy_load", s)
    s = time.time()
    rows = []
    for fold in folds:
        hold = np.isin(P.month, FOLDS[fold])
        for exp in exps:
            pr = np.load(cache / f"{P.name}_f{fold}_{exp}_pred.npz")
            R = pr["R"]
            for var in [k[:-2] for k in pr.files if k.endswith("_v")]:
                v, q = pr[f"{var}_v"], pr[f"{var}_q"]
                V = np.full((P.D, T, P.L), np.nan)
                Q = np.full((P.D, T, P.L), np.nan)
                V[P.td[R], P.tt[R], P.tl[R]] = v
                Q[P.td[R], P.tt[R], P.tl[R]] = q
                ph = px.score_dense(V, Q, hold)
                for row in state_metrics(P, R, v, q):
                    row.update(ph[row["regime"]])
                    rows.append({"exp": exp, "variant": var, "fold": fold, "panel": P.name, **row})
    del px
    gc.collect()
    tick("score", s)
    return rows


def predict_panel(p, fold, exps, models, cache):
    s = time.time()
    d = np.load(cache / f"{p}_f{fold}.npz")
    for exp in exps:
        out = {"R": d["Rte"]}
        for name, (v, q) in variants(exp, d, models.get(exp, {})).items():
            out[f"{name}_v"], out[f"{name}_q"] = v.astype(np.float32), q.astype(np.float32)
        np.savez(cache / f"{p}_f{fold}_{exp}_pred.npz", **out)
    tick("predict", s)


def summarize(out_dir):
    frames = [pd.read_csv(f) for f in sorted(out_dir.glob("resid_cv_*.csv"))]
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    keys = ["exp", "variant"]
    met = ["S_state", "S_FD", "S_LWR", "S_physics", "rmse_v", "rmse_q_lane", "q_rmse_v", "q_rmse_q_lane", "q_mae_k", "nq_rmse_v", "nq_rmse_q_lane", "nq_mae_k"]
    g = df.groupby(keys)
    s = g[met].mean()
    s["n_cells"] = g.n.sum()
    s["q_bias_k"] = g.q_sum_dk.sum() / g.q_n.sum()
    s["q_mean_kt"] = g.q_sum_kt.sum() / g.q_n.sum()
    s["nq_bias_k"] = g.nq_sum_dk.sum() / g.nq_n.sum()
    s["q_bias_rel"] = s.q_bias_k / s.q_mean_kt
    s["guard"] = np.where(s.q_bias_rel < -0.05, "REJECTED", "ok")
    s["J"] = 0.35 * s.S_state + 0.15 * s.S_physics
    s = s.reset_index()
    s.to_csv(out_dir / "resid_summary.csv", index=False)
    return s


def check_csv(rel, P, path):
    sub = pd.read_csv(path, usecols=["timestamp", "link_id", "speed_kmh", "flow_vph"])
    m = P.tpl_keys.reset_index().merge(sub, on=["timestamp", "link_id"], how="left")
    R = m["index"].to_numpy()
    v, q = m.speed_kmh.to_numpy(), m.flow_vph.to_numpy()
    rows = state_metrics(P, R, v, q)
    print("fast S_state", {r["regime"]: round(r["S_state"], 6) for r in rows}, "mean", round(np.mean([r["S_state"] for r in rows]), 6), flush=True)
    prof = P.profiles(np.ones(P.D, dtype=bool))
    diff = []
    for di in range(P.D):
        rr = P.day_rows.get(di)
        if rr is None:
            continue
        _, B = day_features(P, di, rr, prof)
        vv, qq = finalize(P.vclip[P.tl[rr]], B[:, 3].astype(float), B[:, 1].astype(float))
        diff.append(np.nanmax(np.abs(np.r_[vv - v[rr], qq - q[rr]])))
    print("base replication max abs diff vs csv", float(np.nanmax(diff)), flush=True)
    px = FoldProxy(rel, P.name, P)
    V = np.full((P.D, T, P.L), np.nan)
    Q = np.full((P.D, T, P.L), np.nan)
    V[P.td[R], P.tt[R], P.tl[R]] = v
    Q[P.td[R], P.tt[R], P.tl[R]] = q
    ph = px.score_dense(V, Q, np.ones(P.D, dtype=bool))
    print("dense physics", {r: round(x["S_physics"], 4) for r, x in ph.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--panels", nargs="+")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--exp", nargs="+", default=["base", "A", "B"], choices=["base", "A", "B"])
    ap.add_argument("--sample-per-panel", type=int, default=200000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--check-csv")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rel = Path(args.release_root)
    out_dir, cache = Path(args.out_dir), Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    panels = args.panels or sorted(p.name for p in (rel / "corridors").iterdir())
    t_all = time.time()
    if args.check_csv:
        P = Panel(rel, panels[0])
        check_csv(rel, P, args.check_csv)
        print(f"peak {peak_gb():.2f} GB, {time.time()-t_all:.0f}s")
        return
    for p in panels:
        need = [f for f in args.folds if not (args.reuse_cache and (cache / f"{p}_f{f}.npz").exists())]
        if not need:
            continue
        P = Panel(rel, p)
        for f in need:
            ntr, nte = build_fold(P, f, args.sample_per_panel, args.seed, cache)
            print(f"{p} fold {f}: train {ntr:,} held-out {nte:,} | {STAGES} | peak {peak_gb():.2f} GB", flush=True)
        del P
        gc.collect()
    imp = []
    for f in args.folds:
        models = fit_fold(panels, f, args.exp, args, cache, imp)
        for p in panels:
            predict_panel(p, f, args.exp, models, cache)
        print(f"fold {f} trained+predicted | {STAGES} | peak {peak_gb():.2f} GB", flush=True)
    rows = []
    for p in panels:
        P = Panel(rel, p)
        rows += score_panel(rel, P, args.folds, args.exp, cache)
        del P
        gc.collect()
    df = pd.DataFrame(rows)
    for e in args.exp:
        df[df.exp == e].to_csv(out_dir / f"resid_cv_{e}.csv", index=False)
    if imp:
        pd.concat(imp).to_csv(out_dir / "resid_importance.csv", index=False)
    s = summarize(out_dir)
    pd.set_option("display.width", 250)
    print(s[["exp", "variant", "S_state", "S_FD", "S_LWR", "S_physics", "J", "q_rmse_v", "q_mae_k", "q_bias_rel", "nq_rmse_v", "guard"]].round(4).to_string(index=False))
    print(f"stages {({k: round(v) for k, v in STAGES.items()})} total {time.time()-t_all:.0f}s peak {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
