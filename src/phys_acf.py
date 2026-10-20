import argparse
import gc
import resource
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view

LAGS = list(range(1, 7))
H = len(LAGS)
P = 6
QTY = ["v", "ql", "kl", "N"]
CLS = ["all", "free", "cong"]
MIN_PAIRS = 200
SEED = 7


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def to_slot(ts):
    t = ts if pa.types.is_timestamp(ts.type) else pc.strptime(ts, format="%Y-%m-%dT%H:%M:%SZ", unit="s")
    t = pc.cast(t, pa.timestamp("s", tz="UTC"))
    return pc.divide(pc.cast(t, pa.int64()), 300).to_numpy(zero_copy_only=False).astype(np.int64)


def codes(col, links):
    enc = pc.dictionary_encode(col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col)
    lut = pd.Index(links).get_indexer(enc.dictionary.to_pylist())
    return lut[enc.indices.to_numpy(zero_copy_only=False)]


def load_truth(rel, panel):
    pdir = rel / "corridors" / panel
    fd = pd.read_csv(pdir / "network" / "fd_parameters.csv")
    links = fd.link_id.tolist()
    parts = []
    for f in sorted((pdir / "train" / "mainline_states").glob("**/*.parquet")):
        t = pq.read_table(f, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
        t = t.filter(pc.equal(t["is_score_eligible"], 1))
        parts.append((to_slot(t["timestamp"]), codes(t["link_id"], links),
                      t["speed_kmh"].to_numpy(zero_copy_only=False), t["flow_vph"].to_numpy(zero_copy_only=False)))
    s = np.concatenate([p[0] for p in parts])
    li = np.concatenate([p[1] for p in parts])
    v = np.concatenate([p[2] for p in parts])
    q = np.concatenate([p[3] for p in parts])
    ok = (li >= 0) & np.isfinite(v) & np.isfinite(q)
    s, li, v, q = s[ok], li[ok], v[ok], q[ok]
    base = s.min()
    Tn = int(s.max() - base + 1)
    V = np.full((Tn, len(links)), np.nan, np.float32)
    Q = np.full((Tn, len(links)), np.nan, np.float32)
    V[s - base, li] = v
    Q[s - base, li] = q
    tpl = pcsv.read_csv(rel / "task1" / panel / "train" / "sample_submission_state.csv",
                        convert_options=pcsv.ConvertOptions(include_columns=["timestamp", "link_id"]))
    ts = to_slot(tpl["timestamp"]) - base
    tl = codes(tpl["link_id"], links)
    TG = np.zeros((Tn, len(links)), bool)
    m = (tl >= 0) & (ts >= 0) & (ts < Tn)
    TG[ts[m], tl[m]] = True
    return fd, links, base, V, Q, TG


def quantities(V, Q, fd):
    lanes = np.clip(fd.lanes.to_numpy(np.float32), 1, None)
    L = fd.length_km.to_numpy(np.float32)
    k = Q / np.clip(V, 1.0, None)
    return {"v": V, "ql": Q / lanes, "kl": k / lanes, "N": k * L}


def smooth(X, kind):
    w = 5 if kind == "med5" else 3
    h = w // 2
    pad = np.full((h, X.shape[1]), np.nan, np.float32)
    Xp = np.vstack([pad, X, pad])
    out = np.empty_like(X)
    for j in range(0, X.shape[1], 32):
        sw = sliding_window_view(Xp[:, j:j + 32], w, axis=0)
        out[:, j:j + 32] = np.median(sw, axis=-1) if kind == "med5" else sw.mean(axis=-1)
    return out


def demean(R, M):
    D = np.where(M, R, np.nan)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(D, axis=0)
    return np.where(M, R - mu, 0.0).astype(np.float32)


def acf_stats(R, M):
    D = demean(R, M)
    n0 = M.sum(0).astype(np.float64)
    s0 = (D * D).sum(0, dtype=np.float64)
    sh = np.zeros((H, R.shape[1]))
    nh = np.zeros((H, R.shape[1]))
    for i, h in enumerate(LAGS):
        sh[i] = (D[:-h] * D[h:]).sum(0, dtype=np.float64)
        nh[i] = (M[:-h] & M[h:]).sum(0)
    return {"n0": n0, "s0": s0, "sh": sh, "nh": nh}


def acf_from(st):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (st["sh"] / st["nh"]) / (st["s0"] / st["n0"])


def pooled(st):
    return (st["sh"].sum(1) / st["nh"].sum(1)) / (st["s0"].sum() / st["n0"].sum())


def shuffle(R, M, rng):
    S = np.full_like(R, np.nan)
    for j in range(R.shape[1]):
        idx = np.flatnonzero(M[:, j])
        if len(idx):
            S[idx, j] = rng.permutation(R[idx, j])
    return S


def xcorr0(Ra, Rb, M):
    A = demean(Ra, M)
    B = demean(Rb, M)
    ab, aa, bb = (A * B).sum(0, dtype=np.float64), (A * A).sum(0, dtype=np.float64), (B * B).sum(0, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return ab / np.sqrt(aa * bb), M.sum(0)


def ar_gram(R, Mall, C):
    D = demean(R, Mall)
    T = R.shape[0]
    full = Mall[P:].copy()
    for i in range(1, P + 1):
        full &= Mall[P - i:T - i]
    full &= C[P:]
    G = np.zeros((P + 1, P + 1))
    lag = [D[P - i:T - i] for i in range(P + 1)]
    for i in range(P + 1):
        for j in range(i, P + 1):
            G[i, j] = G[j, i] = (lag[i] * lag[j] * full).sum(dtype=np.float64)
    return G, D, full


def ar_fit(G, p):
    a = np.linalg.solve(G[1:p + 1, 1:p + 1], G[1:p + 1, 0])
    sse = G[0, 0] - 2 * a @ G[1:p + 1, 0] + a @ G[1:p + 1, 1:p + 1] @ a
    return a, 1 - sse / G[0, 0]


def delta_ratio(D, full, a, TG):
    T = D.shape[0]
    pred = np.zeros_like(D[P:])
    for i, ai in enumerate(a, 1):
        pred += ai * D[P - i:T - i]
    e = D[P:] - pred
    d = D[P:]
    pair = full[:-1] & full[1:]
    tg = pair & TG[P:-1]
    de = np.abs(e[1:] - e[:-1])
    dd = np.abs(d[1:] - d[:-1])
    return [float((de * pair).sum(dtype=np.float64)), float((dd * pair).sum(dtype=np.float64)),
            float((de * tg).sum(dtype=np.float64)), float((dd * tg).sum(dtype=np.float64))]


def link_rows(panel, links, filt, qty, cls, st, shuf, white):
    a = acf_from(st)
    sa = acf_from(shuf) if shuf is not None else np.full_like(a, np.nan)
    rows = []
    for i, h in enumerate(LAGS):
        rows.append(pd.DataFrame({"panel": panel, "link_id": links, "filter": filt, "qty": qty, "cls": cls, "lag": h,
                                  "acf": a[i], "n": st["nh"][i], "acf_shuf": sa[i],
                                  "acf_white_pooled": white[i] if white is not None else np.nan,
                                  "acf_pooled": pooled(st)[i]}))
    return rows


def classes(V, fd):
    vf = fd.free_speed_kmh.to_numpy(np.float32)
    fin = np.isfinite(V)
    cong = fin & (V <= 0.6 * vf)
    return {"all": fin, "free": fin & ~cong, "cong": cong}


def analyse(rel, panel, pred, acc, rng):
    t0 = time.time()
    fd, links, base, V, Q, TG = load_truth(rel, panel)
    X = quantities(V, Q, fd)
    C = classes(V, fd)
    rows, xc = [], []
    for filt in ["med5", "mean3"]:
        Z = np.where(np.isfinite(V), rng.standard_normal(V.shape).astype(np.float32), np.nan)
        Wr = Z - smooth(Z, filt)
        del Z
        Wvalid = np.isfinite(Wr)
        white = {c: pooled(acf_stats(Wr, Wvalid & C[c])) for c in CLS}
        for c in CLS:
            Gw, Dw, fw = ar_gram(Wr, Wvalid, C[c])
            acc["gram"].setdefault((filt, "white", c), np.zeros((P + 1, P + 1)))
            acc["gram"][(filt, "white", c)] += Gw
            if c == "all":
                aw, _ = ar_fit(Gw, P)
                acc["delta"].setdefault((filt, "white"), np.zeros(4))
                acc["delta"][(filt, "white")] += delta_ratio(Dw, fw, aw, TG)
            del Dw, fw
        del Wr, Wvalid
        res = {}
        for qn in QTY:
            R = X[qn] - smooth(X[qn], filt)
            Mv = np.isfinite(R)
            if qn != "N":
                res[qn] = R
            for c in CLS:
                M = Mv & C[c]
                st = acf_stats(R, M)
                sh = acf_stats(shuffle(R, M, rng), M) if c == "all" or filt == "med5" else None
                rows += link_rows(panel, links, filt, qn, c, st, sh, white[c])
                G, D, full = ar_gram(R, Mv, C[c])
                acc["gram"].setdefault((filt, qn, c), np.zeros((P + 1, P + 1)))
                acc["gram"][(filt, qn, c)] += G
                if c == "all":
                    a, _ = ar_fit(G, P)
                    acc["delta"].setdefault((filt, qn), np.zeros(4))
                    acc["delta"][(filt, qn)] += delta_ratio(D, full, a, TG)
                del D, full
        for c in CLS:
            M = np.isfinite(res["v"]) & np.isfinite(res["ql"]) & C[c]
            for pair in [("v", "ql"), ("v", "kl"), ("ql", "kl")]:
                r, n = xcorr0(res[pair[0]], res[pair[1]], M)
                xc.append(pd.DataFrame({"panel": panel, "link_id": links, "filter": filt, "pair": "-".join(pair), "cls": c, "r": r, "n": n}))
        del res
        gc.collect()
    if pred is not None:
        ts = to_slot(pred["timestamp"]) - base
        tl = codes(pred["link_id"], links)
        m = (tl >= 0) & (ts >= 0) & (ts < V.shape[0])
        PV = np.full(V.shape, np.nan, np.float32)
        PQ = np.full(V.shape, np.nan, np.float32)
        PV[ts[m], tl[m]] = pred["speed_kmh"].to_numpy(zero_copy_only=False)[m]
        PQ[ts[m], tl[m]] = pred["flow_vph"].to_numpy(zero_copy_only=False)[m]
        PX = quantities(PV, PQ, fd)
        del PV, PQ
        for qn in QTY:
            R = np.where(TG, X[qn] - PX[qn], np.nan)
            Mv = np.isfinite(R)
            for c in CLS:
                M = Mv & C[c]
                st = acf_stats(R, M)
                sh = acf_stats(shuffle(R, M, rng), M) if c == "all" else None
                rows += link_rows(panel, links, "interp_v06", qn, c, st, sh, None)
        del PX
    print(panel, "slots", V.shape, "elig", int(np.isfinite(V).sum()), "targets", int(TG.sum()),
          f"{time.time() - t0:.0f}s peak {peak_gb():.2f} GB", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(xc, ignore_index=True)


def stream_pred(path):
    rd = pcsv.open_csv(path, read_options=pcsv.ReadOptions(block_size=64 << 20),
                       convert_options=pcsv.ConvertOptions(include_columns=["panel", "timestamp", "link_id", "speed_kmh", "flow_vph"],
                                                           column_types={"panel": pa.string(), "timestamp": pa.string(), "link_id": pa.string(),
                                                                         "speed_kmh": pa.float32(), "flow_vph": pa.float32()}))
    cur, buf = None, []
    for b in rd:
        pans = b.column("panel")
        for p in pc.unique(pans).to_pylist():
            sub = b.filter(pc.equal(pans, p)).drop_columns(["panel"])
            if p != cur:
                if cur is not None:
                    yield cur, pa.Table.from_batches(buf)
                cur, buf = p, []
            buf.append(sub)
    if cur is not None:
        yield cur, pa.Table.from_batches(buf)


def summarise(acf):
    sig = 1.96 / np.sqrt(acf.n)
    acf = acf.assign(sig=(acf.acf.abs() > sig), sig_white=((acf.acf - acf.acf_white_pooled).abs() > sig))
    keys = ["filter", "qty", "cls", "lag"]
    by = acf.groupby(["panel"] + keys).agg(n_links=("acf", "size"), acf_med=("acf", "median"), acf_q25=("acf", lambda x: x.quantile(0.25)),
                                           acf_q75=("acf", lambda x: x.quantile(0.75)), acf_pooled=("acf_pooled", "first"),
                                           acf_white=("acf_white_pooled", "first"), shuf_med=("acf_shuf", "median"),
                                           shuf_absq95=("acf_shuf", lambda x: x.abs().quantile(0.95)), n_med=("n", "median"),
                                           frac_sig=("sig", "mean"), frac_sig_vs_white=("sig_white", "mean")).reset_index()
    ov = acf.groupby(keys).agg(n_links=("acf", "size"), acf_med=("acf", "median"), acf_q25=("acf", lambda x: x.quantile(0.25)),
                               acf_q75=("acf", lambda x: x.quantile(0.75)), acf_white_med=("acf_white_pooled", "median"),
                               shuf_absq95=("acf_shuf", lambda x: x.abs().quantile(0.95)), n_total=("n", "sum"),
                               frac_sig=("sig", "mean"), frac_sig_vs_white=("sig_white", "mean")).reset_index()
    ov["excess_vs_white"] = ov.acf_med - ov.acf_white_med
    return by, ov


def plot(ov, path):
    fig, ax = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
    col = {"all": "#444444", "free": "#2b7bba", "cong": "#d1495b"}
    for i, filt in enumerate(["med5", "mean3", "interp_v06"]):
        for j, qn in enumerate(QTY):
            a = ax[i, j]
            for c in CLS:
                d = ov[(ov["filter"] == filt) & (ov.qty == qn) & (ov.cls == c)].sort_values("lag")
                a.plot(d.lag, d.acf_med, "o-", color=col[c], label=c)
                a.fill_between(d.lag, d.acf_q25, d.acf_q75, color=col[c], alpha=0.15)
                if filt != "interp_v06":
                    a.plot(d.lag, d.acf_white_med, "--", color=col[c], lw=1)
            a.axhline(0, color="k", lw=0.5)
            a.set_title(f"{filt} | {qn}")
            if i == 2:
                a.set_xlabel("lag (5 dk)")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle("Kalıntı ACF (link medyanı, IQR; kesikli = beyaz gürültü + aynı filtre)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--pred", default="submissions/task1_interp/state_train_v06.csv")
    ap.add_argument("--out", default="reports/phys")
    ap.add_argument("--panels", nargs="*")
    ap.add_argument("--no-pred", action="store_true")
    args = ap.parse_args()
    rel, out = Path(args.release_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    acc = {"gram": {}, "delta": {}}
    acfs, xcs = [], []
    t0 = time.time()
    if args.no_pred:
        it = ((p, None) for p in sorted(x.name for x in (rel / "corridors").iterdir()))
    else:
        it = stream_pred(args.pred)
    for panel, pred in it:
        if args.panels and panel not in args.panels:
            continue
        a, x = analyse(rel, panel, pred, acc, rng)
        acfs.append(a)
        xcs.append(x)
        del pred
        gc.collect()
    acf = pd.concat(acfs, ignore_index=True)
    xc = pd.concat(xcs, ignore_index=True)
    by, ov = summarise(acf)
    by.to_csv(out / "acf_by_corridor.csv", index=False)
    ov.to_csv(out / "acf_summary.csv", index=False)
    xs = xc.dropna().groupby(["filter", "pair", "cls"]).r.describe(percentiles=[0.25, 0.5, 0.75])[["count", "25%", "50%", "75%"]].reset_index()
    xs.to_csv(out / "acf_xcorr0.csv", index=False)
    ar = []
    for (filt, qn, c), G in acc["gram"].items():
        for p in range(1, P + 1):
            _, r2 = ar_fit(G, p)
            ar.append({"filter": filt, "qty": qn, "cls": c, "p": p, "R2": r2, "n": G[0, 0]})
    ar = pd.DataFrame(ar)
    ar.to_csv(out / "acf_ar.csv", index=False)
    dl = pd.DataFrame([{"filter": k[0], "qty": k[1], "ratio_all": v[0] / v[1], "ratio_target": v[2] / v[3]} for k, v in acc["delta"].items()])
    dl.to_csv(out / "acf_delta.csv", index=False)
    plot(ov, out / "acf_plot.png")
    pd.set_option("display.width", 200)
    print(ov[(ov.cls == "all") & (ov.lag <= 3)][["filter", "qty", "lag", "acf_med", "acf_white_med", "excess_vs_white", "frac_sig", "frac_sig_vs_white"]].round(3).to_string(index=False))
    print(ov[(ov.qty == "N") & (ov.lag <= 2)][["filter", "cls", "lag", "acf_med", "acf_white_med", "excess_vs_white"]].round(3).to_string(index=False))
    print(xs.round(3).to_string(index=False))
    print(ar[ar.p == P].pivot_table(index=["filter", "cls"], columns="qty", values="R2").round(3).to_string())
    print(dl.round(3).to_string(index=False))
    print(f"total {time.time() - t0:.0f}s peak {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
