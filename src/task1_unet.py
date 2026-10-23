import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task1_resid as tr
from task1_interp import T

VN, QN, RN = 100.0, 1500.0, 1500.0
SCORE_PANELS = ["D12_I5_N", "D7_I405_N", "D7_I210_W"]


def build_panel(rel, p, fold, out):
    P = tr.Panel(rel, p)
    hold = np.isin(P.month, tr.FOLDS[fold])
    prof = P.profiles(~hold)
    reg = np.zeros((P.D, T, P.L), dtype=np.uint8)
    reg[P.td, P.tt, P.tl] = P.treg
    lanes = np.nan_to_num(P.geo["lanes"], nan=np.nanmedian(P.geo["lanes"]))
    lanes = np.clip(lanes, 1, None)
    geo = np.stack([lanes / 5, np.nan_to_num(P.geo["free_speed_kmh"], nan=110) / 120, P.geo["pos_frac"],
                    P.geo["has_sensor"], np.nan_to_num(P.geo["length_km"], nan=0.5)]).astype(np.float32)
    np.savez(out / f"{p}_f{fold}.npz",
             ov=P.obs["v"].astype(np.float16), oq=P.obs["q"].astype(np.float16),
             tv=P.tv.astype(np.float16), tq=P.tq.astype(np.float16), reg=reg,
             pv=prof["v"].astype(np.float16), pq=prof["q"].astype(np.float16),
             ron=np.nan_to_num(P.ramp_on).astype(np.float16),
             roff=np.nan_to_num(P.ramp_off).astype(np.float16),
             geo=geo, lanes=lanes.astype(np.float32), dow=P.dow, hold=hold, vclip=P.vclip.astype(np.float32))
    del P
    gc.collect()


class Grid:
    def __init__(self, path):
        d = np.load(path)
        self.__dict__.update({k: d[k] for k in d.files})
        self.D, _, self.L = self.ov.shape

    def inputs(self, di, ov, oq, reg):
        dw = self.dow[di]
        ln = self.lanes[None, :]
        obs = ~np.isnan(ov)
        pv = self.pv[dw].astype(np.float32)
        pq = self.pq[dw].astype(np.float32) / ln
        tod = np.arange(T, dtype=np.float32)[:, None] / T * 2 * np.pi
        z = np.zeros((T, self.L), dtype=np.float32)
        ch = [np.where(obs, ov, 0) / VN, np.where(obs, oq / ln, 0) / QN, obs.astype(np.float32),
              (reg == 1).astype(np.float32), (reg == 2).astype(np.float32), (reg == 3).astype(np.float32),
              np.nan_to_num(pv) / VN, np.nan_to_num(pq) / QN,
              self.ron[di].astype(np.float32) / RN, self.roff[di].astype(np.float32) / RN,
              z + np.sin(tod), z + np.cos(tod), z + float(dw >= 5)]
        ch += [z + g[None, :] for g in self.geo]
        base = np.stack([np.nan_to_num(pv, nan=100.0) / VN, np.nan_to_num(pq) / QN])
        return np.stack(ch).astype(np.float32), base.astype(np.float32)


NCH = 18


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class UNet(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = block(NCH, c), block(c, 2 * c), block(2 * c, 4 * c), block(4 * c, 8 * c)
        self.u3, self.u2, self.u1 = nn.ConvTranspose2d(8 * c, 4 * c, 2, 2), nn.ConvTranspose2d(4 * c, 2 * c, 2, 2), nn.ConvTranspose2d(2 * c, c, 2, 2)
        self.d3, self.d2, self.d1 = block(8 * c, 4 * c), block(4 * c, 2 * c), block(2 * c, c)
        self.head = nn.Conv2d(c, 2, 1)

    def forward(self, x, base):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return base + self.head(d1)


def pad_l(a, L):
    return np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, L - a.shape[-1])])


def sample(g, rng, crop, aug):
    di = rng.choice(np.flatnonzero(~g.hold))
    ov, oq, reg = g.ov[di].astype(np.float32), g.oq[di].astype(np.float32), g.reg[di].copy()
    tv, tq = g.tv[di].astype(np.float32), g.tq[di].astype(np.float32)
    if rng.random() < aug:
        dj = rng.integers(g.D)
        extra = (g.reg[dj] > 0) & (reg == 0) & ~np.isnan(ov)
        ov, oq = ov.copy(), oq.copy()
        ov[extra] = np.nan
        oq[extra] = np.nan
        reg[extra] = g.reg[dj][extra]
    x, base = g.inputs(di, ov, oq, reg)
    ln = g.lanes[None, :]
    y = np.stack([np.nan_to_num(tv) / VN, np.nan_to_num(tq / ln) / QN])
    w = ((reg > 0) & np.isfinite(tv) & np.isfinite(tq)).astype(np.float32)
    L = g.L
    if L >= crop:
        s = rng.integers(L - crop + 1)
        sl = slice(s, s + crop)
        return x[..., sl], base[..., sl], y[..., sl], w[..., sl]
    return pad_l(x, crop), pad_l(base, crop), pad_l(y, crop), pad_l(w, crop)


def loss_fn(pred, y, w):
    ev = (pred[:, 0] - y[:, 0]) * VN / 25
    eq = (pred[:, 1] - y[:, 1]) * QN / 600
    return ((0.54 * ev ** 2 + 0.46 * eq ** 2) * w).sum() / w.sum().clamp(min=1)


def train(grids, a, dev, out):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    m = UNet(a.width).to(dev)
    opt = torch.optim.AdamW(m.parameters(), a.lr, weight_decay=1e-4)
    total = a.epochs * a.steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.1)
    sizes = np.array([g.L for g in grids], dtype=float)
    pw = sizes / sizes.sum()
    log = []
    for ep in range(a.epochs):
        s, tot = time.time(), 0.0
        m.train()
        for _ in range(a.steps):
            bs = [sample(grids[rng.choice(len(grids), p=pw)], rng, a.crop, a.aug) for _ in range(a.batch)]
            x, base, y, w = (torch.from_numpy(np.stack(z)).to(dev) for z in zip(*bs))
            opt.zero_grad()
            l = loss_fn(m(x, base), y, w)
            l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += l.item()
        log.append({"epoch": ep, "loss": tot / a.steps, "sec": time.time() - s})
        print(log[-1], flush=True)
    torch.save(m.state_dict(), out / f"unet_f{a.fold}{a.tag}.pt")
    pd.DataFrame(log).to_csv(out / f"unet_f{a.fold}{a.tag}_log.csv", index=False)
    return m


@torch.no_grad()
def predict_grid(m, g, days, dev):
    m.eval()
    L8 = int(np.ceil(g.L / 8) * 8)
    V = np.full((g.D, T, g.L), np.nan, dtype=np.float32)
    Q = np.full((g.D, T, g.L), np.nan, dtype=np.float32)
    for di in days:
        x, base = g.inputs(di, g.ov[di].astype(np.float32), g.oq[di].astype(np.float32), g.reg[di])
        p = m(torch.from_numpy(pad_l(x, L8)[None]).to(dev), torch.from_numpy(pad_l(base, L8)[None]).to(dev))[0].cpu().numpy()[:, :, :g.L]
        V[di] = p[0] * VN
        Q[di] = p[1] * QN * g.lanes[None, :]
    return V, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", default="data/kaggle_public")
    ap.add_argument("--grid-dir", default="cache/task1_unet")
    ap.add_argument("--le-cache", default="cache/task1_resid2")
    ap.add_argument("--out", default="reports/phys/unet_f1")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--stages", nargs="+", default=["build", "train", "predict", "score"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=64)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--aug", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    rel, gd, lec, out = Path(a.release_root), Path(a.grid_dir), Path(a.le_cache), Path(a.out)
    gd.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if "build" in a.stages:
        for p in panels:
            if not (gd / f"{p}_f{a.fold}.npz").exists():
                s = time.time()
                build_panel(rel, p, a.fold, gd)
                print(p, f"built {time.time()-s:.0f}s", flush=True)
    grids = {p: Grid(gd / f"{p}_f{a.fold}.npz") for p in panels} if {"train", "predict"} & set(a.stages) else {}
    if "train" in a.stages:
        m = train(list(grids.values()), a, dev, out)
    elif "predict" in a.stages:
        m = UNet(a.width).to(dev)
        m.load_state_dict(torch.load(out / f"unet_f{a.fold}{a.tag}.pt", map_location=dev))
    if "predict" in a.stages:
        for p in SCORE_PANELS:
            s = time.time()
            g = grids[p]
            P = tr.Panel(rel, p)
            V, Q = predict_grid(m, g, np.flatnonzero(g.hold), dev)
            le = np.load(lec / f"{p}_f{a.fold}_R2C_pred.npz")
            R = le["R"]
            v, q = tr.finalize(g.vclip[P.tl[R]], V[P.td[R], P.tt[R], P.tl[R]], Q[P.td[R], P.tt[R], P.tl[R]])
            o = {"R": R, "LE_v": le["LE_v"], "LE_q": le["LE_q"], "LEC50_v": le["LEC50_v"], "LEC50_q": le["LEC50_q"],
                 "UN_v": v.astype(np.float32), "UN_q": q.astype(np.float32)}
            o["LEUN50_v"], o["LEUN50_q"] = 0.5 * le["LE_v"] + 0.5 * o["UN_v"], 0.5 * le["LE_q"] + 0.5 * o["UN_q"]
            o["LECUN_v"] = (le["LE_v"] + le["C_v"] + o["UN_v"]) / 3
            o["LECUN_q"] = (le["LE_q"] + le["C_q"] + o["UN_q"]) / 3
            np.savez(lec / f"{p}_f{a.fold}_UN{a.tag}_pred.npz", **o)
            tv, tq = P.tv[P.td[R], P.tt[R], P.tl[R]], P.tq[P.td[R], P.tt[R], P.tl[R]]
            ok = np.isfinite(tv) & np.isfinite(tq)
            cor = {"panel": p, "tag": a.tag,
                   "corr_err_v_LE_UN": float(np.corrcoef(le["LE_v"][ok] - tv[ok], v[ok] - tv[ok])[0, 1]),
                   "corr_err_q_LE_UN": float(np.corrcoef(le["LE_q"][ok] - tq[ok], q[ok] - tq[ok])[0, 1]),
                   "corr_err_v_LE_C": float(np.corrcoef(le["LE_v"][ok] - tv[ok], le["C_v"][ok] - tv[ok])[0, 1]),
                   "corr_err_q_LE_C": float(np.corrcoef(le["LE_q"][ok] - tq[ok], le["C_q"][ok] - tq[ok])[0, 1])}
            print(cor, f"{time.time()-s:.0f}s", flush=True)
            with open(out / "corr.jsonl", "a") as f:
                f.write(json.dumps(cor) + "\n")
            del P
            gc.collect()
    if "score" in a.stages:
        rows = []
        for p in SCORE_PANELS:
            P = tr.Panel(rel, p)
            rows += tr.score_panel(rel, P, [a.fold], ["UN"], lec)
            del P
            gc.collect()
        df = pd.DataFrame(rows)
        df.to_csv(out / "unet_cv.csv", index=False)
        g = df.groupby("variant")[["S_state", "S_FD", "S_LWR", "S_physics"]].mean()
        g["J"] = 0.35 * g.S_state + 0.15 * g.S_physics
        g["dJ"] = g.J - g.loc["LE", "J"]
        u = df.groupby(["variant", "panel"])[["S_state", "S_physics"]].mean()
        u["J"] = 0.35 * u.S_state + 0.15 * u.S_physics
        print(g.round(5).to_string(), flush=True)
        print(u.J.unstack(0).round(5).to_string(), flush=True)


if __name__ == "__main__":
    main()
