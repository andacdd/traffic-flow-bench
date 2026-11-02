import sys
sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from task2_predict_sel import eiou_select, iou

d = pd.read_parquet("reports/queue/onset_struct/onset_dump.parquet")


def fill1(s):
    s = s.copy()
    s[1:-1] |= s[:-2] & s[2:]
    return s


def nbp(p, a):
    q = p.copy()
    q[1:-1] = np.maximum(p[1:-1], a * np.minimum(p[:-2], p[2:]))
    return q


def grow(p, s, thr):
    s = s.copy()
    on = np.where(s)[0]
    for c in on:
        for n in (c - 1, c + 1):
            if 0 <= n < len(p) and p[n] >= thr:
                s[n] = True
    return s


res, dec = [], []
for win, g in d.groupby("win", sort=False):
    p, y = g.p.to_numpy(), g.y.to_numpy()
    yb = np.nan_to_num(y) > 0
    s = eiou_select(p).astype(bool)
    var = {"B": s, "base_model": eiou_select(g.p_base.to_numpy()).astype(bool), "fill1": fill1(s),
           "nb05": eiou_select(nbp(p, 0.5)).astype(bool), "nb08": eiou_select(nbp(p, 0.8)).astype(bool),
           "grow20": grow(p, s, 0.2), "grow10": grow(p, s, 0.1)}
    for k, v in var.items():
        res.append({"wk": g.wk.iloc[0], "win": win, "pid": g.pid.iloc[0], "variant": k, "iou": iou(v.astype(int), y)})
    tp = (s & yb).sum()
    dec.append({"k_true": int(yb.sum()), "k_pred": int(s.sum()), "tp": int(tp), "miss_all": tp == 0 and yb.sum() > 0,
                "near": int(((np.convolve(s, [1, 1, 1], "same") > 0) & yb & ~s).sum()), "rare": float(np.mean(g.freq.to_numpy()[yb] < 0.02)) if yb.any() else np.nan})
r = pd.DataFrame(res)
t = r.pivot_table(index="variant", columns="wk", values="iou", aggfunc="mean")
t["all"] = r.groupby("variant").iou.mean()
print(t.sub(t.loc["B"]).assign(B=t.loc["B", "all"]).round(4).to_string())
D = pd.DataFrame(dec)
print(D.describe().round(2).to_string())
print("hepsini kaçır", D.miss_all.mean().round(3), "nadir pencere payı", (D.rare > 0.5).mean().round(3))
print("FN toplam", int((D.k_true - D.tp).sum()), "bunlardan seçili hücreye komşu", int(D.near.sum()), "FP toplam", int((D.k_pred - D.tp).sum()))
