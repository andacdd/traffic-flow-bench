import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LB = {"queue_onset": {"pers": 0.0, "v3": 0.714, "v4": 0.757},
      "queue_ongoing": {"pers": 0.603, "v3": 0.723, "v4": 0.790, "final": 0.742, "sel_ramp": 0.772}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--truth", default="obs", choices=["obs", "fill"])
    ap.add_argument("--rare-cut", type=float, default=0.5)
    a = ap.parse_args()
    q = Path(a.qdir)
    rr = pd.read_csv(q / "dev_rarity.csv").set_index("window_id")
    c = pd.read_csv(q / "cv_results.csv").set_index("window_id")
    fcv = pd.read_csv(q / "final_cv_results.csv").set_index("window_id")
    df = pd.DataFrame({"condition": c.condition, "pers": c[f"pers_{a.truth}"], "v3": c[f"v3_{a.truth}"], "v4": c[f"v4_{a.truth}"],
                       "final": fcv[f"final_{a.truth}"]})
    s = pd.read_csv(q / "sel" / "sel_cv_ramp.csv")
    s = s[(s.cond == "ongoing") & (s.model == "sel_ramp") & (s.rule == "thr0.45")]
    df["sel_ramp"] = s.assign(w=s.win.str[2:]).set_index("w").iou
    s2 = pd.read_csv(q / "sel" / "sel_cv_abl.csv")
    s2 = s2[(s2.cond == "ongoing") & (s2.model == "v4") & (s2.rule == "thr0.35")]
    df["v4_selcv"] = s2.assign(w=s2.win.str[2:]).set_index("w").iou
    df["rare"] = rr.rare_share.reindex(df.index) > a.rare_cut
    df.loc[df.condition == "queue_onset", "final"] = np.nan
    rows = []
    for cond, g in df.groupby("condition"):
        for rho in [None, 0.0, 0.1, 0.2, 0.28, 0.4, 0.5]:
            r = {"condition": cond, "rho": "plain" if rho is None else rho, "n_rare": int(g.rare.sum()), "n": len(g)}
            for m in ["pers", "v3", "v4", "final", "sel_ramp", "v4_selcv"]:
                x = g[[m, "rare"]].dropna()
                if x.empty:
                    continue
                r[m] = x[m].mean() if rho is None else (1 - rho) * x[~x.rare][m].mean() + rho * x[x.rare][m].mean()
            rows.append(r)
    t = pd.DataFrame(rows)
    print(t.round(4).to_string(index=False))
    print("\nLB hedefleri:", LB)
    pairs = [("queue_onset", "v3", "v4"), ("queue_ongoing", "v3", "v4"), ("queue_ongoing", "v4", "final"), ("queue_ongoing", "v4", "sel_ramp"),
             ("queue_ongoing", "final", "sel_ramp"), ("queue_ongoing", "v3", "final")]
    out = []
    for cond, a_, b_ in pairs:
        lb = LB[cond][b_] - LB[cond][a_]
        for _, r in t[t.condition == cond].iterrows():
            ref = "v4_selcv" if "sel_ramp" in (a_, b_) and "v4" in (a_, b_) else None
            av = r[ref] if ref and a_ == "v4" else r[a_]
            bv = r[ref] if ref and b_ == "v4" else r[b_]
            d = bv - av
            out.append({"pair": f"{cond[6:]}:{a_}->{b_}", "rho": r.rho, "dev_delta": round(d, 4), "lb_delta": round(lb, 4), "sign_ok": np.sign(d) == np.sign(lb)})
    o = pd.DataFrame(out)
    print(o.pivot(index="pair", columns="rho", values="dev_delta").to_string())
    print(o.groupby("rho").sign_ok.agg(["sum", "count"]).to_string())
    o.to_csv(q / "dev_backtest.csv", index=False)


if __name__ == "__main__":
    main()
