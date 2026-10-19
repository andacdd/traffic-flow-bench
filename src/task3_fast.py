import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent / "reference" / "trafficflowbench-public"
sys.path.insert(0, str(REPO / "src"))
from task3.score_task3 import network_parameters

KEY = ["timestamp", "link_id"]


class Proxy:
    def __init__(self, rel, panel):
        pdir = rel / "corridors" / panel
        files = sorted((pdir / "train" / "mainline_states").glob("**/*.parquet"))
        f = pd.concat([pd.read_parquet(p, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]) for p in files], ignore_index=True)
        f = f[f.is_score_eligible.astype(bool)].drop(columns="is_score_eligible")
        tpl = pd.read_csv(rel / "task1" / panel / "train" / "sample_submission_state.csv", usecols=KEY + ["mask_regime"])
        f = f.merge(tpl, on=KEY, how="left")
        f["is_target"] = f.mask_regime.notna()
        f["date"] = f.timestamp.str[:10]
        reg_by_date = tpl.assign(date=tpl.timestamp.str[:10]).groupby("date").mask_regime.first()
        f["regime"] = f.date.map(reg_by_date)
        par = network_parameters(pdir).set_index("link_id")
        for c in ["length_km", "lanes", "free_speed_kmh", "capacity_vph", "critical_density", "k_jam"]:
            f[c] = f.link_id.map(par[c]).to_numpy()
        f["ts"] = pd.to_datetime(f.timestamp, utc=True)
        f = f.sort_values(["link_id", "ts"]).reset_index(drop=True)
        gap = (f.groupby("link_id").ts.shift(-1) - f.ts).dt.total_seconds() / 3600.0
        f["has_next"] = (gap > 0) & (gap <= 5.1 / 60.0)
        f["N_true"] = f.flow_vph / f.speed_kmh.clip(lower=1.0) * f.length_km
        self.f = f

    def score(self, sub):
        s = pd.read_csv(sub, usecols=KEY + ["speed_kmh", "flow_vph"]).rename(columns={"speed_kmh": "sp", "flow_vph": "fl"})
        f = self.f.merge(s, on=KEY, how="left")
        t = f.is_target.to_numpy()
        v = np.where(t, f.sp, f.speed_kmh)
        q = np.where(t, f.fl, f.flow_vph)
        N = q / np.clip(v, 1.0, None) * f.length_km.to_numpy()
        Nt = f.N_true.to_numpy()
        nxt = np.r_[N[1:], np.nan]
        nxt_t = np.r_[Nt[1:], np.nan]
        hn = f.has_next.to_numpy()
        lanes = np.clip(f.lanes.to_numpy(), 1, None)
        kl = q / np.clip(v, 1.0, None) / lanes
        kc, kj, cap, vf = f.critical_density.to_numpy() / lanes, f.k_jam.to_numpy() / lanes, f.capacity_vph.to_numpy() / lanes, f.free_speed_kmh.to_numpy()
        qfd = np.where(kl <= kc, vf * kl, cap / np.maximum(kj - kc, 1e-9) * np.maximum(kj - kl, 0))
        ql = q / lanes
        out = []
        for r in ["R1", "R2", "R3"]:
            m = (f.regime == r).to_numpy()
            mm = m & hn
            res = np.abs((nxt - N) - (nxt_t - Nt))[mm]
            den = np.abs(nxt_t - Nt)[mm]
            e = res.sum() / den.sum()
            sfd = max(0.0, 1 - np.abs(ql[m] - qfd[m]).sum() / np.abs(ql[m]).sum())
            slwr = max(0.0, 1 - min(1.0, e))
            out.append({"regime": r, "S_FD": round(sfd, 4), "E_LWR": round(e, 4), "S_LWR": round(slwr, 4), "S_physics": round((sfd + 2 * slwr) / 3, 4)})
        return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--panel", default="D12_I5_N")
    ap.add_argument("--sub", action="append", required=True)
    a = ap.parse_args()
    px = Proxy(Path(a.release_root), a.panel)
    for s in a.sub:
        d = px.score(s)
        print(Path(s).name, d.to_dict("list"), "mean", round(d.S_physics.mean(), 4), flush=True)


if __name__ == "__main__":
    main()
