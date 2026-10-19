import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent / "reference" / "trafficflowbench-public"
sys.path.insert(0, str(REPO / "src"))
from task3.score_task3 import load_ramp_flows, network_parameters

DT_H = 5.0 / 60.0


def build_flux(rel, panel, out):
    pdir = rel / "corridors" / panel
    files = sorted((pdir / "train" / "mainline_states").glob("**/*.parquet"))
    f = pd.concat([pd.read_parquet(p, columns=["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"]) for p in files], ignore_index=True)
    f = f[f.is_score_eligible.astype(bool)].copy()
    f["timestamp"] = pd.to_datetime(f.timestamp, utc=True)
    f["link_id"] = f.link_id.astype(str)
    par = network_parameters(pdir).set_index("link_id")
    f["N"] = f.flow_vph / f.speed_kmh.clip(lower=1.0) * f.link_id.map(par.length_km)
    f = f.sort_values(["link_id", "timestamp"])
    nxt = f.groupby("link_id").N.shift(-1)
    gap = (f.groupby("link_id").timestamp.shift(-1) - f.timestamp).dt.total_seconds() / 3600.0
    f["dN"] = np.where((gap > 0) & (gap <= 5.1 / 60.0), nxt - f.N, 0.0)
    r = load_ramp_flows(pdir, "train")
    f = f.merge(r[["timestamp", "link_id", "on_ramp_flow_vph", "off_ramp_flow_vph"]], on=["timestamp", "link_id"], how="left")
    on = f.on_ramp_flow_vph.fillna(0.0)
    off = f.off_ramp_flow_vph.fillna(0.0)
    f["outflow_vph"] = f.flow_vph
    f["inflow_vph"] = f.flow_vph + f.dN / DT_H - on + off
    f[["timestamp", "link_id", "inflow_vph", "outflow_vph"]].to_parquet(out, index=False)


def truth_submission(rel, panel, out):
    tpl = pd.read_csv(rel / "task1" / panel / "train" / "sample_submission_state.csv", dtype={"timestamp": str})
    files = sorted((rel / "corridors" / panel / "train" / "mainline_states").glob("**/*.parquet"))
    f = pd.concat([pd.read_parquet(p, columns=["timestamp", "link_id", "speed_kmh", "flow_vph"]) for p in files], ignore_index=True)
    tpl = tpl.drop(columns=["speed_kmh", "flow_vph"]).merge(f, on=["timestamp", "link_id"], how="left")
    tpl.to_csv(out, index=False)


def score(rel, panel, sub, flux, out):
    r = subprocess.run([sys.executable, str(REPO / "src" / "task3" / "score_task3.py"), "--state-submission", str(sub), "--split", "train",
                        "--boundary-flux", str(flux), "--release-root", str(rel), "--panel", panel, "--output", str(out)], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit(1)
    d = pd.read_csv(out)
    return d[d.level == "panel_regime"][["regime", "S_FD", "S_LWR", "E_LWR", "S_physics"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--panel", default="D12_I5_N")
    ap.add_argument("--work", required=True)
    ap.add_argument("--sub", action="append", default=[])
    a = ap.parse_args()
    rel, work = Path(a.release_root), Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    flux = work / f"flux_{a.panel}.parquet"
    if not flux.exists():
        build_flux(rel, a.panel, flux)
    subs = list(a.sub)
    tsub = work / f"truth_{a.panel}.csv"
    if not tsub.exists():
        truth_submission(rel, a.panel, tsub)
    subs = [str(tsub)] + subs
    for s in subs:
        d = score(rel, a.panel, s, flux, work / "t3_eval.csv")
        print(Path(s).name, d.round(4).to_dict("list"), "mean S_physics", round(d.S_physics.mean(), 4), flush=True)


if __name__ == "__main__":
    main()
