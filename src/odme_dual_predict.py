import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odme_scan import REPO, counts_vec, solve_dual, solve_nnls
from task4.build_task4_odme_artifacts import REG_LAMBDA, counts_for_split, load_operator, prior_values, released_counts, released_prior

COLS = ["panel", "departure_time", "path_id", "origin_zone", "destination_zone", "path_flow"]


def solve_panel(rel, panel, split, lam, check):
    s = time.time()
    path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
    cf = released_counts(rel, panel, split)
    if cf is None:
        cf = counts_for_split(rel / "corridors" / panel, split)
    m = np.isin(np.asarray(link_ids), cf.link_id.values)
    Am, c, b = A[m], counts_vec(cf, link_ids)[m], prior_values(rel, panel, split, path_ids)
    f = solve_dual(Am, c, b, lam)
    info = {"panel": panel, "split": split, "n_paths": A.shape[1], "n_meas": int(m.sum()), "solve_s": time.time() - s}
    fk = b + Am.T @ (c - Am @ f) / lam
    info["kkt_active"] = float(np.abs(f - fk)[f > 0].max(initial=0.0))
    info["kkt_zero"] = float(np.maximum(fk, 0.0)[f <= 0].max(initial=0.0))
    if check:
        fn = solve_nnls(Am, c, b, lam)
        info["rel_l1_vs_nnls"] = float(np.abs(f - fn).sum() / max(fn.sum(), 1e-12))
    prior = released_prior(rel, panel, split)
    x = paths[["path_id", "origin_zone", "destination_zone"]].copy()
    x["panel"] = panel
    x["departure_time"] = str(prior.departure_time.iloc[0])
    x["path_flow"] = f
    return x[COLS], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", action="append", required=True, choices=["train", "validation", "private"])
    ap.add_argument("--lam", type=float, default=REG_LAMBDA)
    ap.add_argument("--check-nnls", action="store_true")
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out_dir)
    panels = [p["corridor_id"] for p in json.loads((REPO / "config" / "corridors.json").read_text(encoding="utf-8"))["panels"]]
    t0 = time.time()
    infos = []
    for split in a.split:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        parts = []
        for panel in panels:
            x, info = solve_panel(rel, panel, split, a.lam, a.check_nnls)
            parts.append(x)
            infos.append(info)
            print(f"[odme dual] {split} {panel} paths={info['n_paths']} meas={info['n_meas']} {info['solve_s']:.2f}s", flush=True)
        pd.concat(parts, ignore_index=True).to_csv(d / "public_reference.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "dual_solve_info.csv", index=False)
    print(pd.DataFrame(infos).drop(columns=["panel"]).groupby("split").agg(["max", "sum"]).round(6).to_string(), flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
