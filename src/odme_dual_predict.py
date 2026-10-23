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


def solve_kl(A, c, b, mu, tol=1e-9, it=200):
    y = np.zeros(A.shape[0])
    pos = b > 0

    def fy(y):
        return np.where(pos, b * np.exp(np.clip(A.T @ y, -30, 30)), 0.0)

    def F(y, f):
        return A @ f - c + y / mu

    f = fy(y)
    r = F(y, f)
    for _ in range(it):
        Af = A.multiply(f) if hasattr(A, "multiply") else A * f
        J = np.asarray((Af @ A.T).todense() if hasattr(Af, "todense") else Af @ A.T) + np.eye(A.shape[0]) / mu
        dy = np.linalg.solve(J, -r)
        t, n0 = 1.0, np.linalg.norm(r)
        while t > 1e-6:
            yn = y + t * dy
            fn = fy(yn)
            rn = F(yn, fn)
            if np.linalg.norm(rn) < n0:
                break
            t *= 0.5
        y, f, r = yn, fn, rn
        if np.linalg.norm(r) <= tol * max(np.linalg.norm(c), 1.0):
            break
    return f


def solve_panel(rel, panel, split, lam, check, method="l2", mu=1.0):
    s = time.time()
    path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
    cf = released_counts(rel, panel, split)
    if cf is None:
        cf = counts_for_split(rel / "corridors" / panel, split)
    m = np.isin(np.asarray(link_ids), cf.link_id.values)
    Am, c, b = A[m], counts_vec(cf, link_ids)[m], prior_values(rel, panel, split, path_ids)
    info = {"panel": panel, "split": split, "n_paths": A.shape[1], "n_meas": int(m.sum())}
    if method == "kl":
        f = solve_kl(Am, c, b, mu)
        info["rel_resid"] = float(np.abs(Am @ f - c).sum() / max(c.sum(), 1e-12))
    else:
        f = solve_dual(Am, c, b, lam)
        fk = b + Am.T @ (c - Am @ f) / lam
        info["kkt_active"] = float(np.abs(f - fk)[f > 0].max(initial=0.0))
        info["kkt_zero"] = float(np.maximum(fk, 0.0)[f <= 0].max(initial=0.0))
    info["solve_s"] = time.time() - s
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
    ap.add_argument("--method", choices=["l2", "kl"], default="l2")
    ap.add_argument("--mu", type=float, default=1.0)
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
            x, info = solve_panel(rel, panel, split, a.lam, a.check_nnls, a.method, a.mu)
            parts.append(x)
            infos.append(info)
            print(f"[odme dual] {split} {panel} paths={info['n_paths']} meas={info['n_meas']} {info['solve_s']:.2f}s", flush=True)
        pd.concat(parts, ignore_index=True).to_csv(d / "public_reference.csv", index=False)
    pd.DataFrame(infos).to_csv(out / "dual_solve_info.csv", index=False)
    print(pd.DataFrame(infos).drop(columns=["panel"]).groupby("split").agg(["max", "sum"]).round(6).to_string(), flush=True)
    print(f"total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
