import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odme_dual_predict import solve_kl
from odme_scan import counts_vec, solve_dual
from task4.build_task4_odme_artifacts import load_operator, prior_values, released_counts


def scores(f, fs, b, A, c, paths):
    att = lambda x: pd.Series(x, index=paths.destination_zone.to_numpy()).groupby(level=0).sum().pipe(lambda s: s / s.sum())
    s_od = max(0, 1 - np.abs(f - fs).sum() / fs.sum())
    s_link = max(0, 1 - np.abs(A @ f - c).sum() / c.sum())
    dh, ds = np.abs(f - b).sum(), np.abs(fs - b).sum()
    s_dev = np.exp(-abs(dh / ds - 1))
    s_attr = max(0, 1 - 0.5 * np.abs(att(f) - att(fs)).sum())
    return {"S_od": s_od, "S_link": s_link, "S_dev": s_dev, "S_attr": s_attr, "S_ODME": 0.45 * s_od + 0.25 * s_link + 0.15 * s_dev + 0.15 * s_attr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panels", nargs="+", default=["D12_I5_N", "D7_I405_S", "D12_I405_S"])
    a = ap.parse_args()
    rel = Path(a.release_root)
    rows = []
    for panel in a.panels:
        path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
        cf = released_counts(rel, panel, "validation")
        m = np.isin(np.asarray(link_ids), cf.link_id.values)
        Am, c_real = A[m], counts_vec(cf, link_ids)[m]
        b = prior_values(rel, panel, "validation", path_ids)
        real_corr = np.corrcoef(Am @ b, c_real)[0, 1]
        real_ratio = (Am @ b).sum() / c_real.sum()
        for sigma in [0.5, 1.0, 1.5, 2.0]:
            for seed in range(3):
                rng = np.random.default_rng(seed)
                fs = b * np.exp(rng.normal(0, sigma, len(b)))
                fs *= (Am @ b).sum() / real_ratio / (Am @ fs).sum()
                c = Am @ fs * np.exp(rng.normal(0, 0.02, len(c_real)))
                r = {"panel": panel, "sigma": sigma, "seed": seed, "corr_Ab_c": np.corrcoef(Am @ b, c)[0, 1], "real_corr": real_corr}
                for name, f in [("l2_lam0.05", solve_dual(Am, c, b, 0.05)), ("l2_lam100", solve_dual(Am, c, b, 100.0)), ("kl_mu1", solve_kl(Am, c, b, 1.0)), ("prior", b * c.sum() / (Am @ b).sum())]:
                    for k, v in scores(f, fs, b, Am, c, paths).items():
                        r[f"{name}|{k}"] = v
                rows.append(r)
        print(panel, "real corr(Ab,c)", round(real_corr, 3), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    g = df.groupby("sigma").mean(numeric_only=True)
    cols = [c for c in g.columns if "|" in c and (c.endswith("S_ODME") or c.endswith("S_od"))]
    print(g[["corr_Ab_c"] + cols].round(3).T.to_string())


if __name__ == "__main__":
    main()
