import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO = Path(__file__).resolve().parent.parent / "reference" / "trafficflowbench-public"
sys.path.insert(0, str(REPO / "src"))
from task4.build_task4_odme_artifacts import load_operator, released_counts, prior_values

SPLITS = ["train", "validation", "private"]
LAMS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
LAMS_LARGE = [0.05, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 1e5]


def solve_nnls(A, c, b, lam):
    aa = np.vstack([A, np.sqrt(lam) * np.eye(A.shape[1])])
    bb = np.concatenate([c, np.sqrt(lam) * b])
    return nnls(aa, bb)[0]


def solve_dual(A, c, b, lam, tol=1e-10, it=200):
    y = np.zeros(A.shape[0])

    def fy(y):
        return np.maximum(0.0, b - A.T @ y / lam)

    def g(y, f):
        return -y @ y + lam * ((f - b) ** 2).sum() + 2 * y @ (A @ f - c)

    f = fy(y)
    gv = g(y, f)
    for _ in range(it):
        r = A @ f - c - y
        if np.abs(r).max() <= tol * max(1.0, np.abs(c).max()):
            break
        S = f > 0
        As = A[:, S]
        H = np.eye(A.shape[0]) + As @ As.T / lam
        d = np.linalg.solve(H, r)
        t = 1.0
        while True:
            yn = y + t * d
            fn = fy(yn)
            gn = g(yn, fn)
            if gn >= gv - 1e-12 * abs(gv) or t < 1e-8:
                break
            t *= 0.5
        y, f, gv = yn, fn, gn
    return f


def s_link(A, f, c):
    return max(0.0, 1 - np.abs(A @ f - c).sum() / c.sum())


def attraction(v, dest):
    s = pd.Series(v).groupby(dest).sum()
    return s / max(s.sum(), 1e-12)


def counts_vec(frame, link_ids):
    return frame.set_index("link_id").reindex(link_ids).fillna(0.0)["count"].to_numpy(dtype=float)


def fold_err(A, c, b, lam, solver, k=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(A.shape[0])
    errs = []
    for h in np.array_split(idx, k):
        tr = np.setdiff1d(idx, h)
        f = solver(A[tr], c[tr], b, lam)
        errs.append(np.abs(A[h] @ f - c[h]).sum() / c[h].sum())
    return float(np.mean(errs))


def l1_share(x, y):
    return float(np.abs(x / max(x.sum(), 1e-12) - y / max(y.sum(), 1e-12)).sum())


def run_large(rel, out, panels, solver):
    rows, stab = [], []
    t0 = time.time()
    for panel in panels:
        path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
        dest = paths.destination_zone.astype(str).to_numpy()
        lid = np.asarray(link_ids)
        fs, base = {}, {}
        for s in SPLITS:
            cf = released_counts(rel, panel, s)
            m = np.isin(lid, cf.link_id.values)
            Am, c, b = A[m], counts_vec(cf, link_ids)[m], prior_values(rel, panel, s, path_ids)
            base[s] = b
            scale = c.sum() / (Am @ b).sum()
            b_attr = attraction(b, dest)
            for variant, bu in [("prior", b), ("scaled", b * scale)]:
                for lam in LAMS_LARGE:
                    t = time.time()
                    f = solver(Am, c, bu, lam)
                    st = time.time() - t
                    fs[(variant, s, lam)] = f
                    rows.append({
                        "panel": panel, "split": s, "variant": variant, "lam": lam, "scale": scale if variant == "scaled" else 1.0,
                        "S_link_own": s_link(Am, f, c), "S_link_bused": s_link(Am, bu, c),
                        "Dhat": np.abs(f - b).sum(), "Dhat_rel": np.abs(f - b).sum() / b.sum(),
                        "Dhat_rel_bused": np.abs(f - bu).sum() / bu.sum(), "sumf_over_sumb": f.sum() / b.sum(),
                        "attr_l1_f_b": float(np.abs(attraction(f, dest) - b_attr).sum()),
                        "frac_zero": float((f <= 1e-9).mean()), "heldout_rel_err": fold_err(Am, c, bu, lam, solver), "solve_s": st,
                    })
            print(f"{panel} {s} done {time.time() - t0:.1f}s", flush=True)
        for variant in ["prior", "scaled"]:
            for lam in LAMS_LARGE:
                stab.append({
                    "panel": panel, "variant": variant, "lam": lam,
                    "l1_train_val": l1_share(fs[(variant, "train", lam)], fs[(variant, "validation", lam)]),
                    "l1_train_priv": l1_share(fs[(variant, "train", lam)], fs[(variant, "private", lam)]),
                    "l1_val_priv": l1_share(fs[(variant, "validation", lam)], fs[(variant, "private", lam)]),
                    "prior_l1_train_val": l1_share(base["train"], base["validation"]),
                    "prior_l1_val_priv": l1_share(base["validation"], base["private"]),
                })
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "odme_scan_large.csv", index=False)
    cols = ["S_link_own", "Dhat_rel", "Dhat_rel_bused", "sumf_over_sumb", "attr_l1_f_b", "frac_zero", "heldout_rel_err"]
    summ = scan.groupby(["split", "variant", "lam"])[cols].mean().reset_index()
    summ.to_csv(out / "odme_summary_large.csv", index=False)
    sb = pd.DataFrame(stab)
    sb.to_csv(out / "stability_large.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string(index=False))
    print(sb.drop(columns="panel").groupby(["variant", "lam"]).mean().round(4).to_string())
    print(scan.groupby(["split", "variant"]).scale.agg(["min", "mean", "max"]).round(4).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {"train": "tab:blue", "validation": "tab:orange", "private": "tab:green"}
    styles = {"prior": "-o", "scaled": "--s"}
    for ax, col, lab in zip(axes, ["Dhat_rel", "S_link_own", "heldout_rel_err"], ["Dhat / sum(b)", "S_link (kendi sayımlar)", "held-out bağıl hata"]):
        for s in SPLITS:
            for v in ["prior", "scaled"]:
                d = summ[(summ.split == s) & (summ.variant == v)]
                ax.plot(d.lam, d[col], styles[v], color=colors[s], ms=3, label=f"{s} {v}")
        ax.set_xscale("log")
        ax.set_xlabel("lambda (log)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("ODME geniş lambda taraması: ham ve ölçeklenmiş öncül (panel ortalaması)")
    fig.tight_layout()
    fig.savefig(out / "dev_curve_large.png", dpi=120)
    print(f"total {time.time() - t0:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--panel", action="append")
    ap.add_argument("--solver", choices=["dual", "nnls"], default="dual")
    ap.add_argument("--fold-lams", default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--large", action="store_true")
    a = ap.parse_args()
    rel = Path(a.release_root)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    solver = solve_dual if a.solver == "dual" else solve_nnls
    fold_lams = LAMS if a.fold_lams == "all" else [float(x) for x in a.fold_lams.split(",")]
    panels = a.panel or sorted(p.name for p in (rel / "corridors").iterdir() if p.is_dir())
    if a.large:
        run_large(rel, out, panels, solver)
        return
    rows, diag, stab, checks = [], [], [], []
    t0 = time.time()
    for panel in panels:
        path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
        dest = paths.destination_zone.astype(str).to_numpy()
        lid = np.asarray(link_ids)
        cf = {s: released_counts(rel, panel, s) for s in SPLITS}
        base = {s: prior_values(rel, panel, s, path_ids) for s in SPLITS}
        fs = {}
        for s in SPLITS:
            m = np.isin(lid, cf[s].link_id.values)
            Am, c, b = A[m], counts_vec(cf[s], link_ids)[m], base[s]
            ab_ratio = (Am @ b).sum() / c.sum()
            ab_attr = attraction(b, dest)
            for lam in LAMS:
                t = time.time()
                f = solver(Am, c, b, lam)
                st = time.time() - t
                fs[(s, lam)] = f
                if a.check and lam == 0.05:
                    fn = solve_nnls(Am, c, b, lam)
                    checks.append({"panel": panel, "split": s, "rel_l1_vs_nnls": np.abs(f - fn).sum() / max(fn.sum(), 1e-12)})
                dhat = np.abs(f - b).sum()
                he = fold_err(Am, c, b, lam, solver) if lam in fold_lams else np.nan
                rows.append({
                    "panel": panel, "split": s, "lam": lam, "n_paths": A.shape[1], "n_meas": int(m.sum()),
                    "Ab_over_c": ab_ratio, "S_link_prior": s_link(Am, b, c), "S_link_own": s_link(Am, f, c),
                    "Dhat": dhat, "Dhat_rel": dhat / b.sum(), "sumf_over_sumb": f.sum() / b.sum(),
                    "attr_l1_f_b": float(np.abs(attraction(f, dest) - ab_attr).sum()),
                    "frac_zero": float((f <= 1e-9).mean()), "frac_zero_prior": float((b <= 1e-9).mean()),
                    "heldout_rel_err": he, "solve_s": st,
                })
            if s != "train":
                lam = 0.05
                ctr = counts_vec(cf["train"], link_ids)[m]
                ftc = solver(Am, ctr, b, lam)
                fown = fs[(s, lam)]
                diag.append({
                    "panel": panel, "split": s, "S_link_traincount": s_link(Am, ftc, c), "S_link_owncount": s_link(Am, fown, c),
                    "S_link_prior": s_link(Am, b, c), "rel_l1_tc_vs_own": np.abs(ftc - fown).sum() / fown.sum(),
                    "count_rel_l1_train_vs_own": np.abs(ctr - c).sum() / c.sum(),
                    "n_meas_missing_in_train": int((~np.isin(lid[m], cf["train"].link_id.values)).sum()),
                })
            print(f"{panel} {s} done {time.time() - t0:.1f}s", flush=True)
        for lam in LAMS:
            stab.append({
                "panel": panel, "lam": lam,
                "l1_train_val": l1_share(fs[("train", lam)], fs[("validation", lam)]),
                "l1_train_priv": l1_share(fs[("train", lam)], fs[("private", lam)]),
                "l1_val_priv": l1_share(fs[("validation", lam)], fs[("private", lam)]),
                "prior_l1_train_val": l1_share(base["train"], base["validation"]),
                "prior_l1_val_priv": l1_share(base["validation"], base["private"]),
            })
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "odme_scan.csv", index=False)
    cols = ["S_link_prior", "S_link_own", "Dhat_rel", "sumf_over_sumb", "attr_l1_f_b", "frac_zero", "heldout_rel_err", "solve_s"]
    summ = scan.groupby(["split", "lam"])[cols].mean().reset_index()
    summ.to_csv(out / "odme_summary.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(out / "count_source_diag.csv", index=False)
    sb = pd.DataFrame(stab)
    sb.to_csv(out / "stability.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print(summ.round(4).to_string(index=False))
    print(dg.round(4).to_string(index=False))
    print(dg.drop(columns="panel").groupby("split").mean().round(4).to_string())
    print(sb.drop(columns="panel").groupby("lam").mean().round(4).to_string())
    print(scan[scan.lam == 0.05][["panel", "split", "n_paths", "n_meas", "Ab_over_c", "S_link_prior", "frac_zero_prior", "solve_s"]].round(4).to_string(index=False))
    if checks:
        print(pd.DataFrame(checks).to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    colors = {"train": "tab:blue", "validation": "tab:orange", "private": "tab:green"}
    for s in SPLITS:
        d = summ[summ.split == s]
        ax.plot(d.lam, d.Dhat_rel, "-o", color=colors[s], label=f"{s} Dhat/sum(b)")
        ax2.plot(d.lam, d.S_link_own, "--s", color=colors[s], label=f"{s} S_link (kendi)")
        dh = d.dropna(subset=["heldout_rel_err"])
        ax2.plot(dh.lam, dh.heldout_rel_err, ":^", color=colors[s], label=f"{s} held-out hata")
    ax.set_xscale("log")
    ax.set_xlabel("lambda (log)")
    ax.set_ylabel("Dhat / sum(b)")
    ax2.set_ylabel("S_link / held-out bağıl hata")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax.set_title("ODME lambda taraması (panel ortalaması)")
    fig.tight_layout()
    fig.savefig(out / "dev_curve.png", dpi=120)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
