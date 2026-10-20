import argparse
import hashlib
import importlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
PROJECT = SRC.parent
sys.path.insert(0, str(SRC))

STAGES = ["state", "queue", "odme", "merge"]
SPLITS = ["validation", "private"]
SPEED_NORM, FLOW_NORM, TOL = 25.0, 600.0, 0.01
STATE_SCRIPTS = {"A": "task1_resid_predict", "LE": "task1_resid2_predict", "B3": "task1_resid3_predict"}
STAGE_MODULES = {"state": {k: [v] for k, v in STATE_SCRIPTS.items()}, "queue": {"v4": ["task2_predict_v2"], "final": ["queue_final_predict"]}}


def versions():
    out = {"python": platform.python_version()}
    for m in ["numpy", "pandas", "pyarrow", "lightgbm", "scipy"]:
        out[m] = importlib.import_module(m).__version__
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git_info(path):
    def run(*a):
        r = subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    st = run("status", "--porcelain", "--untracked-files=no")
    return {"path": str(path), "commit": run("rev-parse", "HEAD"), "dirty": None if st is None else bool(st)}


def child_peak_gb():
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9


def paths(a):
    out = Path(a.out_dir)
    return {"state_dir": out / "state", "state_vp": Path(a.state_vp) if a.state_vp else out / "state" / "state_vp.csv",
            "queue_dir": out / "queue", "queue_vp": Path(a.queue_vp) if a.queue_vp else out / "queue" / "queue_vp.csv",
            "odme_dir": out / "odme", "odme_vp": Path(a.odme_vp) if a.odme_vp else out / "odme" / "odme_vp.csv",
            "submission": out / "submission.csv", "key": Path(a.release_root) / "submission_key.csv",
            "sample": Path(a.release_root) / "sample_submission.csv"}


def py(script, *args):
    return [sys.executable, "-I", str(script), *map(str, args)]


def plan(a, P):
    rel, repo = Path(a.release_root), Path(a.repo_root)
    steps = {}
    if "state" in a.stages:
        steps["state"] = [py(SRC / f"{STATE_SCRIPTS[a.state]}.py", "--release-root", rel, "--out-dir", P["state_dir"], "--cache-dir", P["state_dir"] / "cache",
                             "--threads", a.threads, "--seed", a.seed)]
    if "queue" in a.stages:
        script = "task2_predict_v2.py" if a.queue == "v4" else "queue_final_predict.py"
        steps["queue"] = [py(SRC / script, "--release-root", rel, "--out", P["queue_dir"])]
    if "odme" in a.stages and a.odme_solver == "dual":
        steps["odme"] = [py(SRC / "odme_dual_predict.py", "--release-root", rel, "--out-dir", P["odme_dir"], *[x for s in SPLITS for x in ("--split", s)])]
    elif "odme" in a.stages:
        steps["odme"] = [py(repo / "src" / "task4" / "build_task4_odme_artifacts.py", "--release-root", rel, "--split", s, "--output-root", P["odme_dir"] / s)
                         for s in SPLITS]
    if "merge" in a.stages:
        steps["merge"] = [py(repo / "src" / "merge_submissions.py", "--state", P["state_vp"], "--queue", P["queue_vp"], "--odme", P["odme_vp"],
                             "--key", P["key"], "--output", P["submission"])]
    return steps


def concat_csv(parts, dst):
    pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in parts], ignore_index=True).to_csv(dst, index=False)


def post(stage, a, P):
    if stage == "queue" and a.queue == "v4":
        concat_csv([P["queue_dir"] / f"queue_{s}.csv" for s in SPLITS], P["queue_vp"])
    if stage == "odme":
        concat_csv([P["odme_dir"] / s / "public_reference.csv" for s in SPLITS], P["odme_vp"])
    return {"state": P["state_vp"], "queue": P["queue_vp"], "odme": P["odme_vp"], "merge": P["submission"]}[stage]


def run_stage(stage, cmds, a, P, man, mpath):
    s = time.time()
    log = Path(a.out_dir) / f"{stage}.log"
    with open(log, "w") as f:
        for c in cmds:
            print(f"[{stage}] {' '.join(c)}", flush=True)
            r = subprocess.run(c, stdout=f, stderr=subprocess.STDOUT, cwd=str(PROJECT))
            if r.returncode:
                man["stages"][stage] = {"status": "failed", "returncode": r.returncode, "runtime_s": round(time.time() - s, 1), "log": str(log)}
                mpath.write_text(json.dumps(man, indent=2))
                sys.exit(f"{stage} failed (rc {r.returncode}), see {log}")
    out = post(stage, a, P)
    man["stages"][stage] = {"status": "ok", "runtime_s": round(time.time() - s, 1), "child_peak_gb_so_far": round(child_peak_gb(), 2), "log": str(log),
                            "output": str(out), "output_sha256": sha256(out), "output_bytes": out.stat().st_size}
    print(f"[{stage}] {man['stages'][stage]['runtime_s']}s -> {out}", flush=True)
    mpath.write_text(json.dumps(man, indent=2))


def lanes_for_state(a, key_state):
    sys.path.insert(0, str(Path(a.repo_root) / "src"))
    from task1.baseline_task1_historical_mean import lane_vector, station_lanes
    ln = np.ones(len(key_state))
    for panel, g in key_state.groupby("panel"):
        lb = station_lanes(Path(a.release_root) / "corridors" / panel)
        ln[g.index.to_numpy()] = lane_vector(g.station_id, g.link_id, lb)
    return ln


def compare(a, P):
    new_path, ref_path = P["submission"], Path(a.compare)
    num = {c: float for c in ["speed_kmh", "flow_vph", "queue_pred", "path_flow"]}
    new = pd.read_csv(new_path, dtype={"submission_id": np.int64, "task": str, **num})
    ref = pd.read_csv(ref_path, dtype={"submission_id": np.int64, "task": str, **num})
    res = {"new": str(new_path), "ref": str(ref_path), "rows_new": len(new), "rows_ref": len(ref)}
    same = len(new) == len(ref) and bool((new.submission_id.to_numpy() == ref.submission_id.to_numpy()).all()) and bool((new.task.to_numpy() == ref.task.to_numpy()).all())
    res["same_id_task_order"] = same
    if not same:
        ref = new[["submission_id"]].merge(ref, on="submission_id", how="left")
    cols = {"state": ["speed_kmh", "flow_vph"], "queue": ["queue_pred"], "odme": ["path_flow"]}
    checks = {}
    for task, cs in cols.items():
        m = (new.task == task).to_numpy()
        x, y = new.loc[m, cs].to_numpy(), ref.loc[m, cs].to_numpy()
        d = np.abs(x - y)
        r = {"rows": int(m.sum()), "share_rows_differ": float((np.nan_to_num(d, nan=np.inf) > 1e-9).any(axis=1).mean()) if m.any() else 0.0}
        for j, c in enumerate(cs):
            r[c] = {"max_abs_diff": float(np.nanmax(d[:, j])), "mean_abs_diff": float(np.nanmean(d[:, j])), "nan_ref": int(np.isnan(y[:, j]).sum())}
        if task == "state":
            key = pd.read_csv(P["key"], usecols=["task", "panel", "station_id", "link_id"], dtype=str)
            key = key[key.task == "state"].reset_index(drop=True)
            assert len(key) == m.sum()
            ln = lanes_for_state(a, key)
            r["rmse_speed_kmh"] = float(np.sqrt(np.mean((x[:, 0] - y[:, 0]) ** 2)))
            r["rmse_flow_vph_per_lane"] = float(np.sqrt(np.mean(((x[:, 1] - y[:, 1]) / ln) ** 2)))
            checks["state_speed"] = r["rmse_speed_kmh"] < TOL * SPEED_NORM
            checks["state_flow"] = r["rmse_flow_vph_per_lane"] < TOL * FLOW_NORM
        if task == "queue":
            r["agreement"] = float((x[:, 0] == y[:, 0]).mean())
            checks["queue"] = r["agreement"] >= 1 - TOL
        if task == "odme":
            r["relative_l1"] = float(d[:, 0].sum() / max(np.abs(y[:, 0]).sum(), 1e-12))
            checks["odme"] = r["relative_l1"] < TOL
        res[task] = r
    res["rule"] = f"state RMSE < {TOL:.0%} of {SPEED_NORM:g} km/h and {FLOW_NORM:g} veh/h/lane; queue agreement >= {1-TOL:.0%}; odme relative L1 < {TOL:.0%}"
    res["checks"] = checks
    res["pass"] = bool(same and all(checks.values()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--state", choices=list(STATE_SCRIPTS), default="A")
    ap.add_argument("--queue", choices=["v4", "final"], default="final")
    ap.add_argument("--odme", choices=["own"], default="own")
    ap.add_argument("--odme-solver", choices=["official", "dual"], default="official")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stages", nargs="*", choices=STAGES, default=STAGES)
    ap.add_argument("--state-vp")
    ap.add_argument("--queue-vp")
    ap.add_argument("--odme-vp")
    ap.add_argument("--compare")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    a.release_root, a.repo_root, a.out_dir = (str(Path(x).resolve()) for x in (a.release_root, a.repo_root, a.out_dir))
    P = paths(a)
    steps = plan(a, P)
    if a.dry_run:
        mods = STAGE_MODULES["state"][a.state] * ("state" in a.stages) + STAGE_MODULES["queue"][a.queue] * ("queue" in a.stages)
        imports = {}
        for m in mods:
            try:
                importlib.import_module(m)
                imports[m] = "ok"
            except Exception as e:
                imports[m] = f"{type(e).__name__}: {e}"
        produced = {"state": P["state_vp"], "queue": P["queue_vp"], "odme": P["odme_vp"]}
        need = [P["key"], P["sample"]] + [c[2] for cs in steps.values() for c in cs]
        if "merge" in a.stages:
            need += [p for s, p in produced.items() if s not in a.stages]
        out = {"stages": {s: [" ".join(c) for c in cs] for s, cs in steps.items()}, "outputs": {k: str(v) for k, v in P.items()},
               "inputs_exist": {str(p): Path(p).exists() for p in need}, "imports": imports, "versions": versions(),
               "git": {"project": git_info(PROJECT), "official": git_info(a.repo_root)}, "compare": a.compare}
        print(json.dumps(out, indent=1))
        return
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    t0 = time.time()
    man = {"args": vars(a), "argv": sys.argv, "versions": versions(), "git": {"project": git_info(PROJECT), "official": git_info(a.repo_root)},
           "inputs": {"submission_key.csv": sha256(P["key"]), "sample_submission.csv": sha256(P["sample"])},
           "inputs_external": {k: str(P[k]) for k, f in [("state_vp", a.state_vp), ("queue_vp", a.queue_vp), ("odme_vp", a.odme_vp)] if f},
           "stages": {}}
    mpath.write_text(json.dumps(man, indent=2))
    for stage in STAGES:
        if stage in steps:
            run_stage(stage, steps[stage], a, P, man, mpath)
    if a.compare:
        s = time.time()
        man["compare"] = compare(a, P)
        man["compare"]["runtime_s"] = round(time.time() - s, 1)
        print(json.dumps(man["compare"], indent=1), flush=True)
        print("COMPARE", "PASS" if man["compare"]["pass"] else "FAIL", flush=True)
    man["outputs"] = {k: {"path": str(P[k]), "sha256": sha256(P[k])} for k in ["state_vp", "queue_vp", "odme_vp", "submission"] if P[k].exists()}
    man["runtime_s"] = round(time.time() - t0, 1)
    man["child_peak_gb"] = round(child_peak_gb(), 2)
    mpath.write_text(json.dumps(man, indent=2))
    print(f"total {man['runtime_s']}s -> {mpath}", flush=True)


if __name__ == "__main__":
    main()
