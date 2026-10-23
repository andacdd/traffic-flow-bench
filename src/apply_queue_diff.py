import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--from-a", required=True)
    ap.add_argument("--from-b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    a = pd.read_csv(args.from_a)
    b = pd.read_csv(args.from_b)
    assert (base.submission_id.values == a.submission_id.values).all()
    assert (a.submission_id.values == b.submission_id.values).all()

    cols = [c for c in a.columns if c not in ("submission_id", "task")]
    other = {c: int((~np.isclose(a[c].values, b[c].values, equal_nan=True)).sum()) for c in cols if c != "queue_pred"}
    assert all(v == 0 for v in other.values()), other

    diff = a.queue_pred.values != b.queue_pred.values
    assert (a.task.values[diff] == "queue").all()
    out = base.copy()
    out.loc[diff, "queue_pred"] = b.queue_pred.values[diff]
    changed_vs_base = int((out.queue_pred.values != base.queue_pred.values).sum())

    od = Path(args.out)
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "submission.csv", index=False)
    info = {
        "args": vars(args),
        "diff_cells": int(diff.sum()),
        "changed_vs_base": changed_vs_base,
        "base_agrees_a_on_diff": bool((base.queue_pred.values[diff] == a.queue_pred.values[diff]).all()),
    }
    (od / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
