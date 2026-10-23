import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(path):
    return pd.read_csv(path, dtype={"task": str}).sort_values("submission_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--state-a", required=True)
    ap.add_argument("--state-b", required=True)
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, A, B = load(a.base), load(a.state_a), load(a.state_b)
    assert (base.submission_id.values == A.submission_id.values).all() and (A.submission_id.values == B.submission_id.values).all()
    st = (base.task == "state").to_numpy()
    out = base.copy()
    for c in ("speed_kmh", "flow_vph"):
        out.loc[st, c] = (1 - a.w) * A.loc[st, c] + a.w * B.loc[st, c]
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    out.to_csv(od / "submission.csv", index=False)
    info = {"args": vars(a),
            "rmse_speed_vs_base": float(np.sqrt(((out.loc[st, "speed_kmh"] - base.loc[st, "speed_kmh"]) ** 2).mean())),
            "rmse_flow_vs_base": float(np.sqrt(((out.loc[st, "flow_vph"] - base.loc[st, "flow_vph"]) ** 2).mean())),
            "non_state_identical": bool((out.loc[~st].fillna(-1).values == base.loc[~st].fillna(-1).values).all())}
    (od / "blend.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
