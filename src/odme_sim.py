import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from odme_scan import REPO
from task4.build_task4_odme_artifacts import load_operator, prior_values, released_prior
from task4.score_task4 import score_panel

FAMILY = lambda p: p.rsplit("_", 1)[0]


def attraction(f, paths):
    a = pd.Series(f, index=paths.destination_zone.to_numpy()).groupby(level=0).sum()
    return a / max(a.sum(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())
    subs = {}
    for c in a.cand:
        name, path = c.split("=", 1)
        subs[name] = pd.read_csv(path) if path != "prior" else None
    rows = []
    for split in ["validation", "private"]:
        for panel in panels:
            path_ids, link_ids, A, paths = load_operator(rel / "corridors" / panel / "network")
            b = prior_values(rel, panel, split, path_ids)
            ab = attraction(b, paths)
            dep = str(released_prior(rel, panel, split).departure_time.iloc[0])
            for name, sub in subs.items():
                if sub is None:
                    s = paths[["path_id", "origin_zone", "destination_zone"]].assign(panel=panel, path_flow=b, departure_time=dep)
                else:
                    s = sub[(sub.panel == panel) & (sub.departure_time == dep)]
                sl = score_panel(panel, rel, split, s)["S_link"]
                f = s.set_index("path_id").reindex(path_ids).path_flow.to_numpy(float)
                rows.append({"split": split, "panel": panel, "cand": name, "S_link": sl, "Dhat_over_sumb": np.abs(f - b).sum() / b.sum(),
                             "sumf_over_sumb": f.sum() / b.sum(), "zero_frac": float((f <= 1e-9).mean()),
                             "attr_L1_vs_prior": float(np.abs(attraction(f, paths) - ab).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out), index=False)
    fam = df.assign(fam=df.panel.map(FAMILY)).groupby(["split", "cand", "fam"]).mean(numeric_only=True).groupby(["split", "cand"]).mean()
    print(fam.round(4).to_string())


if __name__ == "__main__":
    main()
