import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import task2_diag as td
import task2_model as tm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--thr-onset", type=float, default=0.3)
    ap.add_argument("--thr-ongoing", type=float, default=0.35)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    on, og, meta = [], [], {}
    for pid, panel in enumerate(tm.PANELS):
        P = td.load_panel(rel, pid, panel)
        days = list(range(len(P["dates"])))
        full, by_m = td.month_freqs(P, days, True)
        o, g = td.gen(P, pid, "hybrid", rng, 2, days, lambda dt: by_m[dt[:7]])
        on += o
        og += g
        meta[panel] = {"order": P["order"], "vf": P["vf"], "cap": P["cap"], "lanes": P["lanes"], "freq": full}
        tm.FREQ[pid] = full
        print(panel, flush=True)
    on, og = pd.concat(on, ignore_index=True), pd.concat(og, ignore_index=True)
    m_on, c_on = td.fit(on, td.FREQ_COLS, a.rounds)
    m_og, c_og = td.fit(og, td.FREQ_COLS, a.rounds)
    for split in ["validation", "private"]:
        sub, _ = tm.predict_windows(tm.official_windows(rel, split, meta, "mainline_states_masked"), m_on, c_on, m_og, c_og, a.thr_onset, a.thr_ongoing)
        sub.to_csv(out / f"queue_{split}.csv", index=False)
        print(split, len(sub), round(sub.queue_pred.mean(), 4), flush=True)
    (out / "config.json").write_text(json.dumps({"variant": "hybrid+lomo+no_freq", "thr_onset": a.thr_onset, "thr_ongoing": a.thr_ongoing, "rounds": a.rounds}))


if __name__ == "__main__":
    main()
