import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TFB_DATA_ROOT", str(ROOT / "data" / "kaggle_public"))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(ROOT / "reports" / "queue" / "anubisyy"))
sys.path.insert(0, str(ROOT / "reference" / "anubisyy_assets"))

import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from queue_ongoing import HISTORY, HORIZON, Panel, panel_names

sys.path.insert(0, str(ROOT / "src"))
import task2_model as tm


def shifted(panel, model, speed, flow, obs, origin, day, adj):
    t0 = origin.hour * 12 + origin.minute // 5
    vT, qT = day[0, t0].astype(np.float32), day[1, t0].astype(np.float32)
    ok = np.isfinite(vT)
    dv = np.where(ok, vT - speed[-1], np.nan)
    nb = np.nan_to_num(panel.neighboring(adj, dv), nan=0.0)
    vT = np.where(ok, vT, np.clip(speed[-1] + np.clip(nb, -15.0, 15.0), 5.0, 130.0))
    qT = np.where(np.isfinite(qT), qT, flow[-1])
    X = panel.features(np.vstack([speed[1:], vT[None]]), np.vstack([flow[1:], qT[None]]),
                       np.vstack([obs[1:], np.ones((1, len(panel.links)), bool)]), origin + pd.Timedelta(minutes=5), "queue_ongoing")
    p = model.predict(X, num_threads=4).reshape(HORIZON, len(panel.links))
    return np.vstack([p[0:1], p[0:HORIZON - 1]]).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift-w", type=float, default=0.0)
    ap.add_argument("--out-name", default="ongoing_probs.parquet")
    a = ap.parse_args()
    data = Path(os.environ["TFB_DATA_ROOT"])
    model = lgb.Booster(model_file=str(ROOT / "reference" / "anubisyy_assets" / "ongoing_lgb.txt"))
    out = []
    for panel_no, name in enumerate(panel_names()):
        panel = Panel(name, panel_no)
        adj = panel.incoming + panel.outgoing
        for split in ("validation", "private"):
            folder = data / "task2" / name / split
            windows = pd.read_csv(folder / "window_index.csv")
            history = pd.read_parquet(folder / "window_history.parquet")
            history["link_id"] = history.link_id.astype(str)
            if a.shift_w:
                sd, SA = tm.load_days(data, name, split, "mainline_states_masked", panel.links)
                di = {d: i for i, d in enumerate(sd)}
            for w in windows[windows.condition == "queue_ongoing"].itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = history[history.window_id == w.window_id]
                assert pd.to_datetime(h.timestamp, utc=True).max() < origin
                mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(dtype=np.float32)
                speed, flow, eligible = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
                obs = np.nan_to_num(eligible, nan=0).astype(bool)
                X = panel.features(speed, flow, obs, origin, w.condition)
                p = model.predict(X, num_threads=4)
                if a.shift_w:
                    p = (1 - a.shift_w) * p + a.shift_w * shifted(panel, model, speed, flow, obs, origin, SA[di[str(origin.date())]], adj)
                out.append(pd.DataFrame({"window_id": w.window_id, "h": np.repeat(np.arange(1, HORIZON + 1), len(panel.links)),
                                         "link_id": np.tile(panel.links, HORIZON), "p_pub": p}))
        print(name, flush=True)
    r = pd.concat(out, ignore_index=True)
    dst = Path(os.environ["TFB_OUTPUT_ROOT"]) / a.out_name
    r.to_parquet(dst)
    print(len(r), "rows ->", dst, "share>=0.4", round(float((r.p_pub >= 0.4).mean()), 4))


if __name__ == "__main__":
    main()
