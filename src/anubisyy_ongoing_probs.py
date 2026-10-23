import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("TFB_DATA_ROOT", str(ROOT / "data" / "kaggle_public"))
os.environ.setdefault("TFB_OUTPUT_ROOT", str(ROOT / "reports" / "queue" / "anubisyy"))
sys.path.insert(0, str(ROOT / "reference" / "anubisyy_assets"))

import lightgbm as lgb
import numpy as np
import pandas as pd
from queue_ongoing import HISTORY, HORIZON, Panel, panel_names


def main():
    data = Path(os.environ["TFB_DATA_ROOT"])
    model = lgb.Booster(model_file=str(ROOT / "reference" / "anubisyy_assets" / "ongoing_lgb.txt"))
    out = []
    for panel_no, name in enumerate(panel_names()):
        panel = Panel(name, panel_no)
        for split in ("validation", "private"):
            folder = data / "task2" / name / split
            windows = pd.read_csv(folder / "window_index.csv")
            history = pd.read_parquet(folder / "window_history.parquet")
            history["link_id"] = history.link_id.astype(str)
            for w in windows[windows.condition == "queue_ongoing"].itertuples(index=False):
                origin = pd.Timestamp(w.forecast_origin)
                times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY, freq="5min", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
                h = history[history.window_id == w.window_id]
                assert pd.to_datetime(h.timestamp, utc=True).max() < origin
                mat = lambda c: h.pivot(index="timestamp", columns="link_id", values=c).reindex(index=times, columns=panel.links).to_numpy(dtype=np.float32)
                speed, flow, eligible = mat("speed_kmh"), mat("flow_vph"), mat("is_score_eligible")
                X = panel.features(speed, flow, np.nan_to_num(eligible, nan=0).astype(bool), origin, w.condition)
                p = model.predict(X, num_threads=4)
                out.append(pd.DataFrame({"window_id": w.window_id, "h": np.repeat(np.arange(1, HORIZON + 1), len(panel.links)),
                                         "link_id": np.tile(panel.links, HORIZON), "p_pub": p}))
        print(name, flush=True)
    r = pd.concat(out, ignore_index=True)
    dst = Path(os.environ["TFB_OUTPUT_ROOT"]) / "ongoing_probs.parquet"
    r.to_parquet(dst)
    print(len(r), "rows ->", dst, "share>=0.4", round(float((r.p_pub >= 0.4).mean()), 4))


if __name__ == "__main__":
    main()
