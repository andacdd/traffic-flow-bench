from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


md(r"""
# TrafficFlowBench: a guided EDA of all ten corridors

This notebook is a descriptive tour of the **2026 IEEE Big Data – Traffic Flow Bench** data. It does not train a model and it does not make a submission. The goal is to understand what the files contain, how clean they are, and what the traffic in them looks like, so that any modelling choice you make later rests on facts about the data.

> ### Key findings at a glance
>
> 1. **The grid is clean.** Every corridor is a complete 5-minute grid with no duplicate rows, and the quality flags follow the documented rules exactly.
> 2. **Detector availability is identical everywhere.** About 76.4 % of cells are score-eligible in all ten corridors. A further ~7 % hold real values below the 75 % eligibility line: they are not scored but can still be used as input.
> 3. **Two value ceilings.** Speed is clipped at 120.75 km/h (1–18 % of observed cells sit exactly on it). Occupancy saturates at 0.75 in heavy congestion.
> 4. **Validation and private have 90-minute full-width gaps.** After each Task 2 origin `T`, every link is hidden for T+5 … T+90, on top of the Task 1 mask.
> 5. **Free flow is a straight line.** On the free-flow branch, flow vs density fits a line through the origin with R² = 0.987–0.999. The slope equals the free speed in `links.csv`.
> 6. **Congestion is an afternoon event.** Where corridors queue, it is mainly between 13:00 and 20:00. The queued share ranges from 0.4 % (`D7_I10_W`) to 9.7 % (`D7_I405_N`) of link-steps.
> 7. **Queues live at fixed bottlenecks.** A few links per corridor hold most queued time; up to 111 of 297 links (`D7_I210_E`) never queue at all.
> 8. **Months are not alike.** The monthly queued share swings without a trend, for example from 5 % to 14 % on `D7_I405_N`.
> 9. **Two network files disagree.** `lwr_mainline_topology.csv` lists free speeds of about 105 km/h, while `links.csv` and `fd_parameters.csv` list 113–118 km/h.
> 10. **Ramps are small.** Median ramp flows are 13–86 veh/h, against a median of about 4,600 veh/h on a `D12_I5_N` mainline link.

**Contents**

1. The four tasks and how they are scored
2. Data layout: files, splits, the 5-minute grid and the quality flags
3. Missingness and data quality
4. Traffic physics: daily profiles, speed heatmaps, the fundamental diagram
5. Queues: where and when the corridors congest, and how Task 2 windows are laid out
6. Ramps and the conservation identity behind Task 3
7. The Task 4 network: paths, OD pairs and link counts
8. Summary

**How the notebook stays light.** The full release is about 12 GB. Corridor-wide summaries read every third training day (91 of 273 days) and only the columns they need, one corridor at a time. One corridor, `D12_I5_N`, is read in full for the deep-dive plots. Everything runs on the free CPU machine without internet.
""")

code(r"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
})
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#52514e"
T0 = time.time()


def headline(fig, text):
    fig.suptitle(text, x=0.01, ha="left", fontsize=13, fontweight="bold", color="#0b0b0b")


def subtitle(ax, text, pad=6):
    ax.set_title(text, loc="left", fontsize=9, color=INK, pad=pad)


def find_root():
    env = os.environ.get("TFB_ROOT")
    if env and Path(env, "config", "corridors.json").exists():
        return Path(env)
    for depth in range(1, 5):
        pattern = "/kaggle/input/" + "/".join(["*"] * depth) + "/config/corridors.json"
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0]).parent.parent
    local = Path("../../data/kaggle_public")
    if Path(local, "config", "corridors.json").exists():
        return local.resolve()
    raise FileNotFoundError("kaggle_public release not found; set TFB_ROOT")


ROOT = find_root()
CORR = json.load(open(ROOT / "config" / "corridors.json"))
CONTRACT = json.load(open(ROOT / "config" / "competition_contract.json"))
PANELS = [p["corridor_id"] for p in CORR["panels"]]
FAMILY = {p["corridor_id"]: p["family_id"] for p in CORR["panels"]}
DEEP = "D12_I5_N"
print("release root:", ROOT)
print("panels:", PANELS)
""")

code(r"""
def day_files(panel, split, sub):
    return sorted(glob.glob(str(ROOT / "corridors" / panel / split / sub / "*" / "*.parquet")))


def read_files(files, cols):
    parts = []
    for f in files:
        pf = pq.ParquetFile(f)
        use = [c for c in cols if c in pf.schema_arrow.names]
        t = pf.read(columns=use).to_pandas()
        for c in t.columns:
            if t[c].dtype == "float64":
                t[c] = t[c].astype("float32")
        if "mask_regime" in cols:
            t["mask_regime"] = Path(f).parent.name.split("=")[1]
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def net(panel, name):
    return pd.read_csv(ROOT / "corridors" / panel / "network" / name)


def add_time(df):
    ts = pd.to_datetime(df["timestamp"], format="%Y-%m-%dT%H:%M:%SZ")
    df["date"] = ts.dt.normalize()
    df["hour"] = ts.dt.hour.astype("int8")
    df["tod"] = (ts.dt.hour + ts.dt.minute / 60).astype("float32")
    df["dow"] = ts.dt.dayofweek.astype("int8")
    df["month"] = (ts.dt.year * 100 + ts.dt.month).astype("int32")
    return df
""")

md(r"""
## 1. The four tasks and how they are scored

The competition bundles four tasks over the same ten directional freeway corridors (five freeways × two directions). All four are submitted in **one CSV**, and the total score is a fixed weighted sum:

$$S_{total} = 0.35\,S_{state} + 0.30\,S_{queue} + 0.15\,S_{physics} + 0.20\,S_{ODME}$$

| Task | What you produce | Scored on |
|---|---|---|
| 1 – State reconstruction | speed and flow for cells that were blanked by a mask | RMSE of speed (scale 25 km/h) and per-lane flow (scale 600 veh/h), per mask regime |
| 2 – Queue forecasting | a binary queue flag per link for the next 30 minutes after an origin `T` | space–time IoU per window |
| 3 – Physics consistency | nothing new: it is computed from your Task 1 answer | fundamental-diagram validity + vehicle conservation |
| 4 – OD / path flows | one flow per path | OD error, link-count reproduction, deviation from a weak prior, destination attraction |

The configuration file `config/competition_contract.json` carries the exact numbers. The cell below shows the parts that matter for EDA.

> **Note on Task 3.** Task 3 has no submission of its own. The organizers compute it from the speeds and flows you submit for Task 1, so the only way to change `S_physics` is a better Task 1 answer.
""")

code(r"""
from IPython.display import display

tw = CONTRACT["overall_score"]["task_weights"]
agg = CONTRACT["overall_score"]["aggregation"]
st = CONTRACT["tasks"]["state"]
qu = CONTRACT["tasks"]["queue"]
formula = {
    "state": f"mean over R1/R2/R3 of {st['speed_weight']}·max(0, 1 − RMSE_v/{st['speed_normalizer']:g}) + {st['flow_weight']}·max(0, 1 − RMSE_q_lane/{st['flow_normalizer']:g})",
    "queue": "space–time IoU per window, onset and ongoing weighted equally",
    "physics": CONTRACT["tasks"]["physics"]["formula"],
    "odme": CONTRACT["tasks"]["odme"]["formula"],
}
display(pd.DataFrame({"weight": tw, "score": formula, "aggregation": {k: agg[k] for k in tw}}).style.format({"weight": "{:.2f}"}).set_properties(**{"text-align": "left"}))

params = pd.DataFrame([
    ("Task 1", "mask rates", ", ".join(f"{k} = {v:.0%}" for k, v in st["masking_regimes"].items())),
    ("Task 1", "flow units", st["flow_units"]),
    ("Task 2", "history / horizon / step", f"{qu['history_minutes']} / {qu['forecast_horizon_minutes']} / {qu['step_minutes']} min"),
    ("Task 2", "windows per panel and split", ", ".join(f"{k}: {v}" for k, v in qu["windows_per_panel_per_split"].items())),
    ("Task 2", "minimum spacing of origins", f"{qu['minimum_spacing_minutes']} min"),
    ("Task 2", "excluded panels", ", ".join(qu["excluded_panels"])),
    ("all", "eligibility", CONTRACT["data_quality"]["is_score_eligible_definition"]),
], columns=["task", "parameter", "value"])
display(params.style.hide(axis="index").set_properties(**{"text-align": "left"}))
""")

md(r"""
**Reading the aggregation rule.** Every step is an equal mean: directions are averaged inside a freeway family, then the families are averaged. A 44-link corridor therefore counts exactly as much as a 352-link one. Keep this in mind when you look at corridor-level numbers below: a problem on a small corridor is not diluted.

Two further points from the scoring spec are worth knowing before looking at data:

* **Only score-eligible cells are scored** (`pct_observed >= 75` and the value is present).
* Task 2 is scored on 8 panels only; the two `D12_I405` panels are excluded from Task 2 but stay in Tasks 1, 3 and 4.
""")

md(r"""
## 2. Data layout

### 2.1 Files on disk

Each corridor folder holds a `network/` folder (static description of the road) and one folder per split. Only `train` ships the unmasked observations (`mainline_states`); every split ships a masked copy (`mainline_states_masked`, partitioned by mask regime) and ramp observations. The `task1/`, `task2/`, `task4/` folders hold submission templates and task-specific inputs.
""")

code(r"""
def folder_size(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e9


rows = []
for top in ["corridors", "task1", "task2", "task4"]:
    rows.append({"folder": top, "GB": folder_size(ROOT / top), "files": sum(1 for f in (ROOT / top).rglob("*") if f.is_file())})
for f in ["submission_key.csv", "sample_submission.csv"]:
    rows.append({"folder": f, "GB": (ROOT / f).stat().st_size / 1e9, "files": 1})
sizes = pd.DataFrame(rows)
print(sizes.round(3).to_string(index=False))

rows = []
for split in ["train", "validation", "private"]:
    for sub in ["mainline_states", "mainline_states_masked", "ramp_states"]:
        n = sum(len(day_files(p, split, sub)) for p in PANELS)
        sz = sum(folder_size(ROOT / "corridors" / p / split / sub) for p in PANELS if (ROOT / "corridors" / p / split / sub).exists())
        rows.append({"split": split, "layer": sub, "day files (10 panels)": n, "GB": round(sz, 2)})
print()
print(pd.DataFrame(rows).to_string(index=False))
""")

md(r"""
### 2.2 Splits

The calendar is synthetic (the years mean nothing). `train` covers nine months, `validation` (public leaderboard) and `private` (final leaderboard) one month each. The official docs state that validation and private were generated independently, with their own demand draws and incident schedules, so neither is a sample of the other.
""")

code(r"""
sp = pd.DataFrame({k: {kk: v[kk] for kk in ["start", "end", "days", "labels"]} for k, v in CONTRACT["split"].items() if isinstance(v, dict) and "start" in v}).T
print(sp.to_string())
""")

md(r"""
### 2.3 The ten corridors

The cell below builds one row per corridor from the static network files: number of mainline links, total length, lanes, ramps, detectors, paths, and Task 2 windows. This is the table to come back to whenever a corridor looks unusual later.
""")

code(r"""
rows = []
for p in PANELS:
    links = net(p, "links.csv")
    topo = net(p, "lwr_mainline_topology.csv")
    ramps = net(p, "ramp_attachment_map.csv")
    paths = net(p, "path_set.csv")
    wfile = ROOT / "task2" / p / "validation" / "window_index.csv"
    rows.append({
        "panel": p,
        "family": FAMILY[p],
        "links": len(links),
        "length_km": links.length_km.sum(),
        "median_link_m": links.length_km.median() * 1000,
        "lanes_mean": links.lanes.mean(),
        "sensor_links": int(topo.has_sensor.sum()),
        "on_ramps": int((ramps.ramp_type == "OR").sum()),
        "off_ramps": int((ramps.ramp_type == "FR").sum()),
        "paths": len(paths),
        "zones": pd.concat([paths.origin_zone, paths.destination_zone]).nunique(),
        "task2_windows_val": len(pd.read_csv(wfile)) if wfile.exists() else 0,
        "train_days": len(day_files(p, "train", "mainline_states")),
    })
corr = pd.DataFrame(rows).set_index("panel")
corr.round(1)
""")

md(r"""
Corridor sizes differ by a factor of eight (44 to 352 links), while the total road length differs much less: the long corridors are split into many short links. Many links are only tens of metres long, which matters for anything computed per link and per 5 minutes (a car at 100 km/h crosses a 50 m link in under two seconds).

### 2.4 One day file

A mainline day file is a full grid: every link × every 5-minute step of the day. Here are its columns and a few rows.
""")

code(r"""
f0 = day_files(DEEP, "train", "mainline_states")[0]
d0 = pq.ParquetFile(f0).read().to_pandas()
print(Path(f0).relative_to(ROOT), d0.shape)
print(d0.dtypes.to_string())
d0.head()
""")

md(r"""
Key columns:

* `speed_kmh`, `flow_vph` – the two scored channels. Flow is the **total** over all lanes of the link.
* `occupancy`, `density_occ_linear_vehpkm` – released but not scored; density here is a fixed linear transform of occupancy.
* `pct_observed` – share of the 5-minute interval a detector actually reported (0–100).
* `is_observed`, `is_imputed`, `is_missing`, `is_score_eligible` – flags derived from `pct_observed`.

### 2.5 Structural checks on every corridor

The next cell is the main data pass. For each corridor it reads every third training day, verifies the grid and the flags, and keeps only small aggregates (distributions, hourly and monthly shares, a 20k-point sample for scatter plots). Raw rows are dropped after each corridor.

Queue status is computed with the definition from the scoring spec, `speed <= 0.60 × free_speed`, using `free_speed_kmh` from `links.csv`. (The spec computes the true label from the noise-free underlying state; on observations the share is an approximation.)
""")

code(r"""
COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed",
        "is_observed", "is_imputed", "is_missing", "is_score_eligible"]
STEP = 3
checks, pct_dist, hour_q, month_q, dow_q, link_q, ff_fit, qk_samples, prof_rows = [], {}, {}, {}, {}, {}, [], {}, []

for p in PANELS:
    t = time.time()
    links = net(p, "links.csv")
    fd = net(p, "fd_parameters.csv")
    topo = net(p, "lwr_mainline_topology.csv")
    files = day_files(p, "train", "mainline_states")[::STEP]
    df = read_files(files, COLS)
    add_time(df)
    df = df.merge(links[["link_id", "lanes", "free_speed_kmh", "capacity_vph"]], on="link_id", how="left")
    df = df.merge(topo[["link_id", "order_index"]], on="link_id", how="left")
    has = df.speed_kmh.notna() & df.flow_vph.notna()
    elig_rule = (df.pct_observed >= 75) & has
    per_day = df.groupby("date").size()
    checks.append({
        "panel": p,
        "days_read": len(files),
        "rows": len(df),
        "dup_ts_link": int(df.duplicated(["timestamp", "link_id"]).sum()),
        "rows_per_day_ok": bool((per_day == len(links) * 288).all()),
        "imputed_flag_ok": bool(((df.pct_observed < 100) == (df.is_imputed == 1)).all()),
        "eligible_flag_ok": bool((elig_rule == (df.is_score_eligible == 1)).all()),
        "pct0_all_null": bool(df.loc[df.pct_observed == 0, "speed_kmh"].isna().all()),
        "pct>0_all_present": bool(has[df.pct_observed > 0].all()),
        "eligible_share": float((df.is_score_eligible == 1).mean()),
        "speed_max": float(df.speed_kmh.max()),
        "share_at_speed_max": float((df.speed_kmh >= df.speed_kmh.max() - 1e-3).mean() / has.mean()),
        "occ_max": float(df.occupancy.max()),
        "share_at_occ_max": float((df.occupancy >= df.occupancy.max() - 1e-4).mean() / has.mean()),
        "flow_over_capacity": float((df.flow_vph > df.capacity_vph)[has].mean()),
    })
    pct_dist[p] = df.pct_observed.value_counts(normalize=True).sort_index()
    v = df[has].copy()
    v["queued"] = v.speed_kmh <= 0.6 * v.free_speed_kmh
    hour_q[p] = v.groupby("hour").queued.mean()
    month_q[p] = v.groupby("month").queued.mean()
    dow_q[p] = v.groupby("dow").queued.mean()
    link_q[p] = v.groupby("order_index").queued.mean()
    e = v[df.loc[has, "is_score_eligible"].values == 1]
    e = e.assign(q_lane=e.flow_vph / e.lanes, k_lane=e.flow_vph / e.lanes / e.speed_kmh)
    kc = (fd.critical_density / fd.lanes).median()
    ff = e[e.k_lane < 0.8 * kc]
    slope = float((ff.q_lane * ff.k_lane).sum() / (ff.k_lane ** 2).sum())
    resid = ff.q_lane - slope * ff.k_lane
    r2 = 1 - float((resid ** 2).sum() / ((ff.q_lane - ff.q_lane.mean()) ** 2).sum())
    ff_fit.append({"panel": p, "fitted_vf": slope, "median_vf_links": float(links.free_speed_kmh.median()), "R2_free_branch": r2,
                   "share_free_branch": len(ff) / len(e), "kc_lane": kc,
                   "cap_lane": float((fd.capacity_vph / fd.lanes).median()), "kjam_lane": float((fd.k_jam / fd.lanes).median())})
    qk_samples[p] = e[["k_lane", "q_lane", "speed_kmh"]].sample(min(20000, len(e)), random_state=0)
    g = e.groupby(["dow", "tod"]).agg(speed=("speed_kmh", "mean"), q_lane=("q_lane", "mean")).reset_index()
    g["panel"] = p
    prof_rows.append(g)
    print(f"{p:11s} {len(files)} days, {len(df):>10,} rows, {time.time() - t:5.1f}s")
    del df, v, e, ff

checks = pd.DataFrame(checks).set_index("panel")
ff_fit = pd.DataFrame(ff_fit).set_index("panel")
prof = pd.concat(prof_rows, ignore_index=True)
checks
""")

md(r"""
What the checks say:

* **The grid is complete and unique.** No duplicate `(timestamp, link_id)` and every day has exactly `links × 288` rows. You never need to reindex.
* **Flags are fully consistent** with the documented rules: `is_imputed = pct_observed < 100`, `is_score_eligible = pct_observed >= 75 and values present`.
* **`pct_observed = 0` means empty**, and every cell with `pct_observed > 0` has both speed and flow. There are no partially filled rows.
* **About 76 % of cells are score-eligible in every corridor.** The docs say detector health is uniform by design, and the numbers agree.
* **Speed is capped** (`speed_max` is identical everywhere) and **occupancy is capped** at a fixed value; see section 3.

## 3. Missingness and data quality

### 3.1 How much of each corridor is observed?
""")

code(r"""
pdist = pd.DataFrame(pct_dist).fillna(0).T
fig, ax = plt.subplots(figsize=(11, 3.6))
left = np.zeros(len(pdist))
bands = [(0, 0, "0 (empty)", "#c9c8c3"), (1, 74, "1–74 (present, not eligible)", C[3]), (75, 99, "75–99 (eligible, imputed)", C[2]), (100, 100, "100 (fully observed)", C[0])]
for lo, hi, lab, col in bands:
    w = pdist.loc[:, [c for c in pdist.columns if lo <= c <= hi]].sum(axis=1).values
    ax.barh(pdist.index, w, left=left, color=col, label=lab, edgecolor="white", linewidth=1)
    left += w
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlim(0, 1)
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
subtitle(ax, "Share of train cells by pct_observed band", pad=30)
headline(fig, "Every corridor has the same detector availability: ~76 % of cells are score-eligible")
plt.tight_layout()
plt.show()
print((pdist * 100).round(2).to_string())
""")

md(r"""
The distribution is the same in every corridor, roughly:

* ~16–17 % of cells are empty (`pct_observed = 0`).
* ~7 % have values but fall below the 75 % eligibility line. **They are not scored, but they are real observations** and can be used as inputs.
* ~13 % are eligible but partly imputed (75–99).
* ~63 % are fully observed.

### 3.2 Mask regimes (Task 1)

On top of the natural gaps, Task 1 blanks extra cells. Each day belongs to one regime (R1/R2/R3 = 20/30/50 % mask rate). The masked files keep `pct_observed` and `is_score_eligible` of the blanked cell, so on every split we can count how many eligible cells were blanked.
""")

code(r"""
rows = []
for p in PANELS:
    for split in ["train", "validation", "private"]:
        files = day_files(p, split, "mainline_states_masked")
        reg = pd.Series([Path(f).parent.name.split("=")[1] for f in files])
        m = read_files(files[::STEP * 3 if split == "train" else STEP], ["speed_kmh", "pct_observed", "is_score_eligible", "mask_regime"])
        g = m.groupby("mask_regime").apply(lambda x: pd.Series({
            "blank_all": x.speed_kmh.isna().mean(),
            "blank_eligible": x.speed_kmh[x.is_score_eligible == 1].isna().mean(),
        }), include_groups=False)
        for r, c in reg.value_counts().items():
            rows.append({"panel": p, "split": split, "regime": r, "days": c,
                         "blank_all": g.loc[r, "blank_all"] if r in g.index else np.nan,
                         "blank_eligible": g.loc[r, "blank_eligible"] if r in g.index else np.nan})
mask = pd.DataFrame(rows)
summary = mask.groupby(["split", "regime"]).agg(days_per_panel=("days", "mean"), blank_all=("blank_all", "mean"), blank_eligible=("blank_eligible", "mean"))
print(summary.round(3).to_string())
""")

md(r"""
* In `train`, the share of **eligible** cells blanked matches the regime rate almost exactly (≈0.20 / 0.30 / 0.50). The mask is drawn only among eligible cells.
* In `validation` and `private` the blank share among eligible cells is a bit higher than the regime rate. The extra blanks come from the Task 2 windows, which hide a block of time on every link (next cell).
* Days are spread across the three regimes in roughly equal numbers in every split.

### 3.3 A validation day, cell by cell

The picture below shows one validation day of `D12_I5_N`. Each row is a link (in road order), each column a 5-minute step. Grey = no data (`pct_observed = 0`), orange = Task 1 mask, dark blue = the block hidden on every link after a Task 2 forecast origin `T`, light = observed.
""")

code(r"""
wi_val = pd.read_csv(ROOT / "task2" / DEEP / "validation" / "window_index.csv")
day = wi_val.date.iloc[0]
vf = [f for f in day_files(DEEP, "validation", "mainline_states_masked") if day.replace("-", "_") in f][0]
vd = read_files([vf], ["timestamp", "link_id", "speed_kmh", "pct_observed", "is_score_eligible", "mask_regime"])
add_time(vd)
topo = net(DEEP, "lwr_mainline_topology.csv")
vd = vd.merge(topo[["link_id", "order_index"]], on="link_id")
code_ = np.where(vd.pct_observed == 0, 0, np.where(vd.speed_kmh.notna(), 3, 1))
blocks = []
for _, w in wi_val[wi_val.date == day].iterrows():
    after = vd[(vd.timestamp > w.forecast_origin) & (vd.pct_observed > 0)].groupby("timestamp").speed_kmh.apply(lambda x: x.isna().all())
    run = after.cumprod().astype(bool)
    hidden = run[run].index
    blocks.append({"window_id": w.window_id, "condition": w.condition, "origin T": w.forecast_origin[11:16],
                   "scored horizon": f"{w.forecast_start[11:16]}-{w.forecast_end[11:16]}",
                   "hidden on all links": f"{hidden.min()[11:16]}-{hidden.max()[11:16]}", "hidden steps": len(hidden)})
    code_ = np.where(vd.timestamp.isin(hidden) & vd.speed_kmh.isna() & (vd.pct_observed > 0), 2, code_)
vd["code"] = code_
grid = vd.pivot_table(index="order_index", columns="tod", values="code")
from matplotlib.colors import ListedColormap
fig, ax = plt.subplots(figsize=(11, 4.2))
from matplotlib.patches import Patch
cols4 = ["#8f8e89", C[1], "#16365c", "#eef3fa"]
ax.imshow(grid.values, aspect="auto", cmap=ListedColormap(cols4), vmin=0, vmax=3,
          extent=[0, 24, grid.index.max(), 0], interpolation="nearest")
ax.set_xlabel("hour of day")
ax.set_ylabel("link (order_index)")
ax.grid(False)
ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols4, ["empty (pct_observed = 0)", "Task 1 mask", "hidden after Task 2 origin", "observed"])],
          ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
subtitle(ax, f"{DEEP}, validation day {day}, mask regime {vd.mask_regime.iloc[0]}", pad=28)
headline(fig, "A validation day: a scattered Task 1 mask plus 90-minute full-width gaps after each Task 2 origin")
plt.tight_layout()
plt.show()
print(pd.DataFrame(blocks).to_string(index=False))
""")

md(r"""
Things to notice:

* Masked cells (orange) are scattered over the whole day. Empty cells (grey) often come as short horizontal streaks, because a detector can stay down for several steps.
* After each Task 2 origin `T`, **every link is hidden for 18 steps (T+5 … T+90)**: the 30-minute scored horizon plus the following hour. On those days Task 1 has full-width gaps of 90 minutes.
* The hour just before `T` (the Task 2 history) carries no Task 1 mask, which shows up as a lighter vertical band.

### 3.4 Value ceilings
""")

code(r"""
print(checks[["speed_max", "share_at_speed_max", "occ_max", "share_at_occ_max", "flow_over_capacity"]].round(4).to_string())
""")

md(r"""
* **Speed is clipped at 120.75 km/h** in every corridor; between 1 % and 18 % of observed cells sit exactly on it. Long runs of identical night-time speeds are this ceiling, not stuck sensors.
* **Occupancy saturates at 0.75** (up to ~5 % of cells on `D7_I405_N`), which happens in heavy congestion. At that point occupancy, and the density derived from it, stops tracking the traffic. The identity `k = q / v` (used by the scorer for Task 3) does not have this ceiling.
* Flow above link capacity is very rare (fractions of a percent).

### 3.5 Two network files describe the same links

`links.csv`, `fd_parameters.csv` and `lwr_mainline_topology.csv` all list mainline links with lanes, free speed and capacity. The first two agree with each other; the topology file uses different values. The table counts how many links differ and by how much.
""")

code(r"""
rows = []
for p in PANELS:
    links = net(p, "links.csv")
    fd = net(p, "fd_parameters.csv")
    topo = net(p, "lwr_mainline_topology.csv")
    a = links.merge(fd, on="link_id", suffixes=("", "_fd")).merge(topo[["link_id", "free_speed_kmh", "capacity_vph", "lanes"]], on="link_id", suffixes=("", "_topo"))
    rows.append({
        "panel": p,
        "links_vs_fd_diff": int(sum((a[c] - a[c + "_fd"]).abs().gt(1e-6).sum() for c in ["free_speed_kmh", "capacity_vph", "lanes", "length_km"])),
        "vf_links_mean": a.free_speed_kmh.mean(),
        "vf_topo_mean": a.free_speed_kmh_topo.mean(),
        "vf_differs": int((a.free_speed_kmh - a.free_speed_kmh_topo).abs().gt(0.05).sum()),
        "cap_differs": int((a.capacity_vph - a.capacity_vph_topo).abs().gt(1).sum()),
        "lanes_differs": int((a.lanes != a.lanes_topo).sum()),
        "links": len(a),
    })
print(pd.DataFrame(rows).set_index("panel").round(1).to_string())
""")

md(r"""
`links.csv` and `fd_parameters.csv` are identical on their shared columns. `lwr_mainline_topology.csv` carries rounder, lower free speeds and different capacities (and sometimes lane counts) for most links. The scoring spec names `fd_parameters.csv` as the source of lane counts for the per-lane flow RMSE; for anything else, check which file the parameter you need should come from.

## 4. Traffic physics

### 4.1 Daily profiles

Average speed and per-lane flow by time of day, one line per weekday, for the deep-dive corridor (from the full train set of `D12_I5_N`).
""")

code(r"""
t = time.time()
deep = read_files(day_files(DEEP, "train", "mainline_states"), ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "is_score_eligible"])
add_time(deep)
dl = net(DEEP, "links.csv")
topo = net(DEEP, "lwr_mainline_topology.csv")
deep = deep.merge(dl[["link_id", "lanes", "free_speed_kmh", "length_km"]], on="link_id").merge(topo[["link_id", "order_index"]], on="link_id")
deep["q_lane"] = deep.flow_vph / deep.lanes
deep["queued"] = deep.speed_kmh <= 0.6 * deep.free_speed_kmh
print(f"{DEEP}: {len(deep):,} rows, {deep.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory, {time.time() - t:.0f}s")

names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
cols7 = [C[0], C[2], C[6], C[4], C[3], C[1], C[7]]
g = deep[deep.speed_kmh.notna()].groupby(["dow", "tod"]).agg(speed=("speed_kmh", "mean"), q=("q_lane", "mean")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for d, gg in g.groupby("dow"):
    ax[0].plot(gg.tod, gg.q, color=cols7[d], lw=1.6, label=names[d])
    ax[1].plot(gg.tod, gg.speed, color=cols7[d], lw=1.6)
subtitle(ax[0], f"{DEEP}: mean flow per lane (veh/h)")
subtitle(ax[1], f"{DEEP}: mean speed (km/h)")
headline(fig, "Weekend traffic starts about two hours later, and its afternoon slowdown is deeper")
for a in ax:
    a.set_xlabel("hour of day")
    a.set_xticks(range(0, 25, 3))
ax[0].legend(ncol=2)
plt.tight_layout()
plt.show()
""")

md(r"""
* Weekdays look alike. On Saturday and Sunday the morning flow rises about two hours later and stays higher late in the evening.
* On this corridor the afternoon speed dip is actually **deeper at the weekend** (to about 92 km/h vs about 100 km/h on weekdays).
* Averaged over all links the dip looks shallow. It is the mean of a few links that slow down a lot and many that do not (section 4.2).

The same speed profile for all ten corridors, weekdays vs weekend:
""")

code(r"""
fig, axes = plt.subplots(2, 5, figsize=(15, 5.2), sharex=True, sharey=True)
for a, p in zip(axes.flat, PANELS):
    gp = prof[prof.panel == p]
    wk = gp[gp.dow < 5].groupby("tod").speed.mean()
    we = gp[gp.dow >= 5].groupby("tod").speed.mean()
    a.plot(wk.index, wk.values, color=C[0], lw=1.5, label="weekday")
    a.plot(we.index, we.values, color=C[1], lw=1.5, label="weekend")
    a.set_title(p, fontsize=9)
    a.set_xticks(range(0, 25, 6))
axes[0, 0].legend()
fig.supylabel("mean speed (km/h)", fontsize=9)
fig.supxlabel("hour of day", fontsize=9)
headline(fig, "The afternoon slowdown is the same on weekdays and weekends; weekday-only dips are in the morning")
plt.tight_layout()
plt.show()
""")

md(r"""
* Corridors differ a lot. `D7_I10_W` and `D7_I210_E` barely slow down on average, while `D7_I405_N` and `D12_I5_S` have a deep afternoon trough.
* The afternoon dip looks much the same on weekdays and weekends. The visible weekday-only features are short **morning** dips (`D7_I10_E`, `D7_I210_W`, `D7_I405_S`).

### 4.2 Space–time speed heatmap

Rows are links in **road order** (`order_index` from `lwr_mainline_topology.csv`), columns are time of day. Using road order rather than `link_id` order is essential: link IDs are not sorted along the road. Shown: the slowest day of the train set and a typical weekday.
""")

code(r"""
dmean = deep.groupby("date").speed_kmh.mean()
slow_day = dmean.idxmin()
typ_day = (dmean - dmean.median()).abs().idxmin()
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for a, dd, lab in [(ax[0], slow_day, "slowest day"), (ax[1], typ_day, "median day")]:
    hm = deep[deep.date == dd].pivot_table(index="order_index", columns="tod", values="speed_kmh")
    im = a.imshow(hm.values, aspect="auto", cmap="Reds_r", vmin=10, vmax=120, extent=[0, 24, hm.index.max(), 0], interpolation="nearest")
    subtitle(a, f"{DEEP}, {dd.date()} ({lab})")
    a.set_xlabel("hour of day")
    a.grid(False)
ax[0].set_ylabel("link (road order)")
fig.colorbar(im, ax=ax, label="speed (km/h)", fraction=0.02)
headline(fig, "Queues start at fixed bottlenecks and grow upstream, on slow and typical days alike")
plt.show()
""")

md(r"""
White gaps are missing cells. Congestion appears as **dark bands that start at a fixed location and grow upstream** (towards lower `order_index` here) over time: the textbook shape of a queue behind a bottleneck. The bands sit at the same links on different days.

### 4.3 The fundamental diagram

Flow, density and speed are tied by `q = k · v`. Plotting per-lane flow against per-lane density (`k = q / v`) gives the fundamental diagram: a rising, almost straight **free-flow branch** (slope = free speed) up to capacity, and a falling **congested branch** beyond it. The dashed line is the triangular diagram implied by the median `fd_parameters.csv` values (free speed, per-lane capacity, per-lane jam density).
""")

code(r"""
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharex=True, sharey=True)
for a, p in zip(axes.flat, PANELS):
    s = qk_samples[p]
    a.scatter(s.k_lane, s.q_lane, s=1, alpha=0.12, color=C[0], rasterized=True)
    r = ff_fit.loc[p]
    kc = r.cap_lane / r.median_vf_links
    a.plot([0, kc, r.kjam_lane], [0, r.cap_lane, 0], ls="--", color=C[1], lw=1.2)
    a.set_title(p, fontsize=9)
    a.set_xlim(0, 110)
    a.set_ylim(0, 2600)
fig.supxlabel("density k (veh/km/lane)", fontsize=9)
fig.supylabel("flow q (veh/h/lane)", fontsize=9)
headline(fig, "Free flow is a straight line at the free speed; congested states form separate clusters")
plt.tight_layout()
plt.show()
print(ff_fit[["fitted_vf", "median_vf_links", "R2_free_branch", "share_free_branch", "cap_lane", "kc_lane", "kjam_lane"]].round(3).to_string())
""")

md(r"""
* The **free-flow branch is nearly a perfect line** (R² ≈ 1) whose slope matches the free speed in `links.csv`. Most of the data (the `share_free_branch` column) sits on it.
* The congested points do not follow one clean falling line. They form **tight, separate clusters** around the falling branch, one cluster per recurring congested state. Some corridors have only a handful of them (`D12_I405_N`, `D7_I210_E`). These few cells carry most of the difficulty of Tasks 1 and 2.
* The share of eligible cells on the free-flow branch ranges from 79 % (`D7_I405_N`) to 97 %.
* Per-lane parameters are similar across corridors (capacity ≈ 1900–2200 veh/h/lane, jam density 100 veh/km/lane), but the lane count changes from link to link.

## 5. Queues

A link is queued at time `t` when `speed <= 0.60 × free_speed`. This is the threshold in the scoring spec. Here it is applied to observed speeds (with `links.csv` free speed), so it is an approximation of the organizer's label.

### 5.1 When do corridors congest?
""")

code(r"""
hq = pd.DataFrame(hour_q).T
tot = pd.Series({p: float((hour_q[p] * 1).mean()) for p in PANELS})
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(hq.values, aspect="auto", cmap="Blues", vmin=0)
ax.set_yticks(range(len(hq)), hq.index)
ax.set_xticks(range(24), range(24))
ax.set_xlabel("hour of day")
ax.grid(False)
for i in range(hq.shape[0]):
    for j in range(hq.shape[1]):
        val = hq.values[i, j]
        if val >= 0.05:
            ax.text(j, i, f"{val * 100:.0f}", ha="center", va="center", fontsize=7, color="white" if val > hq.values.max() * 0.55 else "#0b0b0b")
fig.colorbar(im, ax=ax, label="share of link-steps queued", fraction=0.02)
subtitle(ax, "Share of link-steps queued by corridor and hour (train sample; numbers in %, shown when >= 5 %)")
headline(fig, "Congestion is concentrated between 13:00 and 20:00; D7_I405_N is hit hardest")
plt.tight_layout()
plt.show()
""")

md(r"""
* Where corridors congest, it is an **afternoon and evening** phenomenon, roughly 13:00–20:00. `D7_I10_W` and `D7_I210_E` hardly queue at all. `D7_I405_N` peaks latest and longest (up to 40 % of link-steps around 17:00).
* Morning queues are weak; only `D7_I210_W` shows some from late morning.
* Nights are free-flowing everywhere.

### 5.2 Where do they congest?

Queued share per link along the road, all corridors. Peaks are recurring bottlenecks.
""")

code(r"""
fig, axes = plt.subplots(2, 5, figsize=(15, 5.4), sharey=True)
for a, p in zip(axes.flat, PANELS):
    lq = link_q[p]
    a.fill_between(lq.index, lq.values, color=C[0], alpha=0.25, lw=0)
    a.plot(lq.index, lq.values, color=C[0], lw=1.2)
    a.set_title(p, fontsize=9)
    a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
fig.supxlabel("link (road order)", fontsize=9)
fig.supylabel("share of time queued", fontsize=9)
headline(fig, "Queueing is concentrated on a few bottleneck links in each corridor")
plt.tight_layout()
plt.show()

top = []
for p in PANELS:
    lq = link_q[p]
    top.append({"panel": p, "overall_queued": float(np.average(lq.values)), "max_link": float(lq.max()),
                "links_>10%": int((lq > 0.10).sum()), "links_never": int((lq == 0).sum()), "links": len(lq)})
print(pd.DataFrame(top).set_index("panel").round(3).to_string())
""")

md(r"""
Congestion is **local and persistent**. In every corridor a small number of links accounts for most queued time, and many links never queue at all. A queue starts at a bottleneck and spreads upstream, so the curve usually has a sharp downstream edge and a gentler upstream slope.

### 5.3 Does the amount of congestion change from month to month?
""")

code(r"""
mq = pd.DataFrame(month_q)
fig, axes = plt.subplots(2, 5, figsize=(15, 5), sharex=True)
for a, p in zip(axes.flat, PANELS):
    a.plot(range(len(mq)), mq[p].values, color=C[0], marker="o", ms=4, lw=1.5)
    a.axhline(mq[p].mean(), color=INK, lw=0.8, ls=":")
    a.set_title(p, fontsize=9)
    a.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    a.set_ylim(0, None)
for a in axes[1]:
    a.set_xticks(range(len(mq)), [f"{m // 100 % 100:02d}-{m % 100:02d}" for m in mq.index], rotation=60)
fig.supylabel("queued share", fontsize=9)
headline(fig, "The monthly queued share jumps around with no trend in most corridors")
plt.tight_layout()
plt.show()
print((mq * 100).round(2).T.to_string())
cv = (mq.std() / mq.mean()).round(2)
print("\ncoefficient of variation across months:\n", cv.to_string())
""")

md(r"""
* In most corridors the queued share moves a lot from month to month, with no smooth trend or season. Examples: `D7_I405_N` ranges from 5 % to 14 %, `D12_I5_N` from 2.2 % to 5.9 %, and `D12_I5_S` jumps from 3 % in the first month to 8 % in the second.
* A few corridors are stable (`D7_I210_E`, `D12_I405_N`, `D7_I210_W`).
* Months behave more like **independent draws** of demand and incidents than like a seasonal series. This fits the docs, which say validation and private are separate draws. A statistic measured on one month is not guaranteed to hold in another.

### 5.4 Weekday pattern
""")

code(r"""
dq = pd.DataFrame(dow_q).T
dq.columns = names
print((dq * 100).round(2).to_string())
""")

md(r"""
Unlike speed, there is no common weekday pattern in queueing: the busiest day differs by corridor, and weekends are not systematically lighter. With only ~13 sampled days per weekday, part of this is noise.

### 5.5 How the Task 2 windows are laid out

Task 2 gives, for each window, 60 minutes of history before an origin `T` and asks for queue flags on `T+5 … T+30`. `window_index.csv` describes the windows and `window_history.parquet` holds the history rows. Each corridor has 5 `queue_onset` and 5 `queue_ongoing` windows per split.
""")

code(r"""
wins = []
for p in PANELS:
    for split in ["train", "validation", "private"]:
        f = ROOT / "task2" / p / split / "window_index.csv"
        if f.exists():
            wins.append(pd.read_csv(f))
wins = pd.concat(wins, ignore_index=True)
wins["origin"] = pd.to_datetime(wins.forecast_origin)
wins["origin_hour"] = wins.origin.dt.hour + wins.origin.dt.minute / 60
print(wins.groupby(["split", "condition"]).agg(windows=("window_id", "size"), panels=("panel", "nunique"), days=("date", "nunique"),
      median_origin_hour=("origin_hour", "median"), history_cov=("history_coverage", "mean"), future_cov=("future_coverage", "mean")).round(3).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
bins = np.arange(0, 25, 1)
for i, (c, gg) in enumerate(wins.groupby("condition")):
    ax[0].hist(gg.origin_hour, bins=bins, color=C[i], alpha=0.75, label=c, edgecolor="white")
ax[0].set_xlabel("forecast origin hour")
ax[0].set_ylabel("windows (all splits)")
ax[0].legend()
gap = wins.sort_values("origin").groupby(["panel", "split", "condition"]).origin.diff().dt.total_seconds().div(3600).dropna()
ax[1].hist(gap, bins=40, color=C[0], edgecolor="white")
ax[1].set_xlabel("hours between consecutive origins (same panel, split, condition)")
headline(fig, "Onset windows start in the morning and around midday, ongoing windows in the afternoon")
plt.tight_layout()
plt.show()
print("minimum spacing between origins (h):", gap.min())
""")

code(r"""
wh = pd.read_parquet(ROOT / "task2" / DEEP / "validation" / "window_history.parquet")
w0 = wh.groupby("window_id").agg(rows=("link_id", "size"), steps=("timestamp", "nunique"), first=("timestamp", "min"), last=("timestamp", "max"),
                                 speed_present=("speed_kmh", lambda s: s.notna().mean()), eligible=("is_score_eligible", "mean"))
print(w0.round(3).head(10).to_string())
""")

md(r"""
* The two conditions sit at different times of day. `queue_onset` origins are mostly in the morning and around midday (median ≈ 9–10 h). `queue_ongoing` origins are in the afternoon and evening (median ≈ 15–16 h), when queues already exist.
* Origins of the same condition are at least 6 hours apart (the contract's `minimum_spacing_minutes = 360`); most are about a day apart.
* Each history has 12 steps (`T-60 … T-5`) for every mainline link. About three quarters of history cells have values, the same detector-availability rate as everywhere else.
* The row at `T` itself is not in `window_history.parquet`; the scoring spec allows using the released row at `T` from the masked mainline file.
* The windows are few (10 per corridor and split), so studying queues on the train mainline data, as in 5.1–5.3, gives far more examples than the windows alone.

## 6. Ramps and the conservation identity (Task 3)

### 6.1 Ramp observations

On-ramps (`OR`) add vehicles to the mainline and off-ramps (`FR`) remove them. Ramp files have the same quality flags as mainline files, but only flow.
""")

code(r"""
rf = day_files(DEEP, "train", "ramp_states")[::STEP]
ramp = read_files(rf, ["timestamp", "ramp_link_id", "ramp_type", "flow_vph", "pct_observed", "is_score_eligible"])
add_time(ramp)
print(f"{DEEP} ramp sample: {len(ramp):,} rows, {ramp.ramp_link_id.nunique()} ramps")
print("eligible share:", round((ramp.is_score_eligible == 1).mean(), 3))
print(ramp.groupby("ramp_type").flow_vph.describe().round(1).to_string())
print("\nmainline flow (veh/h, total):")
print(deep.flow_vph.describe().round(1).to_string())

rows = []
for p in PANELS:
    r = read_files(day_files(p, "train", "ramp_states")[::30], ["ramp_type", "flow_vph", "is_score_eligible"])
    rows.append({"panel": p, "ramps": len(net(p, "ramp_attachment_map.csv")), "eligible": (r.is_score_eligible == 1).mean(),
                 "OR_median_vph": r[r.ramp_type == "OR"].flow_vph.median(), "FR_median_vph": r[r.ramp_type == "FR"].flow_vph.median(),
                 "OR_p99_vph": r[r.ramp_type == "OR"].flow_vph.quantile(0.99), "FR_p99_vph": r[r.ramp_type == "FR"].flow_vph.quantile(0.99)})
print()
print(pd.DataFrame(rows).set_index("panel").round(2).to_string())
""")

code(r"""
rp = ramp[ramp.flow_vph.notna()].groupby(["ramp_type", "tod"]).flow_vph.mean().unstack(0)
mp = deep[deep.flow_vph.notna()].groupby("tod").flow_vph.mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(rp.index, rp["OR"], color=C[0], lw=1.6, label="on-ramp (OR)")
ax[0].plot(rp.index, rp["FR"], color=C[1], lw=1.6, label="off-ramp (FR)")
subtitle(ax[0], f"{DEEP}: mean ramp flow (veh/h)")
ax[0].legend()
ax[1].plot(mp.index, mp.values, color=C[2], lw=1.6)
subtitle(ax[1], f"{DEEP}: mean mainline link flow (veh/h, all lanes)")
headline(fig, "Ramp flows follow the daily cycle but are two orders of magnitude below mainline flows")
for a in ax:
    a.set_xlabel("hour of day")
    a.set_xticks(range(0, 25, 3))
plt.tight_layout()
plt.show()
""")

md(r"""
* Ramp flows follow a daily shape like the mainline, but their **level is two orders of magnitude smaller** (median tens of veh/h vs thousands on the mainline). Whatever the reason (units, a fraction of real ramp volumes), the ramp terms are small next to the mainline terms in a conservation balance.
* An unavailable ramp cell is *unavailable evidence*, never a zero flow. The docs say this explicitly, and the Task 3 scorer drops such transitions.

### 6.2 Conservation: N = k · L

Task 3 is computed from your Task 1 answer. The scorer derives density as an identity, `k = q / v`, then the number of vehicles on a link as `N = k · L` (L = link length), and checks vehicle conservation between consecutive 5-minute steps:

$$N(t+\Delta t) - N(t) = \Delta t\,(q_{in} + r_{on} - q_{out} - r_{off})$$

The plot shows the total number of vehicles on the corridor, `Σ N`, through two days computed from observed cells only (missing cells are filled by the link's same-day linear interpolation so the sum is comparable across time).
""")

code(r"""
fig, ax = plt.subplots(figsize=(12, 3.4))
peaks = {}
for i, (dd, lab) in enumerate([(typ_day, "median day"), (slow_day, "slowest day")]):
    x = deep[deep.date == dd].sort_values(["order_index", "tod"]).copy()
    x["N"] = x.flow_vph / x.speed_kmh * x.length_km
    x["N"] = x.groupby("link_id").N.transform(lambda s: s.interpolate(limit_direction="both"))
    tot = x.groupby("tod").N.sum()
    peaks[lab] = tot.max()
    ax.plot(tot.index, tot.values, color=C[i], lw=1.6, label=f"{dd.date()} ({lab})")
ax.set_xlabel("hour of day")
ax.set_ylabel("vehicles on corridor")
ax.set_xticks(range(0, 25, 3))
ax.legend()
subtitle(ax, f"{DEEP}: vehicles on the corridor, Σ (q / v) · L")
headline(fig, f"On the slowest day the corridor holds up to {peaks['slowest day']:,.0f} vehicles, vs {peaks['median day']:,.0f} on a median day")
plt.tight_layout()
plt.show()
""")

md(r"""
Accumulation grows sharply when queues form: in congestion speed drops faster than flow, so `q / v` (density) jumps. This is why the physics score is sensitive to congested cells: an answer that smooths away a queue also removes the vehicles stored in it and breaks the conservation balance.

## 7. The Task 4 network

Task 4 asks for one flow per path for a single departure period per split. The inputs are the path set, a path–link incidence table, link counts and a weak prior on path flows.
""")

code(r"""
rows = []
for p in PANELS:
    ps = net(p, "path_set.csv")
    inc = net(p, "path_link_incidence.csv")
    nlinks = ps.link_seq.str.count(";") + 1
    lc = pd.read_csv(ROOT / "task4" / p / "train" / "synthetic_link_counts.csv")
    pr = pd.read_csv(ROOT / "task4" / p / "train" / "synthetic_weak_prior.csv")
    rows.append({"panel": p, "paths": len(ps), "od_pairs": ps[["origin_zone", "destination_zone"]].drop_duplicates().shape[0],
                 "origins": ps.origin_zone.nunique(), "destinations": ps.destination_zone.nunique(),
                 "links_per_path_median": nlinks.median(), "links_per_path_max": nlinks.max(),
                 "incidence_rows": len(inc), "incidence_link_ids": inc.link_id.nunique(),
                 "count_links": len(lc), "departure": pr.departure_time.iloc[0], "prior_total": pr.path_flow.sum()})
t4 = pd.DataFrame(rows).set_index("panel")
print(t4.round(1).to_string())
""")

md(r"""
* Every OD pair has exactly one path (`paths == od_pairs`), so path flow and OD flow are the same thing here.
* The incidence table lists mainline links plus zone connectors (`RON-…`, `ROFF-…`); link counts are given for mainline links only.
* There is a single departure period per split (`SYN_PM_TRAIN`, `SYN_PM_VALIDATION`, `SYN_PM_PRIVATE`).

How consistent is the weak prior with the link counts? Pushing the prior through the incidence matrix (`A · b`) gives the link counts the prior implies, which can be compared with the released counts `c`.
""")

code(r"""
fig, axes = plt.subplots(2, 5, figsize=(15, 5.6))
rows = []
for a, p in zip(axes.flat, PANELS):
    inc = net(p, "path_link_incidence.csv")
    pr = pd.read_csv(ROOT / "task4" / p / "train" / "synthetic_weak_prior.csv")
    lc = pd.read_csv(ROOT / "task4" / p / "train" / "synthetic_link_counts.csv")
    ab = inc.merge(pr[["path_id", "path_flow"]], on="path_id").groupby("link_id").path_flow.sum()
    m = lc.set_index("link_id")["count"].to_frame("c").join(ab.rename("Ab"), how="left").fillna(0)
    lim = max(m.c.max(), m.Ab.max()) * 1.05
    a.scatter(m.c, m.Ab, s=6, color=C[0], alpha=0.6)
    a.plot([0, lim], [0, lim], color=INK, lw=0.8, ls=":")
    a.set_title(p, fontsize=9)
    rows.append({"panel": p, "sum_c": m.c.sum(), "sum_Ab": m.Ab.sum(), "ratio": m.Ab.sum() / m.c.sum(),
                 "rel_L1": (m.Ab - m.c).abs().sum() / m.c.sum(), "corr": m.c.corr(m.Ab)})
fig.supxlabel("released link count c", fontsize=9)
fig.supylabel("count implied by prior, A·b", fontsize=9)
headline(fig, "The weak prior implies link counts that bear little resemblance to the released counts")
plt.tight_layout()
plt.show()
print(pd.DataFrame(rows).set_index("panel").round(3).to_string())
""")

md(r"""
* The released link counts are almost flat along each corridor (roughly 4,000–7,500 per link).
* The counts implied by the prior vary several-fold from link to link and sum to 0.7–2.3 times the released total.
* The two correlate only weakly (0.1–0.4).

So the prior is a loose starting point, both in level and in its spatial pattern. Task 4 is about moving from it towards path flows that agree with the counts.

## 8. Summary

**Structure**
* Ten directional corridors, 44–352 mainline links each, and a complete 5-minute grid with no duplicates. Splits: 273 train days, 31 validation, 30 private.
* Quality flags follow the documented rules exactly. About 76 % of cells are score-eligible in every corridor, with the same `pct_observed` distribution everywhere.
* Cells with `pct_observed` between 1 and 74 are not scored but contain usable observations.

**Quality quirks**
* Speed is clipped at 120.75 km/h; occupancy saturates at 0.75 in heavy congestion.
* `links.csv` = `fd_parameters.csv`, but `lwr_mainline_topology.csv` holds different free speeds, capacities and some lane counts.
* In validation/private, 90 minutes after each Task 2 origin are hidden on all links. This adds full-width gaps to the Task 1 data.

**Traffic**
* Strong daily profiles. Congestion is concentrated between 13:00 and 20:00, with large differences in severity between corridors.
* The free-flow branch of the fundamental diagram is almost exactly linear with slope = free speed; the congested branch is a scattered cloud.
* Queues are rare, local and recurrent: a few bottleneck links hold most queued time, and they spread upstream.
* In most corridors the amount of congestion varies strongly from month to month, with no clear trend.

**Ramps and paths**
* Ramp flows are tiny compared with mainline flows; an unavailable ramp cell is not a zero.
* Task 4 has one path per OD pair. The weak prior matches the link counts only loosely.

Thanks for reading. If you spot a mistake or have a question about a plot, please leave a comment.
""")

code(r"""
print(f"total runtime: {time.time() - T0:.0f}s")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = HERE / "traffic-flow-bench-eda.ipynb"
nbf.write(nb, out)
print(out)
