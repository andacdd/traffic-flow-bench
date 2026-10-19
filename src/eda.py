import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

COLS = ["timestamp", "link_id", "speed_kmh", "flow_vph", "occupancy", "pct_observed", "is_score_eligible"]


def read_dir(path, cols):
    files = sorted(glob.glob(str(path / "**" / "*.parquet"), recursive=True))
    parts = []
    for f in files:
        t = pq.ParquetFile(f).read(columns=[c for c in cols if c in pq.ParquetFile(f).schema_arrow.names]).to_pandas()
        regime = [p.split("=")[1] for p in Path(f).parts if p.startswith("mask_regime=")]
        if regime:
            t["mask_regime"] = regime[0]
        parts.append(t)
    return pd.concat(parts, ignore_index=True), len(files)


def network_summary(rel):
    rows = []
    for panel in json.load(open(rel / "config" / "corridors.json")).get("panels", []) or sorted(p.name for p in (rel / "corridors").iterdir()):
        name = panel if isinstance(panel, str) else panel.get("corridor_id")
        net = rel / "corridors" / name / "network"
        links = pd.read_csv(net / "links.csv")
        paths = pd.read_csv(net / "path_set.csv")
        q2 = rel / "task2" / name / "train" / "window_index.csv"
        rows.append({
            "panel": name,
            "links": len(links),
            "paths": len(paths),
            "od_pairs": paths[["origin_zone", "destination_zone"]].drop_duplicates().shape[0] if {"origin_zone", "destination_zone"} <= set(paths.columns) else np.nan,
            "task2_train_windows": len(pd.read_csv(q2)) if q2.exists() else 0,
        })
    return pd.DataFrame(rows), links.columns.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--panel", default="D12_I5_N")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel, out = Path(a.release_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    try:
        net_df, link_cols = network_summary(rel)
    except Exception as e:
        net_df, link_cols = pd.DataFrame(), [str(e)]
    lines += ["## Koridor özeti", net_df.to_string(index=False), "", f"links.csv kolonları: {link_cols}", ""]

    base = rel / "corridors" / a.panel
    links = pd.read_csv(base / "network" / "links.csv")
    fd = pd.read_csv(base / "network" / "fd_parameters.csv")
    lines += ["## fd_parameters.csv (ilk 5)", fd.head().to_string(index=False), "", fd.describe().T.to_string(), ""]

    full, nf = read_dir(base / "train" / "mainline_states", COLS)
    masked, nm = read_dir(base / "train" / "mainline_states_masked", COLS)
    full["ts"] = pd.to_datetime(full["timestamp"])
    lines += [f"## {a.panel} train", f"gün dosyası: {nf} tam, {nm} maskeli; satır: {len(full):,}; link: {full.link_id.nunique()}", ""]

    lines += ["pct_observed dağılımı (%)", (full.pct_observed.value_counts(normalize=True).sort_index() * 100).round(2).to_string(), ""]
    lines += [f"score-eligible oranı: {full.is_score_eligible.mean():.3f}", ""]
    lines += ["speed/flow/occupancy özet", full[["speed_kmh", "flow_vph", "occupancy"]].describe().T.round(3).to_string(), ""]

    m = masked.merge(full[["timestamp", "link_id", "speed_kmh", "is_score_eligible"]].rename(columns={"speed_kmh": "speed_true", "is_score_eligible": "elig_true"}), on=["timestamp", "link_id"], how="left")
    m["blanked"] = m["speed_kmh"].isna() & m["speed_true"].notna()
    reg = m.groupby("mask_regime").agg(days=("timestamp", lambda s: s.str[:10].nunique()), cells=("blanked", "size"), blanked=("blanked", "mean"), blanked_eligible=("elig_true", lambda s: s[m.loc[s.index, "blanked"]].mean()))
    lines += ["## Maske rejimleri (train)", reg.round(3).to_string(), ""]

    lane_col = next((c for c in ["lanes", "num_lanes", "n_lanes", "lane_count"] if c in links.columns), None)
    vf_col = next((c for c in ["free_speed_kmh", "free_flow_speed_kmh", "vf_kmh", "free_speed"] if c in links.columns), None)
    full = full.merge(links[["link_id"] + [c for c in [lane_col, vf_col] if c]], on="link_id", how="left")
    if vf_col:
        full["queued"] = full.speed_kmh <= 0.6 * full[vf_col]
        lines += [f"queued (v <= 0.6*{vf_col}) oranı: {full.queued.mean():.4f}", ""]
        qb = full.assign(h=full.ts.dt.hour).groupby("h").queued.mean()
        lines += ["saate göre queued oranı", qb.round(4).to_string(), ""]

    full["dow"] = full.ts.dt.dayofweek
    full["tod"] = full.ts.dt.hour + full.ts.dt.minute / 60
    prof = full.groupby(["dow", "tod"])[["speed_kmh", "flow_vph"]].mean().reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for d, g in prof.groupby("dow"):
        ax[0].plot(g.tod, g.flow_vph, label=["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][d])
        ax[1].plot(g.tod, g.speed_kmh)
    ax[0].set_title("Ortalama akış (veh/h), gün x saat"); ax[1].set_title("Ortalama hız (km/h)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "profil_gun_saat.png", dpi=110); plt.close(fig)

    day = full.ts.dt.date.astype(str)
    busiest = full.assign(day=day).groupby("day").speed_kmh.mean().idxmin()
    hm = full[day == busiest].pivot_table(index="link_id", columns="tod", values="speed_kmh")
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdYlGn", extent=[0, 24, len(hm), 0])
    ax.set_title(f"{a.panel} {busiest} hız ısı haritası (en yavaş gün)"); ax.set_xlabel("saat"); ax.set_ylabel("link sırası")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out / "hiz_isi_haritasi.png", dpi=110); plt.close(fig)

    s = full[full.is_score_eligible == 1].sample(200000, random_state=0)
    lanes = s[lane_col] if lane_col else 1
    s = s.assign(q_lane=s.flow_vph / lanes, k_lane=s.flow_vph / lanes / s.speed_kmh.clip(lower=1))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(s.k_lane, s.q_lane, s=1, alpha=0.15)
    ax.set_xlabel("yoğunluk k (veh/km/şerit)"); ax.set_ylabel("akış q (veh/h/şerit)"); ax.set_title("Temel diyagram (q-k)")
    fig.tight_layout(); fig.savefig(out / "temel_diyagram.png", dpi=110); plt.close(fig)

    t2 = rel / "task2" / a.panel / "train" / "window_index.csv"
    if t2.exists():
        wi = pd.read_csv(t2)
        lines += ["## Task 2 window_index (train)", wi.head(10).to_string(index=False), "", wi.iloc[:, :].nunique().to_string(), ""]

    (out / "eda_ozet.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
