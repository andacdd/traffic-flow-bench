import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

OUT = []


def say(*xs):
    s = " ".join(str(x) for x in xs)
    OUT.append(s)
    print(s, flush=True)


def read_dir(path, cols=None, days=None):
    files = sorted(glob.glob(str(path / "**" / "*.parquet"), recursive=True))
    if days:
        files = files[::max(1, len(files) // days)][:days]
    parts = []
    for f in files:
        pf = pq.ParquetFile(f)
        use = [c for c in (cols or pf.schema_arrow.names) if c in pf.schema_arrow.names]
        t = pf.read(columns=use).to_pandas()
        reg = [p.split("=")[1] for p in Path(f).parts if p.startswith("mask_regime=")]
        if reg:
            t["mask_regime"] = reg[0]
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), len(files)


def network_checks(rel, panel):
    net = rel / "corridors" / panel / "network"
    links = pd.read_csv(net / "links.csv")
    fd = pd.read_csv(net / "fd_parameters.csv")
    topo = pd.read_csv(net / "lwr_mainline_topology.csv")
    ramp = pd.read_csv(net / "ramp_attachment_map.csv")
    m = links.merge(fd, on="link_id", suffixes=("_links", "_fd"), how="outer", indicator=True).merge(
        topo[["mainline_link_id", "free_speed_kmh", "capacity_vph", "lanes", "length_km", "has_sensor", "order_index"]].rename(
            columns={"mainline_link_id": "link_id", "free_speed_kmh": "vf_topo", "capacity_vph": "cap_topo", "lanes": "lanes_topo", "length_km": "len_topo"}),
        on="link_id", how="left")
    r = {"panel": panel, "links": len(links), "fd_rows": len(fd), "topo_rows": len(topo),
         "links_not_in_fd": int((m._merge == "left_only").sum()), "fd_not_in_links": int((m._merge == "right_only").sum()),
         "sensored": int(topo.has_sensor.sum()), "unsensored": int((topo.has_sensor == 0).sum()),
         "on_ramps": int((ramp.ramp_type == "OR").sum()), "off_ramps": int((ramp.ramp_type == "FR").sum())}
    for a, b, k in [("free_speed_kmh_links", "free_speed_kmh_fd", "vf_links_vs_fd"), ("free_speed_kmh_fd", "vf_topo", "vf_fd_vs_topo"),
                    ("capacity_vph_links", "capacity_vph_fd", "cap_links_vs_fd"), ("capacity_vph_fd", "cap_topo", "cap_fd_vs_topo"),
                    ("lanes_links", "lanes_fd", "lanes_links_vs_fd"), ("lanes_fd", "lanes_topo", "lanes_fd_vs_topo"),
                    ("length_km_links", "length_km_fd", "len_links_vs_fd")]:
        if a in m and b in m:
            d = (m[a] - m[b]).abs()
            r[k] = f"{int((d > 1e-6).sum())} fark, max {d.max():.1f}"
    order_gap = topo.sort_values("order_index").order_index.diff().dropna()
    r["order_index_gaps"] = int((order_gap != 1).sum())
    return r, m


def mainline_checks(rel, panel, split, days):
    base = rel / "corridors" / panel / split
    say(f"\n### {panel} / {split} mainline")
    sub = "mainline_states" if split == "train" else "mainline_states_masked"
    df, nf = read_dir(base / sub, days=days)
    say(f"dosya {nf}, satır {len(df):,}, kolon {len(df.columns)}")
    say("kolonlar:", list(df.columns))
    say("\nnull oranı (%):")
    say((df.isna().mean() * 100).round(2)[lambda s: s > 0].to_string() or "yok")
    dup = df.duplicated(["timestamp", "link_id"]).sum()
    say(f"\n(timestamp, link_id) tekrar: {dup:,}")
    say(f"link başına station_id sayısı max: {df.groupby('link_id').station_id.nunique().max()}")
    df["ts"] = pd.to_datetime(df.timestamp)
    per_day = df.groupby(df.ts.dt.date).size()
    exp = df.link_id.nunique() * 288
    say(f"gün başına satır: min {per_day.min()}, max {per_day.max()}, beklenen {exp}")
    say(f"dakika ızgarası dışı: {(df.ts.dt.minute % 5 != 0).sum()}")

    pobs = df.pct_observed
    v = df.speed_kmh.notna()
    say("\npct_observed bandı x değer var mı (speed notna oranı):")
    band = pd.cut(pobs, [-1, 0, 74, 99, 100], labels=["0", "1-74", "75-99", "100"])
    say(df.assign(band=band, has=v).groupby("band", observed=True).agg(oran=("has", "size"), deger_var=("has", "mean")).assign(
        oran=lambda x: (x.oran / len(df) * 100).round(2), deger_var=lambda x: x.deger_var.round(3)).to_string())
    flag = df[["is_observed", "is_imputed", "is_score_eligible", "is_missing"]].astype(int)
    say("\nbayrak tutarlılığı:")
    say(f"  is_imputed != (pct<100): {(flag.is_imputed != (pobs < 100)).sum():,}")
    say(f"  is_score_eligible != (pct>=75 & değer var): {(flag.is_score_eligible != ((pobs >= 75) & v)).sum():,}")
    say(f"  is_missing=1 ama değer var: {((flag.is_missing == 1) & v).sum():,}")
    say(f"  is_observed=0 ama değer var: {((flag.is_observed == 0) & v).sum():,}")

    say("\nmetrik dağılımları (değer olan hücreler):")
    q = df[["speed_kmh", "flow_vph", "occupancy", "density_occ_linear_vehpkm"]].describe(percentiles=[.01, .05, .5, .95, .99]).T
    say(q.round(3).to_string())
    say(f"  speed==0: {(df.speed_kmh == 0).sum()}, flow==0: {(df.flow_vph == 0).sum()}, negatif: {(df[['speed_kmh','flow_vph','occupancy']] < 0).sum().sum()}")
    say(f"  occupancy>1: {(df.occupancy > 1).sum()}")
    return df


def physics_checks(df, net):
    say("\nfiziksel tutarlılık (değer olan hücreler):")
    d = df.dropna(subset=["speed_kmh", "flow_vph"]).merge(net[["link_id", "lanes_fd", "free_speed_kmh_fd", "capacity_vph_fd", "k_jam", "critical_density", "length_km_fd"]], on="link_id", how="left")
    d["q_lane"] = d.flow_vph / d.lanes_fd
    d["k_lane"] = d.q_lane / d.speed_kmh
    say(f"  speed > vf+5: {(d.speed_kmh > d.free_speed_kmh_fd + 5).mean()*100:.3f}%  (max fark {(d.speed_kmh - d.free_speed_kmh_fd).max():.1f} km/h)")
    say(f"  flow > kapasite: {(d.flow_vph > d.capacity_vph_fd).mean()*100:.3f}%  (max oran {(d.flow_vph / d.capacity_vph_fd).max():.2f})")
    say(f"  k_lane > k_jam: {(d.k_lane > d.k_jam).mean()*100:.4f}%")
    kd = d.density_occ_linear_vehpkm / d.lanes_fd
    say(f"  corr(q/v, density_occ): {np.corrcoef(d.flow_vph / d.speed_kmh, d.density_occ_linear_vehpkm)[0,1]:.4f}")
    say(f"  density_occ / (q/v) medyan: {(d.density_occ_linear_vehpkm / (d.flow_vph / d.speed_kmh)).median():.3f}")
    ff = d[d.k_lane < d.critical_density / d.lanes_fd * 0 + d.critical_density * 0.8]
    say(f"  serbest akışta |v - vf| medyan: {(ff.speed_kmh - ff.free_speed_kmh_fd).abs().median():.2f} km/h, p95 {(ff.speed_kmh - ff.free_speed_kmh_fd).abs().quantile(.95):.2f}")
    elig = d.is_score_eligible == 1
    for name, sub in [("eligible", d[elig]), ("pct 1-74 (imputed, eligible değil)", d[(d.pct_observed > 0) & (d.pct_observed < 75)])]:
        cong = (sub.speed_kmh <= 0.6 * sub.free_speed_kmh_fd).mean()
        say(f"  {name}: n={len(sub):,}, kuyruk oranı {cong*100:.2f}%, hız std {sub.speed_kmh.std():.2f}")


def stuck_checks(df):
    say("\ntakılı sensör (aynı link, ardışık 6+ adım birebir aynı hız):")
    d = df.dropna(subset=["speed_kmh"]).sort_values(["link_id", "ts"])
    same = d.speed_kmh.diff().eq(0) & d.link_id.eq(d.link_id.shift())
    grp = (~same).cumsum()
    runs = same.groupby(grp).sum()
    say(f"  6+ uzunlukta tekrar sayısı: {(runs >= 5).sum()}, en uzun {int(runs.max())}")


def mask_checks(rel, panel, days):
    say(f"\n### {panel} maske tutarlılığı (train)")
    base = rel / "corridors" / panel / "train"
    full, _ = read_dir(base / "mainline_states", ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"], days=days)
    msk, _ = read_dir(base / "mainline_states_masked", ["timestamp", "link_id", "speed_kmh", "flow_vph", "is_score_eligible"])
    m = full.merge(msk, on=["timestamp", "link_id"], suffixes=("", "_m"))
    kept = m.speed_kmh_m.notna()
    say(f"  eşleşen satır {len(m):,}; maskelenmemiş hücrelerde değer farkı: {(m.loc[kept, 'speed_kmh'] - m.loc[kept, 'speed_kmh_m']).abs().max()}")
    say(f"  maskede kayıp ama tamda var: {(m.speed_kmh.notna() & m.speed_kmh_m.isna()).mean()*100:.2f}%")
    say(f"  maskede var ama tamda yok: {(m.speed_kmh.isna() & m.speed_kmh_m.notna()).sum()}")
    t1 = pd.read_csv(rel / "task1" / panel / "train" / "sample_submission_state.csv", nrows=5)
    say(f"  task1 şablon kolonları: {list(t1.columns)}")


def ramp_checks(rel, panel, split, days):
    say(f"\n### {panel} / {split} ramp_states")
    df, nf = read_dir(rel / "corridors" / panel / split / "ramp_states", days=days)
    if df.empty:
        say("  yok")
        return
    say(f"dosya {nf}, satır {len(df):,}, kolonlar {list(df.columns)}")
    say("null oranı (%):", (df.isna().mean() * 100).round(2)[lambda s: s > 0].to_dict())
    fc = next((c for c in df.columns if "flow" in c), None)
    if fc:
        say(f"  {fc}: ", df[fc].describe(percentiles=[.01, .5, .99]).round(1).to_dict())
        say(f"  {fc}==0: {(df[fc] == 0).mean()*100:.2f}%  | pct_observed<75 iken değer var: {((df.pct_observed < 75) & df[fc].notna()).mean()*100:.2f}%")
    say(f"  eligible oranı: {(df.pct_observed >= 75).mean():.3f}; ramp sayısı {df.iloc[:, [i for i,c in enumerate(df.columns) if 'ramp' in c and 'id' in c][0]].nunique() if any('ramp' in c and 'id' in c for c in df.columns) else '?'}")


def heldout_gap_checks(rel, panel, split):
    say(f"\n### {panel} / {split} maske + Task2 boşlukları")
    df, nf = read_dir(rel / "corridors" / panel / split / "mainline_states_masked", ["timestamp", "link_id", "speed_kmh", "pct_observed", "mask_regime"])
    df["ts"] = pd.to_datetime(df.timestamp)
    say(f"gün {nf}, satır {len(df):,}, değer yok oranı {df.speed_kmh.isna().mean()*100:.1f}%")
    say(df.groupby("mask_regime").agg(gun=("ts", lambda s: s.dt.date.nunique()), bos=("speed_kmh", lambda s: s.isna().mean().round(3))).to_string())
    full_blank = df.groupby("ts").speed_kmh.apply(lambda s: s.isna().all())
    say(f"tüm linklerin boş olduğu zaman adımı: {int(full_blank.sum())} ({int(full_blank.sum())*5} dk)")
    wi = rel / "task2" / panel / split / "window_index.csv"
    if wi.exists():
        w = pd.read_csv(wi)
        say(f"Task2 pencere: {len(w)} ({w.condition.value_counts().to_dict()}); her biri 30 dk ufuk + 60 dk tampon = 18 adım boş beklenir")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--panel", default="D12_I5_N")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.release_root)
    panels = sorted(p.name for p in (rel / "corridors").iterdir())

    say("## Ağ dosyaları tutarlılığı (tüm paneller)")
    rows, net_main = [], None
    for p in panels:
        r, m = network_checks(rel, p)
        rows.append(r)
        if p == a.panel:
            net_main = m
    say(pd.DataFrame(rows).to_string(index=False))

    df = mainline_checks(rel, a.panel, "train", a.days)
    physics_checks(df, net_main)
    stuck_checks(df)
    mask_checks(rel, a.panel, a.days)
    ramp_checks(rel, a.panel, "train", a.days)
    for sp in ["validation", "private"]:
        heldout_gap_checks(rel, a.panel, sp)
        ramp_checks(rel, a.panel, sp, None)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(OUT))


if __name__ == "__main__":
    main()
