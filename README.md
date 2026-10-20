# ieee-traffic-flow-bench

- **Amaç:** Kaggle "2026 IEEE Big Data - Traffic Flow Bench" yarışmasında dört görevli (state, queue, physics, ODME) otoyol trafiği çözümü kurmak; öğrenerek ilerlemek.
- **Durum:** devam ediyor
- **Başlangıç:** 2026-10-16
- **Yarışma:** https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench
- **Resmi repo (baseline + scorer):** https://github.com/jacky850/trafficflowbench-public (commit `f9e3fbc`)

## Durum

> Başka bir LLM/oturumda devam etmek için önce [`HANDOFF.md`](HANDOFF.md) dosyasını okuyun.

En iyi public LB: **0.85929** (S5: LE state = lead/lag LightGBM + ikinci model ortalaması, v4 LGBM queue, kendi-sayım NNLS ODME; `src/run_all.py --state LE --queue v4 --odme own`). Tüm deneyler, kararlar ve bulgular `NOTES.md` içinde (deney tablosu + faz bölümleri).

## Klasör yapısı

- `src/` — tüm kod
  - Task 1/3 state: `task1_interp.py` (zaman+uzay kalıntı interpolasyonu, `--reconcile v:0.6`), `task1_resid.py` (LightGBM kalıntı modeli, ay-CV), `task1_resid_predict.py` (üretim)
  - Physics proxy: `task3_proxy.py` (resmi scorer + gerçeği koruyan sınır akışı), `task3_fast.py` (hızlı birebir proxy)
  - Task 2 queue: `task2_model.py`, `task2_diag.py`, `queue_devset.py` (organizatör seçicisini taklit eden kalibre dev set), `queue_cv.py`, `queue_exp3c.py`, `queue_error_analysis.py`, `queue_final_cv.py`, `queue_final_predict.py`
  - Task 4 ODME: `odme_scan.py`
  - EDA / kalite: `eda.py`, `profile_quality.py`
- `notebooks/` — EDA çıktıları ve Kaggle EDA notebook'u
- `reports/` — faz tabloları ve grafikleri (`odme/`, `phys/`, `queue/`)
- `submissions/` — üretim çalıştırma kayıtları (`run.json`, `versions.json`, model dosyaları); CSV'ler repoda yok

Repoda olmayanlar: yarışma verisi (`data/`, yeniden dağıtılamaz), cache (`cache/`), submission CSV'leri, resmi repo klonu ve public notebook kopyaları.

## Kurulum

```bash
git clone https://github.com/jacky850/trafficflowbench-public.git reference/trafficflowbench-public
git -C reference/trafficflowbench-public checkout f9e3fbc
kaggle competitions download 2026-ieee-big-data-traffic-flow-bench -p data
unzip data/2026-ieee-big-data-traffic-flow-bench.zip -d data
```

Ortam: python 3.14.6, numpy 2.5.1, pandas 2.3.3, pyarrow 24.0.0, lightgbm 4.7.0, scipy 1.18.0. Script'ler `python -I src/<script>.py --release-root data/kaggle_public ...` şeklinde çalışır (argümanlar için `--help`).

## Submission üretimi (S2/S3)

1. State: `src/task1_resid_predict.py` -> `submissions/state_A/state_vp.csv`
2. Queue: `src/queue_final_predict.py` -> `submissions/queue_final/queue_vp.csv` (S2 için `src/task2_predict_v2.py`)
3. ODME: resmi repo `src/task4/build_task4_odme_artifacts.py --split <validation|private>` çıktısındaki `*_public_reference.csv` (kendi split sayımı)
4. Birleştirme: resmi repo `src/merge_submissions.py --state ... --queue ... --odme ... --key data/kaggle_public/submission_key.csv --output submission.csv`
