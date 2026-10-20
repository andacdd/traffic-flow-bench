# HANDOFF — Traffic Flow Bench (başka bir LLM/oturum için giriş noktası)

Son güncelleme: 2026-10-20. Ayrıntılı geçmiş, tüm deneyler ve gerekçeler: `NOTES.md` (deney tablosu + faz bölümleri). Bu dosya "şu an neredeyiz, sırada ne var" özetidir.

## 1. Durum

- Yarışma: Kaggle "2026 IEEE Big Data - Traffic Flow Bench" (dosya tabanlı, tek CSV submit, günde 5 hak). Kullanıcı: `andacdemirdelen`
- Metrik: `S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`
- **En iyi public LB: 0.85929 (S5)** — `submissions/runall_s5/submission.csv`. Lider 0.924, public notebook'lar 0.865–0.871
- Takvim: **25 Eki 23:59 AoE** ödül kayıt e-postası (kullanıcı gönderir, trafficflowbench@gmail.com), **6 Kas** son submit, **10 Kas** kod + PDF rapor (seçilen submission %1 içinde yeniden üretilmeli)

## 2. Yeniden üretim

```bash
git clone https://github.com/jacky850/trafficflowbench-public.git reference/trafficflowbench-public
git -C reference/trafficflowbench-public checkout f9e3fbc
kaggle competitions download 2026-ieee-big-data-traffic-flow-bench -p data
unzip data/2026-ieee-big-data-traffic-flow-bench.zip -d data
```

S5'i baştan üretmek (state ~25 dk, queue ~3 dk, ODME dual ~5 sn, merge ~30 sn; tepe ~5–7 GB RAM):

```bash
cd /private/tmp && python -I "<proje>/src/run_all.py" --release-root "<proje>/data/kaggle_public" --repo-root "<proje>/reference/trafficflowbench-public" --out-dir "<proje>/submissions/runall_new" --state LE --queue v4 --odme own --odme-solver dual --threads 4 --seed 42 --compare "<proje>/submissions/runall_s5/submission.csv"
```

Ortam: python 3.14.6, numpy 2.5.1, pandas 2.3.3, pyarrow 24.0.0, lightgbm 4.7.0, scipy 1.18.0 (CatBoost kurulu değil). Script'ler `-I` ile `/private/tmp`'den çalıştırılır.

## 3. Mevcut çözüm (S5)

| Görev | Yöntem | Kod | Durum |
|---|---|---|---|
| Task 1 state | Kalıntı interpolasyonu (zaman+uzay, `--reconcile v:0.6`) tabanı + LightGBM kalıntı modeli; LE = lead/lag (±5..60 dk) modeli L ile ikinci model M2'nin ortalaması | `task1_interp.py`, `task1_resid2.py`, `task1_resid2_predict.py` | CV S_state 0.9397; LB'ye aktarımı ~%97 |
| Task 3 physics | Ayrı submission yok; state'ten hesaplanır | `task3_fast.py` (resmi scorer'la birebir proxy) | ~0.71; tavan ölçüm gürültüsü (ACF ile kanıtlandı), kapatıldı |
| Task 2 queue | v4: LightGBM, hibrit girdi, frekans özelliği YOK; onset sadece T+30 (eşik 0.30, top-1 fallback), ongoing eşik 0.35 | `task2_model.py`, `task2_diag.py`, `task2_predict_v2.py` | LB onset ≈ 0.757; ongoing bilinmiyor |
| Task 4 ODME | min ‖Af−c‖² + 0.05‖f−b‖², f≥0, split'in KENDİ sayımı ve prior'ı, kesin çözüm (dual) | `odme_dual_predict.py`, `odme_scan.py` | Kendi sayım +0.032 LB getirdi; gizli terimler yerelde ölçülemez |

## 4. Bilinen gerçekler (LB çapaları)

- Persistence queue LB ≈ 0.3017 (onset ~0, ongoing ~0.603)
- v3 queue (frekans özellikli): 0.7187 (onset 0.714, ongoing 0.723)
- v4 onset ≈ 0.757 (v3'ten +0.043). Final ongoing (seçici pencere + fill etiket + eşik 0.40) v4 ongoing'den −0.048
- State LE etkisi iki farklı queue ile de +0.0035 -> bileşenler toplanabilir
- Physics: kâhin komşu-ortalaması 0.699 ≈ modelimiz; kalıntı gürültüsü beyaz -> modelleme ile aşılamaz

## 5. Kurallar (kullanıcı tercihleri ve dersler)

- **Dev set queue model seçimi için GÜVENİLMEZ:** 3/3 karşılaştırmada LB yönünü ters gösterdi (modelleri 0.07–0.11 iyimser ölçüyor; persistence'ı ise tutturuyor). Queue kararları yalnızca tek değişkenli LB probu ile
- Her LB probu S5'e göre TEK değişken değiştirir; sonuç `NOTES.md` deney tablosuna yazılır. LB queue gürültüsü ±0.015 (S_total'de ±0.0045)
- State/physics/ODME-S_link yerelde güvenilir ölçülür (`task3_fast.py`, `score_task1.py`)
- Kodda yorum satırı yok. Cevaplar Türkçe. Çıktılar proje klasöründe
- Submit, veri/paket indirme, GPU/ağır paralel iş ve bilgisayar/sistem ayarı (uyku engelleme vb.) öncesi kullanıcıya SOR
- Seed ve versiyon sabit; her submit `run_all.py`'dan çıkar (manifest.json yazar)

## 6. Sıradaki adımlar (öncelik sırasıyla)

Hazır dosyalar `submissions/queue_variants/<ad>/queue_vp.csv` (S5'e göre tek değişken; `run_all.py --stages merge --queue-vp ...` ile birleştirilir):
1. `q2b_og045` — ongoing eşik 0.45 (sadece ongoing; fill etiket içermez)
2. `q4_onset055` — anubisyy tipi onset özellikleri (sadece onset)
3. ODME λ=100 probu (yerelde ölçülemez; günde en fazla 1 λ)
4. `q5_avg040` — ongoing iki model ortalaması
5. State B3 (mask augmentation, CV J +0.0009) için üretim script'i yaz (`task1_resid3.py` temel alınır)
6. State: daha çok tur + early-stop (B2 yarıda kaldı; 2000 turda hâlâ iyileşiyordu)
7. Final rapor (rubrik: teknik %40, analiz/insight %40, netlik %20) — `reports/` altındaki tablolar ve grafikler kullanılacak

## 7. Dosya haritası

- `src/run_all.py` — tek giriş: state / queue / odme / merge aşamaları, `--compare` ile %1 kontrolü
- `src/queue_devset.py` — organizatör seçicisini taklit eden kalibre dev set (persistence'ı tutturur)
- `src/queue_cv.py`, `queue_variants.py`, `queue_diag2.py` — queue CV, varyantlar, teşhis (adversarial, maske, LOMO)
- `src/task1_resid*.py` — state ay-CV ve üretim
- `reports/odme/`, `reports/phys/`, `reports/queue/` — tüm faz tabloları/grafikleri (final rapora girecek)
- `reports/anubisyy_comparison.md`, `reports/queue/public_notebooks_summary.md` — public notebook analizleri
- `submissions/runall_s5/` — en iyi submission + manifest
