# HANDOFF — Traffic Flow Bench (başka bir LLM/oturum için giriş noktası)

Son güncelleme: 2026-10-23. Ayrıntılı geçmiş, tüm deneyler ve gerekçeler `NOTES.md` içinde (deney tablosu + faz bölümleri). Terimler `GLOSSARY.md` içinde. Bu dosya "şu an neredeyiz, sırada ne var" özetidir.

## 0. Devam etme promptu (yeni oturuma aynen yapıştır)

```
/Users/andac/Claude Personal Projects/ieee-traffic-flow-bench projesine devam ediyoruz (Kaggle 2026 IEEE Big Data Traffic Flow Bench). Önce HANDOFF.md, sonra GLOSSARY.md ve NOTES.md'nin "Deney tablosu" bölümünün son 25 satırını oku. Cevaplar Türkçe. Python: /opt/miniconda3/bin/python3, kaggle: /opt/miniconda3/bin/kaggle (PATH'te değiller). Submit, paket kurma, veri indirme ve ağır iş öncesi bana sor. Kodda yorum satırı yok. En iyi public LB 0.87018 (submissions/blend3_s_og50). Hedef: birinci olmak (lider ~0.924). Sıradaki adım HANDOFF.md §6'daki listenin ilk maddesi; başlamadan önce planı bana 3-5 satırda söyle.
```

## 1. Durum

- Yarışma: Kaggle "2026 IEEE Big Data - Traffic Flow Bench" (dosya tabanlı, tek CSV submit, günde 5 hak, gün 00:00 UTC'de yenilenir). Kullanıcı: `andacdemirdelen`. Repo: https://github.com/andacdd/traffic-flow-bench
- Metrik: `S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME`
- **En iyi public LB: 0.87346** — `submissions/blend6_og_og_refit/submission.csv` (blend5_s3_po + ongoing 3'lü: v4 + anubisyy + haftalık refit/olay hafızası, src/online_refit_ongoing.py). Final aday: `blend6_og_og_refit_q9` (+ private onset q9)
- **En iyi tamamen kendi modelimiz: 0.86559** — `submissions/runall_lec50/submission.csv`
- Görev dökümü (tahmini): state ~0.92, queue ~0.774 (onset ~0.757, ongoing ~0.79), physics ~0.72 (gürültü tabanı), ODME ~0.994
- Takvim: ödül kaydı TAMAM (onay 23 Eki), **6 Kas** son submit, **10 Kas** kod + PDF rapor (seçilen submission %1 içinde yeniden üretilmeli). Ödül = 0.85·LB + 0.15·rapor

## 2. Submit geçmişi (son)

| Submission | LB | Not |
|---|---|---|
| runall_b3_eiou | 0.86502 | Kendi: LEB3 state + v4 queue + EIoU karar + dual ODME |
| runall_q9_refit_onset_daily | 0.86377 | Günlük onset refit; val'de 2 pencere az veriyle bozuldu |
| blend_s50 | 0.86890 | State 0.5 bizim + 0.5 anubisyy public çıktısı |
| blend_s50_og50 | 0.87007 | + ongoing olasılık 0.5/0.5 (anubisyy ongoing LGBM), EIoU |
| blend_s50_og50_on50 / onfull | 0.86955 / 0.86963 | Onset harmanı zararlı; bizim onset kalır |
| runall_lec50 | 0.86559 | Kendi CatBoost state (aynı özellikler) ile LE 50/50 |
| **blend3_s_og50** | **0.87018** | State 0.25 LE + 0.25 CatBoost + 0.5 anubisyy, OG50 |

Final seçimi için adaylar: `blend3_s_og50` (en iyi public) ve `blend_s50_og50_q9` (aynı + private onset olay hafızası, 11 private pencere; public değişmez). İkinci adayın üstüne blend3 state'i de taşınmalı (henüz yapılmadı).

## 3. Yeniden üretim

```bash
git clone https://github.com/jacky850/trafficflowbench-public.git reference/trafficflowbench-public
git -C reference/trafficflowbench-public checkout f9e3fbc
kaggle competitions download 2026-ieee-big-data-traffic-flow-bench -p data
unzip data/2026-ieee-big-data-traffic-flow-bench.zip -d data
kaggle kernels output anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517 -p reference/anubisyy_output
kaggle datasets download anubisyy/trafficflowbench-086517-public-assets -f <dosya> -p reference/anubisyy_assets
```

anubisyy asset dosyaları: ongoing_lgb.txt, onset_lgb.txt, onset.cbm, onset_settings.json, queue_features.py, queue_ongoing.py, base_panel.py, config.py (lisans CC0).

Zincir:
1. Kendi state: `run_all.py --state LEB3` (LE + mask augmentation 0.5), önbellek `submissions/runall_b3/state/cache`
2. Kendi CatBoost state: `src/task1_cat_predict.py --cache-dir submissions/runall_b3/state/cache --base-vp submissions/runall_b3/state/state_vp.csv --out-dir submissions/state_cat` (~4 saat CPU)
3. Kendi queue: `src/task2_predict_sel.py` -> `submissions/queue_variants/q7_sel/eiou/queue_vp.csv` (+ `probs_*.npz`)
4. Birleştirme: `run_all.py --stages merge --state-vp ... --queue-vp ... --odme-vp submissions/runall_s3/odme/odme_vp.csv`
5. Harman: `src/blend_public.py` (state ağırlığı), `src/anubisyy_ongoing_probs.py` + `src/blend_ongoing.py` (ongoing olasılık ortalaması, EIoU)
6. Private onset olay hafızası: `src/online_refit_predict.py --base-queue ... --out submissions/queue_variants/q9_refit_onset` (haftalık; günlük `--daily` val'de zararlı)

Ortam: python 3.14 (miniconda), numpy 2.5, pandas 2.3, lightgbm 4.7, catboost 1.2.10, scipy 1.18. torch kurulu değil, GPU yok.

## 4. Bilinen gerçekler

- **Physics kapalı:** resmi scorer korunumda organizatörün gizli sınır akışlarını kullanır; maskeli bloklarda kalıntı ≈ Σ|Δgürültü|. Kâhin 0.71, biz ~0.72. Sadece state ensemble'ı küçük artış verir (gürültü ortalaması).
- **Queue açığı 0.068 (S_total):** onset nadir (olay) 0.029, ongoing normal 0.018, ongoing nadir 0.013, onset normal 0.009.
- **Olay tekrarı:** eval aylarında nadir kuyruk aynı linkte ertesi gün tekrar eder (lift val 3.9, private 2.8; train 1.25). Olay hafızası özelliği + eval ayı refit, Mart geri testinde nadir onset'i 0.52 -> 0.73 yapar. Resmi onset pencereleri ayın 1-9. günlerinde olduğu için public LB'de görünmez; private'ta etkili.
- Maskeli parquet dosyaları **tarih değil mask_regime sırasında**; gün komşuluğu takvimle kurulmalı (`online_refit.calendar`).
- Eval split ufukları maskeli dosyada boş (cov 0); resmi val pencereleri yerelde puanlanamaz.
- Normal onset kuyruğu train frekansının en yüksek linklerinde başlar (kâhin-k top-freq IoU 0.78).
- Çeşitlilik **özellik setinden** gelir: aynı özelliklerle CatBoost LB'de +0.0006, farklı pipeline (anubisyy) +0.0039.
- Ödül incelemesi: T sonrası satırlar silinip yeniden koşulur; eval ayı T öncesi günleri ile refit serbest (host #748789, raporda beyan); kopya kontrolü var — anubisyy (CC0) harmanı raporda açıkça belirtilmeli.

## 5. Kurallar

- Cevaplar Türkçe. Kodda yorum satırı yok. Çıktılar proje klasöründe
- Submit, paket/veri indirme, ağır iş, sistem ayarı öncesi kullanıcıya SOR
- Her LB probu tek değişken; sonuç `NOTES.md` deney tablosuna yazılır
- Queue kararları için ölçüt: ağırlıklı dev (ρ=0.28 val, 0.4 private) veya Mart hafta-ileri geri testi; düz dev LB'yi ters gösterir
- Kullanıcı küçük kazançlı "öğretici" probları istemez; büyük artış arar

## 6. Sıradaki adımlar (2026-10-27 grilling kararları; araştırma: reports/queue/research_leaders.md)

Hedef: private (Nisan) + ödül. Public LB yalnızca ongoing/state yön doğrulaması. Günde en fazla 3 tek değişkenli prob, 2 hak yedek. 6 Kas son deney günü; 7-10 Kas rapor + yeniden üretim.

1. Uyum testi: her pencere için T sonrası satırlar silinip yeniden koşulunca queue tahmini (q9 dahil) birebir aynı kalmalı (host #749601, #748789). q9 geçemezse günlük nedensel refit; o da Mart geri testinde negatifse q9 bırakılır
2. Tek prob: state = yalnız anubisyy (Larmuseau 0.9416 iddiası; bizim ~0.935 tahminini doğrular)
3. Ongoing: ham uzay-zaman yaması LGBM (21 link × 9 adım, Larmuseau) 3. üye, yalnız train ile; LB probu
4. Ongoing'e olay hafızası / ay-içi refit; LB probu (ongoing val ≤17 Mart, public'te görünür)
5. Onset: darboğaz aç/kapa kuralı (ay-içi ≥10 gün oran <0.05) + flow/cap kapısı; yalnız Mart hafta-ileri geri testi, sadece private
6. Arka planda: honghanhhh N-interpolasyonu fold-1 testi (physics)
7. Kazananlar final adaya (`blend5_s3_po_q9` tabanı) eklenir, uyum testi tekrar
8. Final iki dosya: public en iyi + private iyileştirmeli (uyum testinden geçmeli). yeashusemwal çıktısı açık lisans yoksa finalden çıkar

## 7. Dosya haritası

- `src/run_all.py` — tek giriş: state / queue / odme / merge, `--compare` ile %1 kontrolü
- `src/task1_resid*.py`, `src/task1_cat*.py` — state ay-CV ve üretim (LGBM, CatBoost)
- `src/task2_predict_sel.py` — üretim queue (v4 + EIoU), `src/online_refit*.py` — olay hafızası + refit, `src/refit_rarity.py`, `src/incident_recur.py`, `src/onset_anatomy.py` — queue analizleri
- `src/blend_*.py`, `src/anubisyy_*_probs.py` — public notebook harmanı
- `reports/queue/refit/`, `reports/phys/cat_f1/` — son geri testler
- `reference/public_notebooks/` — indirilmiş public notebook'lar (git dışı)
