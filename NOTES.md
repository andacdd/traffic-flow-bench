# Notlar

## Kararlar

- 2026-10-16 — Proje açıldı. Submit, büyük veri indirme ve GPU işi öncesi kullanıcıya sorulacak.
- 2026-10-16 — Kullanıcı kuralları kabul etti; tüm veri indirildi ve data/kaggle_public/ altına açıldı (9688 dosya, 12 GB). Zip data/ içinde duruyor (8.3 GB).
- 2026-10-22 — Online refit + olay hafızası planı (grill turu, kullanıcı onayladı): (1) Mart haftalara bölünür (1-7, 8-14, 15-21, 22-31), 2-4. haftalarda base (train) ile train + o haftadan önceki Mart günleri karşılaştırılır. Resmi val ufukları maskeli dosyada boş olduğu için test, Mart'ta gözlenen hücrelerden kurulan sahte pencerelerle yapılır (ufkun >=%50'si gözlenmiş). (2) Eval günü etiketi sadece gözlenen hücreden. (3) Sadece T'nin gününden önceki günler; aynı gün yok. (4) Mart satır ağırlığı {1,3,5}. (5) A = sadece ongoing refit; B = refit + olay hafızası (link bazında önceki 1/3/7 günde nadir kuyruk, gün geneli ve T±1 saat), onset ve ongoing. Hafıza her yerde maskeli dosyadan. (6) Submit eşiği: queue ortalaması (onset+ongoing) >= +0.01. (7) Val üretimi: 1. hafta B3+EIoU, 2-4. hafta haftalık refit. Private: train + tüm Mart + Nisan'ın o haftadan önceki günleri (host #748789; raporda beyan edilecek).

## Açık işler

- [ ] Strateji ve veri kapsamı kararı (kullanıcı)
- [x] Yarışmaya katıl (2026-10-16)
- [ ] Ödül istenirse 25 Ekim 2026 23:59 AoE'ye kadar trafficflowbench@gmail.com'a kayıt e-postası (kullanıcı gönderir)
- [x] Resmi repo baseline'ını çalıştır, yerel scorer ile train üzerinde skorla (2026-10-16)
- [x] Kısa EDA (D12_I5_N) — notebooks/eda_D12_I5_N/
- [x] v1 baseline submission dosyası hazır (submissions/v1_baseline/submission.csv), kullanıcı kararıyla submit EDİLMEDİ
- [x] Task 1 zaman+uzay interpolasyonu (train 0.905)
- [x] v2 submit edildi 2026-10-19 (ref 57385124): public LB 0.64843. Repo naive baseline validation 0.5534
- HATA DERSİ: build_task2_persistence_submission.py --split all = train+validation, private'ı ATLIYOR. Split'leri tek tek üret. v1_baseline/submission.csv bu yüzden private queue'da hep 0 (kullanma)

## Önemli bilgiler

### Takvim (2026)
- 25 Eki: ödül için kayıt e-postası son gün
- 7 Kas 06:55 UTC (6 Kas AoE): son Kaggle submission, private LB açılır, takım birleştirme son günü
- 10 Kas: kod + PDF rapor (ödül için)
- 18 Kas: kazananlar; 14-17 Ara IEEE BigData 2026, Phoenix

### Kurallar
- Günlük 5 submission, takım en fazla 4 kişi
- Teslim: tek CSV (`sample_submission.csv` şablonu, ~7M satır). Code competition değil, notebook şart değil
- Kolonlar: submission_id, task, speed_kmh, flow_vph, queue_pred, path_flow. `submission_key.csv` id -> hücre eşlemesi
- Boş hücre yok, id eklenmez/değişmez; bir görevin tek satırı eksikse o görev 0
- Task 2 online: T'den sonraki zaman damgalı hiçbir veri kullanılamaz (ramp_states dahil, buffer sonrası dahil)
- Task 1/3/4 offline: tüm dönemin verisi kullanılabilir
- Points/medal vermiyor. Ödül: 1500/1000/500 USD + K-12 500 USD. Final skor = 0.85 LB + 0.15 rapor, top-10 reproducibility kontrolü

### Metrik
- S_total = 0.35 S_state + 0.30 S_queue + 0.15 S_physics + 0.20 S_ODME
- 10 yönlü panel; yön çiftleri ortalanır, sonra 5 koridor eşit ağırlık
- S_state: rejim başına (R1/R2/R3 = %20/30/50 maske) 0.54*max(0,1-RMSE_v/25) + 0.46*max(0,1-RMSE_q_lane/600)
- S_queue: pencere başına uzay-zaman IoU (queued = hız <= 0.6 * serbest akış hızı), T+5..T+30
- S_physics = 1/3 S_FD (üçgen FD) + 2/3 S_LWR (araç korunumu); sadece Task 1 değerlerinden hesaplanır
- S_ODME = 0.45 S_od + 0.25 S_link + 0.15 S_dev + 0.15 S_attr
- Skorlanan hücre: pct_observed >= 75 (~%76 hücre)

### Veri
- Sentetik (kalibre trafik modeli), 5 dk çözünürlük, 10 yönlü koridor, 11 ay
- Split: train 9 ay (cevaplar var), validation 1 ay (public LB), private 1 ay (final)
- Boyut: corridors/train 8.1 GB, task1/train 2.0 GB, submission_key 0.64 GB, sample_submission 0.15 GB; validation+private ağ/maske dosyaları ~1 GB
- Tüm koridorlar Mode A (ramp coverage 0.756)
- D12_I405_N/S Task 2'de yok

### Baseline (2026-10-16, train, yerel scorer)
- Repo: reference/trafficflowbench-public (commit f9e3fbc). Çıktılar: submissions/baseline/
- Task 1 historical mean (gün x saat profili): S_state 0.7320 (train, in-sample, iyimser). En kötü D7_I405_N 0.558, D12_I5_N 0.625; en iyi D7_I210_E 0.888
- Hata çoğunlukla hızda: S_speed 0.35-0.87, S_flow 0.80-0.90. Kuyruklu koridorlar zayıf -> kazanç sıkışmayı yakalamakta
- Task 4 NNLS (lambda 0.05): S_link 0.9999 (S_ODME'nin sadece %25'i; geri kalanı yerelde ölçülemez)
- Task 2 ve Task 3 yerelde skorlanamaz (etiket yok / organizatör sınır akışları yok)
- Repo validation tablosu: naive S_total 0.5534; LB 1. 0.924
- score_task1.py raporu split ne olursa olsun reports/task1_eval_validation.csv adıyla yazıyor
- Tüm train build 221 s, skor dahil ~48 dk (8 çekirdek, 16 GB)

### EDA (D12_I5_N train)
- 258 link, 273 gün, 20.3M satır. pct_observed: %63 tam, %16.5 sıfır; eligible %76.4
- Maske: R1 96 gün, R2 84, R3 93. Maskelenen hücrelerin tamamı eligible
- Serbest akış dalı neredeyse kusursuz doğru (q = v_f * k, v_f ~117.5). Sıkışma dalı kümelenmiş bantlar
- Kuyruk oranı %4.3; 13:00-20:00 arası yoğun (16:00'da %19); hafta sonu daha yavaş
- Sıkışma link-özel ve kalıcı bantlar (darboğazlar); komşu link + aynı gün komşu zaman bilgisi baseline'ın kullanmadığı ana sinyal
- Task 2: her panelde split başına 10 pencere (train dahil) -> çok az örnek; train mainline_states'ten kendi pencerelerimizi türetmek gerekecek

### Task 2 modeli (src/task2_model.py, 2026-10-19)
- Train horizonları train/mainline_states'te GÖZLENİYOR -> 80 resmi train penceresi yerelde skorlanabilir (etiket = gözlenen hız <= 0.6*vf)
- Eşik: fd_parameters'ta v_cut yok -> 0.6 * links.free_speed_kmh
- Sentetik pencereler: onset = 90 dk sessiz + t'de kuyruk, T=t-30 (2327 pencere); ongoing = T'de kuyruk olan rastgele 2 origin/gün (3917 pencere)
- Özellikler: T-60..T hız/akış/doluluk (oran, ortalama, min, eğim), komşu ±3 link, aşağı/yukarı 5 link min hız, koridor geneli, train kuyruk frekansı (link x saat), tod/dow/şerit/kapasite/konum/panel. T satırı %35 rastgele düşürülerek Task1 maskesi taklit edildi
- LightGBM (4 thread, 400 tur). Onset: sadece T+30, eşik 0.35, en az 1 link. Ongoing: her ufuk adımı, eşik 0.5
- Holdout (gün >= 218): onset IoU 0.930 (persistence 0.0), ongoing 0.889 (persistence 0.791)
- Resmi train pencereleri (in-sample, iyimser): 0.919 (onset 0.975, ongoing 0.863) vs persistence 0.395
- Önemli özellik: onset freq_max_h %55 (bilinen darboğaz), tod %16; ongoing komşu fc_last ve doluluk
- v3 submit 2026-10-19 (ref 57385565): public LB 0.77351 (v2 0.64843). Fark 0.125 / 0.30 ağırlık -> validation S_queue ~0.30'dan ~0.72'ye. Holdout 0.91'den düşük: ay farkı/olay farkı, sentetik pencere seçimi gerçek seçiciyle tam aynı değil

## Deney tablosu

| Tarih | Faz | Değişiklik (tek değişken) | Yerel skor | LB |
|---|---|---|---|---|
| 2026-10-19 | - | v2: interp state | S_state train 0.905 | 0.64843 |
| 2026-10-19 | - | v3: LGBM queue | proxy queue 0.919 (in-sample) | 0.77351 |
| 2026-10-19 | - | v5: v:0.6 reconcile + no_freq queue (2 değişken) | S_state 0.9115, phys proxy 0.626 | 0.80550 |
| 2026-10-19 | 0 | ODME sayım kaynağı: train -> kendi split (λ=0.05) | val S_link 0.9646 -> 0.9999, f %23 L1 değişir | - |
| 2026-10-19 | 0 | ODME λ 0.005-0.5 | etkisiz (çözüm aynı) | - |
| 2026-10-19 | 0 | ODME λ 10/30/100/300/1000 (kendi sayım) | val S_link .992/.983/.966/.941/.901; Dhat/Σb .324/.296/.255/.211/.162 | - |
| 2026-10-19 | 0 | ODME prior global ölçekli (b*Σc/ΣAb) | gizli-link hatası her λ'da ham prior'dan kötü (λ<30) | - |
| 2026-10-19 | 1 | Kâhin (gerçek t±1 ort.) tüm panel | phys 0.699 (S_LWR 0.555); mevcut v:0.6 0.626 (S_LWR 0.446) | - |
| 2026-10-19 | 1 | Smoke: LGBM kalıntı A (dv,dq) D12_I5_N fold1 | S_state 0.9032->0.9382, phys 0.515->0.655, J 0.393->0.427 | - |
| 2026-10-19 | 1 | Smoke: LGBM k modeli B_all / B_cong | S_state 0.903/0.905, phys 0.629/0.592 | - |
| 2026-10-19 | 1 | Tam ay-CV (3 kat, 10 panel): base v06 | S_state 0.9088, phys 0.6031, J 0.4085 | - |
| 2026-10-19 | 1 | Tam ay-CV: A (LGBM dv,dq kalıntı) | S_state 0.9365, phys 0.6949, J 0.4320 | - |
| 2026-10-19 | 1 | Tam ay-CV: B_all / B_cong (k modeli) | S_state 0.9121/0.9106, phys 0.6773/0.6508 | - |
| 2026-10-19 | 2 | Seçici-taklit dev set (3360 pencere), persistence | 0.298 (onset 0.000, ongoing 0.595) | val persistence 0.3017 |
| 2026-10-19 | 2 | Dev ay-CV (3 kat): v3 tarifi | dev 0.802 (onset 0.784, ongoing 0.833) | ~0.72 (tahmin) |
| 2026-10-19 | 2 | Dev ay-CV (3 kat): v4 tarifi (no_freq, hybrid) | dev 0.793 (onset 0.772, ongoing 0.827) | ~0.77 (düşük güven) |
| 2026-10-19 | 3 | E1: ongoing eğitim pencereleri seçici-taklit (v3'e karşı) | dev fam fill +0.0016 (obs +0.0050), 3/3 katta + ; boot SE 0.008 | - |
| 2026-10-19 | 3 | E2: onset eğitim pencereleri seçici-taklit | +0.0004 (gürültü) | - |
| 2026-10-19 | 3 | E3: fill truth etiketleri (onset+ongoing) | -0.0061 (onset -0.020, ongoing +0.007) | - |
| 2026-10-19 | 4 | Final queue (v3 onset + E1 ongoing + fill etiket + thr 0.40) | dev fam fill 0.8129 (v3 +0.011, v4 +0.020), 3/3 kat + | - |
| 2026-10-19 | 4 | S1 = v5 + ODME kendi sayım | val S_link 0.9646 -> 0.9999 | 0.83707 (+0.0316) |
| 2026-10-19 | 4 | S2 = S1 + state A | CV S_state +0.028, phys +0.092 (beklenen +0.024) | 0.85582 (+0.0188) |
| 2026-10-19 | 4 | S3 = S2 + final queue (run_all ile byte-özdeş yeniden üretildi) | dev queue +0.020 vs v4 | 0.84217 (-0.01365 -> final queue LB'de v4'ten ~0.046 KÖTÜ) |
| 2026-10-20 | A1 | Queue ölçüm teşhisi (adversarial/maske/LOMO/sızıntı) | açığın ~0.02'si açıklandı; v3 dev-LB +0.083 -> ölçüt SAĞLANMADI | - |
| 2026-10-20 | A1 | LB probu: v3'te onset satırları = persistence (tek değişken) | - | 0.66642 (v3 0.77351'e göre -0.10709) |
| 2026-10-20 | A2 | State L (±5..60 dk lead/lag, 122 özellik) | S_state 0.9391, phys 0.7070, J +0.0028±0.0009 | - |
| 2026-10-20 | A2 | State E (A + M2 ortalaması) | S_state 0.9389, phys 0.7084, J +0.0029 | - |
| 2026-10-20 | A2 | State LE (L + M2 ortalaması) | S_state 0.9397, phys 0.7117, J +0.0037±0.0013 (30/30 birim) | - |
| 2026-10-20 | A2 | S4 = S3 + state LE (run_all) | beklenen +0.0036 | 0.84565 (+0.00348) |
| 2026-10-20 | A2 | S5 = S2 + state LE (queue v4, run_all merge) | beklenen ≈ 0.859 | **0.85929** (S2'ye +0.00347; EN İYİ) |
| 2026-10-20 | T | P1 = S5 + final ongoing (v4 onset korunur) | dev +0.013 | 0.85208 (-0.00721) |
| 2026-10-20 | T | ODME dual çözücü run_all'da (--odme-solver dual) | S5 ile compare PASS (max |Δf| 2.8e-11), ODME 1408 sn -> 4.8 sn | - |
| 2026-10-20 | T | State B1 eligible/kalite özellikleri (LE'ye karşı) | J -0.00001 (nötr) | - |
| 2026-10-20 | T | State B3 mask augmentation +%50 satır | S_state +.0005, phys +.0049, J +.0009±.0011 (30/30 birim) | - |
| 2026-10-20 | A4 | Physics kalıntı ACF (gözlem - med5 düzgün) | hız lag1 -0.220 vs iid null -0.225; AR(6) R² 0.09 < null 0.18 -> tahmin edilebilir pay ≈ 0 | - |
| 2026-10-20 | C | Queue CTM ileri simülasyon (eğitimsiz, zsh6), dev ongoing | 0.652 vs persistence 0.656, v3 LGBM 0.846; iyi panellerde +0.09-0.14 pers üstü, I10_W 0.136 | - |
| 2026-10-20 | C | LGBM(v3 dev prob) + CTM harman | en iyi bm0.7 0.843 < LGBM 0.846 -> fayda yok | - |
| 2026-10-20 | S | Ongoing: seçici-taklit dev pencereleri eğitime ×3 ağırlık + is_sel (src/task2_predict_sel.py, reports/queue/sel/) | dev ongoing thr0.45: v4 0.839 -> nogap 0.897 (+0.058, 3/3 kat), +at_gate 0.902, +tüm gap 0.905 | hazır: runall_q7_sel (val ongoing hücre %0.86 değişir) |
| 2026-10-20 | S | EIoU kararı (top-k, E[IoU]≈Σp_S/(k+Σp_dış)) | ongoing +0.003-0.005, onset 0.921->0.923 | hazır: runall_q7_eiou, runall_q7_sel_eiou |
| 2026-10-21 | S | Onset seçici-pencere eğitimi ×3 | dev onset 0.921 -> 0.927 (2/3 kat, gürültü) -> kullanılmadı | - |
| 2026-10-21 | S | Onset dev-LB açığı (0.92 vs ~0.757) gürültü mü? | T+30 kuyruk hücreleri cut'ın medyan 30 km/h altında, ±2 km/h %0.8 -> HAYIR; olası neden val olayları (5 olay / 40 onset penceresi) | - |
| 2026-10-21 | S | Rampa özellikleri (src/ramp_feats.py, ≤T, rampa başına train ortalamasıyla normalize) | ongoing sel 0.8973 -> 0.9066 (+0.009, 3/3); onset +0.002 (gürültü) | hazır: runall_q8_sel_ramp (queue agreement S5 0.9942) |
| 2026-10-21 | S | Q8 = S5 + ongoing sel×3 + rampa, thr 0.45 (onset v4) | dev ongoing +0.068 | **0.85655** (S5'e -0.0027 -> LB ongoing ~-0.018; dev ongoing yönü 4/4 TERS: v3->v4, S3, P1, Q8) |
| 2026-10-21 | P | B3 = S5 + state mask augmentation (run_all --state LEB3) | CV J +0.0009 | **0.85955** (+0.00026, EN İYİ) |
| 2026-10-21 | P | PROBE Q0: S5 queue tümü 0 | - | 0.62719 -> S_queue = 0.7737 (onset ~0.757, ongoing ~0.79); physics ≈0.70 (ODME 0.99, state 0.92 varsayımıyla) |
| 2026-10-21 | I | Olay taraması (src/incident_scan.py, reports/queue/incident_scan.csv): train freq<0.02 olan yerlerde kuyruk payı | train ayları %0.8-2.9, val %5.2, private %8.5; nadir-küme günü/30: train 1.9-4.6, val 11.9, private 12.3 | dev-LB ters yön açıklaması: train'de olay az, model tekrar eden kalıbı ezberliyor |
| 2026-10-21 | I | Dev nadirlik (src/dev_rarity.py): onset nadir pencere (>%50 nadir konum) v4 IoU 0.254 vs normal 0.947; karışım ρ=0.28 -> onset 0.753 (LB ~0.757), ongoing 0.781 (LB ~0.79) | ağırlıklı dev v4 mutlak LB'yi tutturuyor | - |
| 2026-10-21 | I | Kimliksiz model (pos/panel/tod/dow/lanes/cap yok) | onset karışım 0.753 -> 0.577 (nadirde de kötü 0.254->0.096) RED; ongoing karışım +0.005 (gürültü) | - |
| 2026-10-21 | I | Dev backtest (src/dev_backtest.py, reports/queue/dev_backtest.csv): bilinen 6 LB farkının işareti | ağırlıksız 2/6, ρ=0.28 3/6. Olay ağırlığı v3->v4'ü düzeltiyor (frekans ezberi); v4->final ve v4->sel+rampa her ρ'da TERS -> ortak neden: seçici-taklit pencerelerle EĞİTİM (taklit artefaktı, dev aynı taklitle ölçülüyor). Karar: taklit-pencere eğitimi kalıcı bırakıldı | - |
| 2026-10-21 | I | EIoU kararı ağırlıklı dev'de (v4) | ongoing nadir 0.633->0.679, ρ0.28 +0.016, ρ0.4 +0.021; onset +0.001 | hazır: runall_b3_eiou (B3'e göre sadece queue, agreement 0.998); beklenen +0.0024 |
| 2026-10-21 | I | Olay öncülü (src/incident_precursor.py, reports/queue/incident_precursor.csv): nadir onset pencerelerinde T, T-15, T-30'da kuyruk başı linkleri | hız/vf 0.98 (serbest), profile göre hız sapması 0.000, akış sapması uzak linklerle aynı -> T'de iz YOK; olay T sonrası başlıyor, onset nadir pencereleri tahmin edilemez (~0.029'luk teorik kalem kapalı). runall_b3_eiou submit edilmedi (beklenen +0.0024, öğretici değil) | - |
| 2026-10-21 | P | PROBE PHYS: B3 + state akış ×1.05 (src/phys_probe_local.py, reports/phys/phys_probe_q105.csv) | yerel ΔS_state -0.0228, ΔS_phys -0.177 (S_LWR -0.264); beklenen ΔLB -0.034 | 0.82789 (Δ -0.0317) -> LB ΔS_phys ≈ -0.160 = yerelin %90'ı. LB physics scorer ≈ yerel proxy; physics gürültü tabanında, gizli kaldıraç YOK |
| 2026-10-21 | I | Veri ölçeği (src/queue_scale_cv.py, reports/queue/scale/): ongoing günde 8 origin + uzak link alt örnekleme | normal 0.838->0.851, nadir 0.694->0.678, ρ0.28 +0.005 -> ölçüt (+0.02) sağlanmadı | - |
| 2026-10-21 | I | CTM nadir ongoing pencerelerde (21 dev) | pers 0.603, CTM 0.556, LGBM 0.694, kâhin seçim 0.710 -> yol kapalı | - |
| 2026-10-21 | I | Yasaklı T+95 verisiyle interpolasyon tavanı (src/future_leak_ceiling.py; KULLANILMADI, sadece teşhis) | onset 0.43, ongoing 0.66 -> liderleri açıklamıyor | - |
| 2026-10-21 | I | Task 1 maskesi değer sızıntısı | maskeli hücrede speed/flow/occupancy/density %0 mevcut; sadece bayraklar -> sızıntı yok | - |
| 2026-10-21 | I | Korunum tahmincisi fizibilitesi: E = Σ|ΔN - dt(q_up - q)| / Σ|ΔN| (train) | D12_I5_N 19.7 (ort. 14.5), D7_I405_N 22.5 (16.4) ≫ 1; ΔN'nin ~yarısı gürültü -> physics yolu kapalı | - |
| 2026-10-21 | P | B3_EIOU = B3 + EIoU queue kararı (v4 modelleri; submissions/runall_b3_eiou) | ağırlıklı dev ongoing +0.016, beklenen +0.0024 | **0.86502** (+0.0055, EN İYİ; ağırlıklı dev işareti doğru) |
| 2026-10-22 | I | Olay tekrarı ölçümü (src/incident_recur.py, reports/queue/incident_recur.csv): bir linkte nadir kuyruk varsa sonraki günlerde aynı linkte nadir kuyruk olasılığı | lag-1 lift: train 1.25, validation 3.58, private 2.59 (lag-7: 1.34 / 2.59 / 1.62) -> eval aylarında olay aynı yerde tekrar ediyor, train'de etmiyor; online refit + olay hafızası gerekçesi | - |
| 2026-10-22 | I | Online refit geri testi (src/online_refit.py, reports/queue/refit/): Mart hafta-ileri, sahte pencereler (maskeli gözlem etiketi), base vs A (ongoing refit) vs B (refit+olay hafızası), Mart ağırlığı {1,3,5}. İLK KOŞU GEÇERSİZ: parquet dosyaları mask_regime sırasında, tarih sırasında değil; hafıza gelecek günleri görüyordu. Takvim günüyle düzeltildi (calendar()) | düzeltilmiş: queue base 0.8345, B_w3 0.8592 (+0.0247, her hafta pozitif, min +0.0043); onset 0.810 -> 0.859 (hafta 2: +0.06, hafta 3: +0.07); ongoing nötr, ağırlığa duyarlı. Olay tekrarı düzeltmeden sonra da duruyor (lag-1 lift val 3.89, private 2.79) | üretim: ongoing v4+EIoU aynen, onset B_w3 |
| 2026-10-22 | I | Resmi pencere tarihleri: val onset 1-7 Mart, private onset 1-9 Nisan (ongoing <=17) | val onset'te önceki Mart günü neredeyse yok -> public LB pratikte değişmez; private onset tüm Mart ile eğitilir (geri testte hafta-3 eşdeğeri). Değişen: haftalık 11 private pencere/28 hücre, günlük 12 pencere (2 val)/33 hücre | hazır: runall_q9_refit_onset, runall_q9_refit_onset_daily (submit edilmedi) |
| 2026-10-22 | P | Q9 = B3_EIOU + onset online refit günlük + olay hafızası (submissions/runall_q9_refit_onset_daily) | val'de sadece 2 onset penceresi değişti (5 ve 7 Mart, 404/2481 Mart satırıyla refit) | 0.86377 (-0.00125; 2 pencere, az veriyle refit zararlı). Final 2. aday: haftalık varyant runall_q9_refit_onset (val'e dokunmaz, public = 0.86502; private onset tüm Mart ile) |
| 2026-10-22 | I | Onset anatomisi (src/onset_anatomy.py, reports/queue/onset_anatomy.csv): 40 resmi train onset penceresi, gerçek etiket | T+30 kuyruk hep train frekansının en yüksek linklerinde (freq_rank ort. 2, medyan 1.25); kâhin-k top-freq IoU 0.777 (medyan 1.0); k 0-9, panel karakteristik; T'de kuyruk hücresinde r=0.99 (iz yok); dünkü kuyrukla IoU 0.15. Kayıp dökümü (LB, ρ=0.28): onset nadir 0.029, ongoing normal 0.018, ongoing nadir 0.013, onset normal 0.009 -> açığın ~%60'ı olay pencerelerinde; private'ta ρ≈0.4 ile daha büyük | olay hafızası tek kaldıraç; sadece eval-ay refit ile öğrenilir -> private (q9 haftalık) |
| 2026-10-22 | I | Refit geri testi nadir/normal ayrımı (src/refit_rarity.py, reports/queue/refit/refit_backtest_rarity.csv) | onset nadir (n=21) base 0.516 -> B_w3 0.729 (+0.21); onset normal (132) 0.857 -> 0.880; ongoing nadir (25) 0.713 -> 0.730; ongoing normal (470) 0.8750 -> 0.8749, ama w1/w5'te -0.019 (kararsız) | ongoing base kalır; final: b3_eiou + q9_refit_onset (haftalık); private beklenen ~+0.014 |
| 2026-10-22 | I | Genişletilmiş hafıza C (mem_last, mem_cnt14, mem_h30_3/7, mem_nb1/3 komşu ±2 link; reports/queue/refit_ext/) vs B, w3 | queue C 0.8537 vs B 0.8592; onset nadir C 0.667 vs B 0.729, normal 0.883 vs 0.880; ongoing normal 0.8695 vs 0.8749 -> C kötü (az Mart verisinde aşırı uyum) | B kalır |
| 2026-10-22 | P | BLEND S50: B3_EIOU + Task1 state = 0.5 bizim + 0.5 anubisyy public notebook çıktısı (CC0; src/blend_public.py; reference/anubisyy_output) | state farkı speed RMSE 1.27, flow 81 vph | **0.86890** (+0.0039) |
| 2026-10-22 | P | BLEND S50+OG50: + ongoing p = 0.5 bizim v4 + 0.5 anubisyy ongoing LGBM (kendi script'imizle yeniden koşuldu, yayınlanan etiketlerle %100 eşleşme; src/anubisyy_ongoing_probs.py, src/blend_ongoing.py), EIoU | 752 hücre değişti | **0.87007** (+0.0012, EN İYİ) |
| 2026-10-22 | P | BLEND S50+OG50+ON50: + onset p = 0.5 bizim + 0.5 anubisyy onset LGBM (CatBoost yok; src/anubisyy_onset_probs.py, src/blend_onset.py) | 12 pencere / 33 hücre | 0.86955 (-0.0005) -> bırakıldı |
| 2026-10-22 | P | BLEND S50+OG50+ONFULL: onset p = 0.5 bizim + 0.5 anubisyy tam onset (0.5 LGBM + 0.5 CatBoost; catboost 1.2.10 kuruldu; kendi koşumumuz yayınlanan etiketlerle %100 eşleşti), EIoU | 14 pencere / 34 hücre | 0.86963 (-0.0004) -> onset harmanı zararlı, bizim onset kalır |
| 2026-10-22 | 1 | Kendi 2. state modeli: CatBoost dv,dq (depth 8, 1500 iter, lr 0.08) aynı LE özellikleriyle, fold 1, 3 panel (src/task1_cat.py, reports/phys/cat_f1/) | J: LE 0.42995, C 0.43141 (+0.0015), LE+C 50/50 0.43166 (+0.0017; S_state +0.0013, physics +0.0084), 3/3 panel | üretim: src/task1_cat_predict.py -> submissions/state_cat |
| 2026-10-22 | 1 | Üretim CatBoost state (src/task1_cat_predict.py, submissions/state_cat; LEB3 önbelleği, 4.5M satır, 4.2 saat) | C vs LE: speed RMSE 0.80, flow RMSE 87 vph; hazır: runall_lec50 (kendi: B3_EIOU + state LE/C 50/50), blend3_s_og50 (state 0.25 LE + 0.25 C + 0.5 anubisyy + OG50) | submit bekliyor (günlük hak bitti) |
| 2026-10-23 | P | LEC50 = B3_EIOU + state 0.5 LE + 0.5 kendi CatBoost (submissions/runall_lec50; tamamen kendi) | CV +0.0017 | 0.86559 (+0.0006; CV'nin ~1/3'ü) |
| 2026-10-23 | P | BLEND3 = state 0.25 LE + 0.25 CatBoost + 0.5 anubisyy + OG50 (submissions/blend3_s_og50) | - | **0.87018** (+0.0001 vs 0.87007, EN İYİ) |
| 2026-10-23 | - | Final 2. aday güncellendi: blend3_s_og50_q9 = blend3_s_og50 + private onset q9 haftalık (src/apply_queue_diff.py; 28 hücre, hepsi private; public = 0.87018 beklenir) | - | submit edilmedi |
| 2026-10-23 | 1 | Kendi 3. state modeli: 2B U-Net (gün ızgarası link×288, 18 kanal: maskeli v/q, maske, rejim, dow profil, rampa, tod, geometri; profil + artık çıkış; başka günün maskesiyle augmentation 0.5; 30 ep × 300 adım, MPS 38 dk; src/task1_unet.py, reports/phys/unet_f1/), fold 1, 3 panel, LE girdi değil | UN tek başına J -0.0109 (S_state 0.9287, LWR 0.448); hata korelasyonu LE-UN v 0.62-0.73 / q 0.75-0.83 (LE-C 0.95-0.98); LEUN50 +0.0020; **LE+C+UN 1/3 +0.0044** (S_state +0.0033, physics +0.021; S_state 3/3 panel artı, J 2/3); UN flow zayıf (rmse_q_lane 37.6 vs LE 32.0), hız R3'te iyi | - |
| 2026-10-23 | 1 | U-Net 3 seed ort. (42/43/44) + v/q ayrı harman (src/task1_unet_blend.py, reports/phys/unet_f1/unet_blend_cv.csv): v = (1-wv)/2·(LE+C) + wv·UN, q aynı wq ile; wv {0.33,0.5,0.6} × wq {0,0.2,0.33} | UNavg tek başına -0.0052 (1 seed -0.0109); **v33q20 +0.0055** (S_state +0.0040, physics +0.027), 3/3 panel artı (+0.0072/+0.0012/+0.0079); ızgara düz (+0.0042..+0.0055), wv 0.5+ D7_I210_W'de eksi; LECUN1 +0.0044 | üretim: 3 seed, v33q20 |
| 2026-10-23 | 1 | Üretim U-Net: fold 0 (9 ay), 3 seed × 30 ep (MPS ~2 saat; submissions/state_unet), val/private tahmini (src/task1_unet_predict.py), state v33q20 = v 0.335 LE + 0.335 C + 0.33 UN, q 0.4 LE + 0.4 C + 0.2 UN | UN vs LE: speed RMSE 2.30, flow 104 vph; NaN 0. Hazır: runall_un33 (tamamen kendi; vs runall_lec50 0.86559, beklenen ~0.867), blend4_un_og50 (blend3_s_og50 + state 0.5 v33q20 + 0.5 anubisyy, src/swap_state.py; state farkı v 0.37 / q 11.5; beklenen ~0.871), blend4_un_og50_q9 (+ private onset q9, 28 hücre) | submit bekliyor |
| 2026-10-23 | P | BLEND4 = blend3_s_og50 + state 0.5 v33q20 (LE+C+UN) + 0.5 anubisyy (submissions/blend4_un_og50) | fold-1 CV +0.0038 (LEC50 üstüne), beklenen +0.0007 | **0.87046** (+0.0003 vs 0.87018, EN İYİ; CV'nin ~1/13'ü, anubisyy harmanında seyreldi) |
| 2026-10-22 | - | Final 2. aday: blend_s50_og50_q9 = S50+OG50 + private onset q9 haftalık (11 private pencere, 28 hücre; public = 0.87007) | - | submit edilmedi |
| 2026-10-20 | S | ODME yeniden yorum: repo baseline S_ODME 0.8359 + S1 (+0.158) -> S_ODME ≈ 0.994; KL/λ100 probu GEREKSİZ (muhtemelen zarar) | - | - |
| 2026-10-20 | P | Hazır probe'lar (submit bekliyor): runall_q0 (queue=0, S_queue ölçümü), runall_kl1 (ODME KL/MaxEnt μ=1, src/odme_dual_predict.py --method kl), runall_lam100, runall_q6 (ongoing+CTM özellik, src/task2_predict_ctm.py), runall_b3 (run_all --state LEB3) | ODME sim: KL S_link 1.0, sıfır yol %0; sentetik f* deneyinde KL > L2 λ0.05 > λ100 (reports/odme/odme_synth.csv). B3 vs S5 state RMSE v 0.29, q 2.9/şerit | beklenen: q0 ~0.637, kl1 0.860-0.869, lam100 0.839-0.858, q6 0.857-0.861, b3 ~0.860 |

### Faz C — Üreteç fiziği ve CTM (2026-10-20, src/queue_latent.py, scenario_shift.py, queue_ctm.py, queue_ctm_blend.py, reports/queue/latent/, reports/queue/ctm/)
- Host (#742761, repo README): queue etiketi gürültülü gözlemden değil "underlying state"ten. Veri sentetik, kalibre trafik akış modeli. Organizatör mükemmel cevap S_total 0.9945 (queue 1.0, physics ~0.96)
- Etiket gürültüsü etkisi küçük: serbest akış hız gürültüsü σ 1.3-2.1 km/h; σ=2-5 ile gözlem vs gizli etiket ufuk IoU 0.98-0.999. Eşiğe ±2 km/h yakın kuyruk hücresi %0.3-0.5. Dev-LB açığını AÇIKLAMIYOR
- Resmi train pencerelerinde (80) gözlem etiketi onset kuralını tam tutuyor (erken kuyruk 0/40, T+30 39/40); zaman düzgünleştiriciler (t3/t5/st3) erken kuyruk üretiyor -> gizli durum ≈ gözlem, düzgünleştirme etiketi bozar
- Senaryo kayması yok: val/private kuyruk oranı (0.020/0.022) ve akış p99/şerit train ay aralığında (Şubat 0.021). synthetic_release_v1.json capacity_multiplier 1.2 tüm split'lere ait
- FD tam üçgen: serbest akışta v/vf medyan 0.9997; sıkışıkta q/(w(kj-k)) medyan 1.000 (w = C/(kj-kc), fd_parameters). Kuyruk başları fd kapasitesi düşük linklerde, deşarj ≈ o kapasite
- CTM (Godunov, alt adım 0.6 sn, T'den 6 adım; başlangıç k sıkışıkta w kj/(v+w), serbestte q/vf; kuyruk başı düğüm kapasitesi = son 6 adım akışı; kaynak 0; giriş talebi son 3 adım): ongoing dev I405_N 0.832 (pers 0.722), I210_E 0.813 (0.708), I5_S 0.748 (0.661), I5_N 0.735 (0.645); ama I10_W 0.136 (0.531), I10_E 0.512 (0.640), I405_S 0.695 (0.704)
- Kaynak terimleri (bugünkü q_i-q_{i-1}, profil, topoloji CONN maskesi) ortalamada kötüleştiriyor (gürültü; topoloji 258 düğümün 159'unda CONN). Profil giriş talebi kötüleştiriyor. Linkleri ters çevirmek S/W panellerini bozuyor (order_index = artan milepost, üreteç bu yönde)
- I10_W: kuyruklar eşiğe yakın (v 55-75, cut 63-68), arada rampa dengeleri (+800/-800); CTM kritik durumları serbest akışa ya da derin sıkışmaya itiyor
- Karar: CTM tek başına / harman olarak kullanılmıyor (dev'de LGBM 0.846 > en iyi harman 0.843). Olası sonraki kullanım: CTM çıktısı LGBM'e özellik (LB probu gerekir)
- Host (#749601): ödül incelemesinde Task 2 kodu her pencere için T sonrası tüm satırlar silinerek yeniden çalıştırılacak, tahminler değişmemeli. Bizim day_min_r S[:t0+1] ile nedensel; eğitim ayları val/private'tan önce

### Adım 1 — Queue ölçüm teşhisi (2026-10-20, src/queue_diag2.py, reports/queue/adv_*, maskmimic_*, lomo_*, leakage_audit.md)
- Çapalar: v2 queue = organizatör persistence'ı (87000 hücre %100 aynı) -> LB ≈ 0.3017; v3 = 0.3017 + 0.12508/0.30 = 0.7187. v4 ve final queue LB'de hiç izole edilmedi
- a) Adversarial: dev vs val+private pencere AUC 0.841 (onset 0.60, ongoing 0.98) — ayrımın çoğu maske artefaktı (val'de aynı gün önceki pencerenin ufuk+tampon boşluğu 18 satır: gap_max 16.9 vs 7.6). Sadece durum özellikleri: AUC 0.643; val/private ongoing kuyrukları daha küçük (geçmiş kuyruk hücresi 285/211/196). Hücre düzeyi AUC ≈ 0.5. Dev vs resmi train pencereleri durum özellikleriyle 0.474 (üreteç doğru)
  - Dev'i val'e benzetecek şekilde yeniden ağırlıklandırma: v3 0.802 -> 0.790 (açığın 0.012'si)
- b) Maske taklidi: release formatı sorun değil (geçmiş maskesiz, 1-74% hücreler değerli, T satırı rejim oranında maskeli, boşluk dağılımları aynı). Bozulmuş girdiyle v3 +0.0002
- c) LOMO (9 kat): v3 0.8065 (3 kattan +0.0045). Aylar 0.784-0.831. İleri zincirleme -0.008 (daha az veriyle karışık)
- d) Sızıntı: T sonrası veri kullanan özellik yok. Freq profilleri dev'de sonraki ayları da içerir (val'de sadece önceki) — küçük. Resmi scorer sadece eligible hücreleri sayar; dev eligible truth ile v3 0.878 -> açık +0.159'a büyür, persistence yine tutar (0.305)
- KABUL ÖLÇÜTÜ SAĞLANMADI: persistence dev-LB farkı 0.02 içinde, v3 +0.07..+0.16. Açıklanmayan ~0.06: ya Mart/Nisan senaryosunda kuyruk dinamiği farklı (daha küçük kuyruklar), ya da LB truth (queue_true) bizim tanımımızdan farklı; persistence onset'te her truth altında ~0 olduğu için bunları ayıramaz
- config/synthetic_release_v1.json scenario_contract: capacity_multiplier 1.2, incident_severity 1.0, val/private seed organizatörde (train için ayrı değer verilmemiş -> farklı olduğu kanıtlanmadı). Organizatör: val persistence pencere IoU std 0.3752
- Kural gereği queue modeline dokunulmuyor. Önerilen tek değişkenli LB probu: v3 submission'ında sadece onset satırları persistence (hepsi 0) -> Δ = 0.30*0.5*onset_v3 -> v3'ün LB onset ve ongoing IoU'su ayrışır (submissions/probe_v3_onset_pers, hazır; 19 Eki'de günlük limit (5/5) nedeniyle 400 aldı, 20 Eki'de gönderilecek)

### P1 sonucu (2026-10-20)
- Δ = -0.00721 = 0.15 * (final_og - v4_og) -> final ongoing, v4 ongoing'den LB'de 0.048 KÖTÜ
- S3 kaybı (-0.0455 queue) = 0.5*(Δon + Δog) -> Δon (v3 onset - v4 onset) ≈ -0.043. Yani S3'te HEM onset (v3, freq'li) HEM ongoing (E1 + fill etiket + thr .40) v4'ten kötüydü; dev ikisini de iyi göstermişti
- v4 onset LB ≈ 0.714 + 0.043 ≈ 0.757. Dersler: freq özellikleri onset'te zararlı (Faz 2 bulgusuyla tutarlı); ongoing'de fill etiket/E1/thr .40 paketi zararlı. Dev set queue model seçimi için güvenilmez (3/3 karşılaştırmada yön yanlış)
- Sonraki queue probları (S5'e göre): q2b (sadece eşik .45) fill etiketi içermediği için P1'den ayrı bilgi verir; q4 onset

### State tuning (2026-10-20, src/task1_resid3.py, reports/phys/resid3_*)
- B1 nötr. B3 tüm 30 birimde pozitif ama küçük (+0.0009 J ≈ +0.0009 LB); üretim scripti yok. B2 (early-stop) iptal edildi; yarıda kalan koşuda L dv modeli 2000 turda hâlâ iyileşiyordu -> daha çok tur state'e katkı verebilir (ileride)
- ODME artık saniyeler içinde (dual), sonuç official ile 1e-10 içinde aynı

### Queue aday varyantları (2026-10-20, src/queue_variants.py, submissions/queue_variants/<ad>/, reports/queue/variants_*)
- v4 yeniden eğitimi gönderilen dosyayla 0 hücre farkla aynı. 763 sn
- Dev (aile, fill / elig; v4 .7929 / .8649), v4'e fark (SE80):
  - q1_final_og (v4 onset + final ongoing, SADECE ongoing değişir): +.0138 / +.0132 (SE .009), 3/3 kat — ama dev daha önce final'i ters sıralamıştı
  - q2a ongoing thr .40: +.0014 ; q2b .45: +.0011 (sadece ongoing)
  - q3 eligible-only etiket+ağırlık: ±.001 (nötr); q3og (sadece ongoing) +.0005; q3on (sadece onset) -.0011
  - q4 anubisyy onset özellikleri thr .55 (sadece onset): -.0014 / -.0026
  - q5 ongoing iki model ortalaması: .35 -.0009, .40 +.0012, .45 +.0004
- Q1 dışındakiler dev'de ~1 SE içinde, LB gürültüsünün çok altında -> dev ayırt edemiyor. Ongoing eşik eğrileri .40-.45'te tepe (public notebook'larla aynı yön)
- Tek değişkenli prob adayları (S5'e göre): sadece ongoing -> q1, q2a/b, q3og, q5; sadece onset -> q3on, q4

### anubisyy notebook karşılaştırması (2026-10-20, reports/anubisyy_comparison.md)
- Onların LB yolu: 0.84833 -> 0.86025 (ODME sıkı çözüm +0.012) -> 0.86446 (state +0.004) -> 0.86517 (onset blend +0.0007). ODME bizimle eşdeğer (biz kesin çözüyoruz)
- Queue farkları: onset sadece T-60..T-5 girdi (T satırı yok), FREQ YOK, raw lag 1/2/3/6/9/12 + milepost ±2 komşu, LGBM+CatBoost blend, thr 0.55; ongoing tek LGBM, thr 0.40; etiket/aday/kayıp sadece ELIGIBLE hücrelerde, satır ağırlığı 1000/eligible hücre, ufuk eligible kapsaması >= %60 şartı
- Yeni yorum: S3 final queue'nun onset'i v3 onset'iydi (freq'li). v4 ≈ 0.77 ise final ≈ 0.7245 -> final ongoing ≈ 0.735 > v3 ongoing 0.723; kaybın kaynağı ONSET (freq özellikleri LB'de onset'i bozuyor; Faz 2'de de freq onset'te zararlıydı). Bu, v4'ün mutlak değerine bağlı bir çıkarım
- Test: S5 + queue (v4 onset + final ongoing) = S5'e göre sadece ongoing satırları değişir -> final_ongoing - v4_ongoing doğrudan ölçülür

### LB sonuçları 20 Eki (S3, S4)
- S3 (S2 + final queue) 0.84217: final queue, v4 queue'dan LB'de 0.0455 kötü (0.01365/0.30). Dev'de final v4'ten +0.020 iyiydi -> dev sırayı TERS çevirdi, fark LB gürültüsünün (±0.015 queue) 3 katı. Adım 1 kuralı doğrulandı: dev ölçütü sağlanmadan queue değişikliği LB'de güvenilmez. Şüpheli: ongoing için fill etiketleri ve/veya E1 seçici pencereleri + eşik 0.40
- S4 (S3 + LE) 0.84565: +0.00348, beklenti +0.0036 ile birebir -> state/physics proxy'leri LB'ye iyi aktarılıyor
- S5 (S2 queue v4 + LE state) = 0.85929, beklentiyle birebir; LE'nin etkisi iki farklı queue ile de +0.0035 (S3->S4 ve S2->S5) -> bileşenler toplanabilir. En iyi submission: submissions/runall_s5/submission.csv

### Adım 2 — State lead/lag + ensemble (2026-10-20, src/task1_resid2.py, reports/phys/resid2_*)
- Aynı 3 kat ay-CV, A ile eşli. M2 = farklı parametreli/seed'li LightGBM + L özellikleri (CatBoost kullanılmadı). E = A+M2 ortalaması, LE = L+M2 ortalaması
- LE en iyi: S_state 0.9365 -> 0.9397, S_LWR 0.549 -> 0.574, physics 0.695 -> 0.712, J 0.4320 -> 0.4357 (fark std 0.0013, 30 kat×panel biriminde)
- Rejim (A -> LE): S_state R1 .9401/.9428, R2 .9380/.9410, R3 .9313/.9354; physics R1 .810/.820, R2 .722/.736, R3 .553/.580. 10/10 panel iyileşti
- Kuyruklu hücreler: rmse_v 4.73 -> 4.36, MAE k 10.95 -> 9.98, k bias +0.16% (guard ok, kuyruk silinmiyor)
- Beklenen LB ≈ +0.35*0.0032 + 0.15*0.0168 ≈ +0.0036 (S2'deki %78 aktarımla ~+0.003)
- Not: koşu 16.4 saat sürdü; tahmin süresinin büyük kısmı gece uykusu (uyanıkken kat başına tahmin ~28 dk)

### Adım 1 — LB probu sonucu (2026-10-20)
- Δ = -0.10709 = 0.30 * 0.5 * (onset_v3 - onset_pers); onset_pers ≈ 0 -> v3 LB onset IoU ≈ 0.714; v3 queue 0.7187 -> v3 LB ongoing ≈ 0.723. Persistence LB ongoing ≈ 2*0.3017 = 0.603
- Dev'e karşı (fill truth): onset 0.784 -> LB 0.714 (dev +0.07 iyimser); ongoing 0.833 -> LB 0.723 (dev +0.11 iyimser). Persistence ongoing dev 0.595 vs LB 0.603 tutuyor
- Sonuç: açık iki koşulda da var, büyüğü ongoing'de ve modele özgü (persistence tutarken model tutmuyor) -> val ongoing dinamiği (daha küçük kuyruklar, adversarial bulgusu) model davranışını bozuyor; truth tanımı farkı onset'te de olası

### Adım 3 — Tek giriş script'i (2026-10-20, src/run_all.py)
- Aşamalar subprocess ile mevcut script'leri çağırır (model kodu kopyalanmaz): state (task1_resid_predict), queue (final | v4), odme (resmi builder, split başına, *_public_reference birleştirme), merge (resmi merge_submissions). manifest.json: argümanlar, versiyonlar, iki repo commit'i, girdi/çıktı sha256, aşama süreleri ve bellek
- --compare: task bazında fark + proxy (state RMSE, queue uyuşma, odme rel L1); geçme kuralı: hız RMSE < 0.25, akış/şerit RMSE < 6, queue uyuşma >= %99, odme rel L1 < %1
- Hafif test (odme+merge, mevcut state_A ve v4 queue CSV'leri): submission S2 ile BYTE-ÖZDEŞ, compare PASS. ODME 1408 sn (tek çekirdek)
- Tam yeniden üretim (~39 dk, tepe ~4.1 GB) onay bekliyor; bundan sonraki her submit run_all.py'dan çıkacak

### Adım 4a — Physics kapanış testi (2026-10-20, src/phys_acf.py, reports/phys/acf_*)
- 10 panel, 1583 link, 535 sn. Merkezli filtre beyaz gürültüye de ACF verdiği için null = aynı maske+filtreden geçmiş iid gürültü
- Hız gürültüsü fiilen beyaz (lag1 -0.220 vs null -0.225); N'de küçük fazlalık (serbest +0.029, kuyruklu +0.073) yavaş sinyal sızıntısı deseni, renkli gürültü değil
- Aynı hücrede hız-akış gürültü korelasyonu ≈ 0 (serbest 0.001, kuyruklu 0.067)
- AR(6) açıklanan varyans N 0.09, null 0.18 -> gerçek kalıntı null'dan DAHA az tahmin edilebilir. S_LWR vekili (|ΔN| AR ile/olmadan) 0.912 vs null 0.819 -> pay yok
- r2 (gözlem - v06) yavaş sönen düzgün interpolasyon hatası; A modeli bunu zaten kâhin seviyesine indirdi
- KARAR: physics için yeni modelleme yok. 0.70 tavanı ölçüm gürültüsü kaynaklı. Physics = A'nın tam eğitimi (yapıldı)

### Faz 4 — Birleştirme ve submit planı (2026-10-19)
- Üretim: src/task1_resid_predict.py (state A, 9 ay, 3M satır, seed 42; CV yolu resid_cv_A ile 2.2e-16 birebir; yeniden tahmin bit-özdeş), src/queue_final_predict.py (onset v3 aynen, 80/80 pencere v3 ile özdeş), ODME = repo *_public_reference (kendi sayım+kendi prior, λ=0.05). Versiyonlar: python 3.14.6, numpy 2.5.1, pandas 2.3.3, pyarrow 24.0.0, lightgbm 4.7.0, scipy 1.18.0 (versions.json her çıktı klasöründe)
- Tek değişkenli merdiven (her adım doğrulandı: sadece ilgili task satırları değişiyor):
  - S1 submissions/s1_odme_own: v5 + ODME kendi sayım. Beklenen: S_link kısmı +0.0018 kesin; S_od/S_attr/S_dev bilinmiyor (public notebook aynı değişiklikle +0.031)
  - S2 submissions/s2_stateA: + state A. Beklenen ≈ +0.35*0.028 + 0.15*0.092 ≈ +0.024 (state gürültüsü düşük: milyonlarca hücre; risk ay aktarımı)
  - S3 submissions/s3_queue_final: + final queue. Beklenen ≈ 0.30*0.020 ≈ +0.006; LB queue gürültüsü ±0.0045 S_total -> ~1.3 SE, tek başına anlamlı ölçülemeyebilir
- Kırılım tahmini (S3, val): state ~0.937*0.35, physics ~0.69*0.15, queue ~0.77-0.79*0.30, ODME ~0.84+?*0.20
- Günlük hak: 19 Eki'de 5/5 kullanıldı; S3 20 Eki'de (UTC 00:00 sonrası)
- LB sonuçları: S1 0.83707 (+0.0316; S_link payı sadece +0.0018 -> kalan ~+0.03 S_od/S_attr/S_dev'den: kendi ay sayımı talebi gerçeğe çok yaklaştırıyor; public notebook +0.031 ile tutarlı). S2 0.85582 (+0.0188; CV beklentisinin %78'i -> state/physics ay aktarımı iyi, physics proxy biraz iyimser olabilir)

### Faz 3c — Queue tek-değişkenli deneyler (2026-10-19, src/queue_exp3c.py, reports/queue/exp3c_*.csv)
- v3 yeniden koşusu cv_results ile birebir (1e-16). 413 sn, 4 thread
- E1 (ongoing seçici-taklit pencereler) tek temiz kazanç ama küçük: fam fill +0.0016, obs +0.0050, ongoing obs +0.0099. Geç-sönme kaybı .031 -> .017, erken-sönme .041 -> .060 (artık kuyruğu erken bitiriyor)
- E2 nötr. E3: fill etiketleri onset'i bozuyor (-0.020) -> Faz 3a'daki "onset extent_over truth artefaktı" hipotezi zayıfladı; fill truth'un onset'te T+30'u eski serbest akışla doldurması etiketi de bozuyor. LB truth'u bilmeden onset FP'lerinin ne kadarının gerçek hata olduğu belirsiz
- Ongoing eşik eğrileri: E1 0.40'ta 0.8413 (0.5'te 0.8359); v3 0.45'te tepe. Dev'den seçmek meşru (LB değil)
- Sonuç: queue tarafında denenen fikirlerin hepsi LB gürültüsünün (±0.015) altında. Dev-LB açığı (v3 ~0.08) hâlâ açıklanamadı; kesin çözüm için tek değişkenli bir LB çapası gerekir
- Birleşim önerisi (Faz 4): E1 pencereleri + ongoing için fill etiketi + ongoing eşik 0.40; onset v3 olduğu gibi

### Faz 3a — Queue hata analizi (2026-10-19, src/queue_error_analysis.py, reports/queue/error_*.csv/png)
- v3, fill truth, 3360 dev penceresi; FP/FN hücreleri tek kategoriye atanıp kayıp hücre bazında bölündü
- Kayıp payı: onset %56.5, ongoing %43.5
  - onset extent_over %39.7 (en büyük) — bu FP hücrelerinin %81'i T+30'da GÖZLEMSİZ; fill truth eski serbest akış hızını taşıyor. Model eminken (medyan p 0.997) "yanlış" sayılıyor -> büyük kısmı truth tanımı artefaktı olabilir; LB truth altta yatan durum
  - ongoing: dissipation_early %10.6 + late %8.1 (ongoing kaybının %43'ü), threshold %11.0, extent_under/over %6.7/%7.0
  - onset wrong_link %7.4, onset_timing %2.9
- En kötü 100 = IoU 0 olan küçük kuyruklar (medyan birleşim 5 hücre): 62 onset (çoğu olay-benzeri, darboğaz dışı link; doğru linkte p≈0.001), 38 ongoing (T'de 1-2 linklik kalıntı kuyruğun kaybolacağı tahmin ediliyor)
- Ayrışan tanımlayıcılar: olay-benzeri ani düşüş %39 vs %10; ufuk kuyruğu 5 vs 90 hücre; panel D7_I10_W ve D12_I5_S fazla; ongoing'de daralan kuyruk %76 vs %58
- Hipotez desteği: (1) ongoing büyüme/sönme dinamiği + seçici-taklit eğitim en güçlü model kaldıracı; (2) truth tanımı uyumsuzluğu en büyük ölçüm sorunu; (3) şok dalgası özellikleri ve panel eşikleri zayıf

### Faz 3b — Public notebook özetleri (reports/queue/public_notebooks_summary.md)
- ODME'yi kendi split sayımlarıyla çözmek yeashusemwal'da +0.031 LB (bizim Faz 0 düzeltmesi). Sıkı çözüm (anubisyy L-BFGS-B maxiter 8000: 0.848 -> 0.860) bizde zaten kesin (nnls/dual)
- Onset sadece T+30 (+0.027), top-1 fallback; daha fazla fallback zarar
- Ongoing: ufuk özellik, eşik 0.40-0.45, iki model olasılık ortalaması (+0.002); seçici-benzeri eğitim +0.0003 (onlarda küçük)
- LB'ye zarar verenler: tek link boşluk doldurma, BFS yayılım, CNN ongoing, top-2/3 fallback
- İki yazar da yerel doğrulamanın modelleri yanlış sıraladığını söylüyor

### Faz 2b — Queue model ay-CV (2026-10-19, src/queue_cv.py, reports/queue/cv_*.csv, error_decomp.csv)
- 3 ardışık ay katı, 4 thread, 467 sn. Persistence kontrolü dev set ile birebir (fark 0.0)
- Dev (aile toplamlı, fill truth): pers 0.293, v3 0.802, v4 0.793. v4-v3 = -0.009, dev'de anlamlı (eşli SE 0.0017, ~5 SE; onset -0.012, ongoing -0.006; 8 panelin 7'sinde v3 önde) ama 80 pencerelik LB'de görünmez (SE 0.011) -> freq özelliği dev'de hafif faydalı
- Truth tanımı onset'te kritik: fill (altta yatan ffill) onset 0.77-0.78, obs (sadece gözlenen) 0.91-0.92. LB future_coverage=1.0 -> fill daha yakın
- Hata ayrışımı (v4 genel / onset / ongoing):
  - kaçınılabilir bias (1-train): 0.044 / 0.0 / 0.089
  - varyans (train - train_dev): 0.063 / 0.084 / 0.042
  - VERİ UYUMSUZLUĞU (train_dev - dev): 0.093 / 0.144 / 0.043  <- baskın, özellikle onset
  - dev - LB: v4 0.023, v3 0.082
- Yorum: sentetik eğitim pencerelerimiz (rastgele origin / kendi onset tanımımız) organizatör seçicisinden farklı. Onset'te seçici "kuyruk ilk kez T+30'da + kapsama >= 0.7" pencereleri alıyor; ongoing'de sadece kuyruğun değiştiği pencereler (persistence <= 0.9). Eğitimi seçici-taklit pencerelerle yapmak doğrudan bu açığa saldırır
- Spearman (n=3) 0.5: dev sırası v3>v4>pers, LB tahmini v4>v3>pers. v3/v4 farkı her iki tarafta gürültü içinde -> sıra testi bilgilendirici değil; v3'ün dev-LB açığı (0.08) dev'in hâlâ iyimser olduğunu gösteriyor
- Ongoing eşik eğrisi düz (0.35: 0.827, 0.40: 0.829) -> eşik değişikliği gerekmez

### Faz 2a — Queue dev set onarımı (2026-10-19, src/queue_devset.py, reports/queue/)
- Seçici kuralı (config/synthetic_release_v1.json + competition_contract.json + 80 resmi train penceresinin yeniden üretimi):
  - geçmiş (T-60..T-5) eligible kapsaması >= 0.7 (= window_index.history_coverage)
  - ufukta (T+5..T+30) altta yatan durumda kuyruk var
  - koşul: bir link T-60..T-5'te >= 2 görünür kuyruklu gözlem -> ongoing, değilse onset
  - ALTTA YATAN durumda persistence IoU <= 0.9 (maximum_persistence_iou)
  - ayın 1. gününden kronolojik açgözlü seçim, 360 dk aralık İKİ koşul arasında ortak, 5+5'te durur
- KİLİT BULGU: sentetik kuyruklar basamak fonksiyonu; stabil kuyrukta persistence ~1.0. Bu yüzden ongoing pencereler sadece kuyruk kapsamı ufukta DEĞİŞİRKEN açılıyor (çoğunlukla akşam dağılması). Eski sentetik ongoing'lerimiz rastgele origin'di -> LB açığının ana kaynağı muhtemelen bu
- Resmi Haziran train pencerelerini yeniden üretim: 73/80 tam, 79/80 ±10 dk
- Dev set: 336 örnek x (5+5) = 3360 pencere, ay başına zincirleme örnekleme (gün paylaşmıyor); 9 tam sözde-sürüm
- Kalibrasyon (organizatör persistence'ı, T-5 görünür): dev 0.298, resmi train 0.306, val LB 0.3017 -> kalibre. Eski 0.394 ölçümümüz farklı persistence tanımıydı
- Ay bazında 0.292-0.303 düz; panel 0.255-0.339
- LB gürültüsü: 80 pencere ortalamasının SE ≈ 0.0145 (queue) ≈ ±0.0045 S_total. İki model farkının SE 0.013-0.016. Dev set SE ≈ 0.003
- Hafif fark: val/private ongoing geçmiş kuyruk boyutu (211/196 hücre) dev'den (285) küçük, persistence ile r=0.20 -> etkisi < 0.01

### Faz 1 — Physics kalıntı modeli (2026-10-19, src/task1_resid.py, reports/phys/)
- Ay-CV: 3 ardışık ay katı (Haz-Ağu, Eyl-Kas, Ara-Şub), eğitim 6 ay, panel başına 300k örnek, 400 tur, 4 thread, seed sabit. 34 dk, tepe 7.2 GB
- Hızlı S_state resmi score_task1.py ile birebir (fark 0); hızlı physics resmi scorer ile birebir
- A kazandı: S_state 0.9088 -> 0.9365, S_FD 0.985 -> 0.987, S_LWR 0.412 -> 0.549, physics 0.603 -> 0.695. 10/10 panel ve 3/3 kat iyileşti, katlar arası std ~0.001
- Rejim (A vs v06): R1 phys 0.810/0.741, R2 0.722/0.625, R3 0.553/0.443
- Kuyruklu hücreler: rmse_v 6.56 -> 4.73, MAE k 18.0 -> 10.9, k bias +0.06% (kuyruk silinmiyor, guard ok). Kuyruksuz rmse_v 2.64 -> 1.56
- B (k hedefi) yoğunlukta en iyi (kuyruklu MAE k 8.9) ama q'ya dokunmadığı için S_state düşük -> ileride A+B birleşimi denenebilir
- Beklenen LB etkisi (v5'e göre): +0.35*0.028 + 0.15*0.092 ≈ +0.024
- Tavan analizi (Bölüm 57): A (0.695) kâhin komşu-ortalamasına (0.699) ulaştı; yani "kâhin" tavan değil, yerel bilgiyle ulaşılabilecek seviye. 0.70 ile organizatör mükemmel cevabı (~0.96) arasındaki açık mimari değil, ölçüm kaynaklı: S_LWR hedef hücrelerde gerçek GÖZLEMİ (gürültülü) referans alıyor ve paydası sum|ΔN_true| çok küçük (birikimin ~%0.75'i). Hücre başına bağımsız ölçüm gürültüsü komşulardan tahmin edilemez; her hedef hücre bu gürültünün ~2 katını kalıntıya ekler, R3'te (%50 maske) etki iki katına çıkar. Bu yüzden 0.70 üstü kazanç denoising'den gelmez; ancak gürültünün yapısı (ör. hız-akış gürültüsünün aynı hücrede korelasyonu, ardışık zamanda otokorelasyon) modellenirse küçük pay kalır. Hipotez: train'de gözlem - pürüzsüz tahmin kalıntısının otokorelasyonu ölçülerek doğrulanabilir
- Üretim için A'nın tüm 9 ayla eğitilip val/private'a uygulanması Faz 4'e kaldı

### Faz 0 — ODME ölçülebilirlik (2026-10-19, src/odme_scan.py, reports/odme/)
- f* hiçbir split'te yok (contract: path_flow_truth = organizer_ground_truth). score_task4.py sadece S_link hesaplar; repo'nun "public_reference"ı prior+sayım çözümü, S_od için anlamsız
- HATA: repo builder'ın baseline_submission'ı her split için TRAIN sayımlarıyla çözülüyor (build_task4_odme_artifacts.py: train_f = solve(A, train_counts, prior)). v1-v5 ODME bu. Kendi sayımlarla val S_link 0.9646 -> 0.9999
- Her split tek departure_time (statik). Prior zayıf: ΣAb/Σc ≈ 0.7-2.3, corr(Ab, c) ≈ 0.3; NNLS uyumsuzluğu yolların %57'sini sıfırlayarak çözüyor (prior'da %0)
- λ ancak 10+ etkili. Gizli-link proxy'si her zaman en küçük λ'yı seçer (bağlar çok redundant) -> S_od/S_dev için λ seçemez
- Aylar arası yol payı L1 ~0.6 (prior'lar da öyle) -> talep aydan aya stabil değil
- Dual semismooth-Newton çözücü nnls ile özdeş (rel L1 ≤ 4e-15), 57 sn -> <0.1 sn

### Task 3 physics (2026-10-19, src/task3_proxy.py, src/task3_fast.py)
- Scorer: tam alan = eligible hücreler; hedefte bizim değer, diğerlerinde gerçek. S_FD = üçgen FD (fd_parameters otoriter), S_LWR = organizatör sınır akışlarıyla korunum; akışlar gözleme projekte -> pratikte hedef hücrelerde N=(q/v)L hatasının ardışık farkı / sum|dN_true|
- Yerel proxy: train'de outflow=q_true, inflow=q_true+dN_true/dt-on+off akışı üretip resmi scorer'a --boundary-flux. Gerçek 0.9968, baseline 0.3243 (doküman 0.3242 ile aynı). task3_fast.py aynı sonucu 1 dk'da veriyor
- Topoloji akış dengesi gerçeği açıklamıyor (E 13-16) -> korunum tahminci olarak kullanılamaz
- Kâhin (gerçek t±1 ortalaması) physics 0.71 -> gürültü tabanı yüksek
- task1_interp.py --reconcile: k=q/v ayrıca interpole edilir. D12_I5_N train:
  - none: S_state 0.8978, phys 0.382 (J=0.35S+0.15P=0.3715)
  - v (tüm hücre): 0.8149 / 0.465 (0.355) ; q: 0.8231 / 0.424 ; geo: 0.8299
  - v:0.6 (sadece v̂<0.6vf hücrelerde v=q̂/k̂): 0.9074 / 0.5475 (J=0.3997)  <- en iyi, ikisini de iyileştiriyor
  - v:0.8 0.8915/0.534 ; v:0.9 0.8590/0.513 ; q:0.8 0.8602/0.537
- Eşik taraması D12_I5_N (J): 0.5 0.3978 | 0.55 0.3992 | 0.6 0.3997 | 0.65 0.3992 | 0.7 0.3983 -> 0.6
- Tüm paneller train (v:0.6 vs none): S_state 0.9050 -> 0.9115 (10/10 panel iyileşti); physics proxy 0.528 -> 0.626 (S_LWR 0.300 -> 0.446, S_FD 0.983 -> 0.985), 10/10 panel iyileşti
- Beklenen LB etkisi ~ +0.35*0.0065 + 0.15*0.098 = +0.017
- v5 submit 2026-10-19 (ref 57389543): public LB 0.80550 (v3 0.77351, +0.032). Beklenen state+physics etkisi +0.017; kalan ~+0.015 muhtemelen no_freq queue (v4 ayrı submit edilmediği için ayrıştırılamıyor)

### Task 2 genelleme teşhisi (2026-10-19, src/task2_diag.py)
- Val girdileri: T satırı ve önceki saatler Task1 maskesiyle %46-69 dolu (train tam ~%83); T-60..T-5 geçmişi maskesiz (~%83)
- v_cut = 0.6 * links.free_speed_kmh doğrulandı (40 onset penceresinin 39'unda T+30 kuyruk; topology vf ile 34)
- Aylar arası kuyruk oranı büyük oynuyor (D12_I5_N Tem 0.036, Kas 0.068) -> freq özelliği ay-özel
- Dürüst proxy: Haziran (resmi train pencereleri ayı) ve Şubat (eşik holdout) dışarıda, maskeli/hibrit girdi, LOMO freq
  - plain freq all 0.893 | LOMO all 0.872 | LOMO no_freq 0.897 | no_day 0.848 | no_freq_no_day 0.872 (80 pencere, persistence 0.394)
  - Hiçbiri LB ~0.72'yi açıklamıyor; val persistence 0.30 < train 0.39 -> val pencereleri daha zor (olaylar?). Val tahminlerinde hat hatası bulunmadı
- day_min_r/corr özellikleri faydalı (çıkarınca düşüyor); freq onset'te zararlı (0.913 -> 0.955)
- v4: submissions/v4_queue_nofreq/submission.csv = v2 state + hybrid/LOMO/no_freq LGBM queue (thr onset 0.3, ongoing 0.35) + NNLS odme. v3 ile queue uyuşma %99.2

### LB analizi (v2, 0.648)
- Tahmini kırılım (baseline tablosu + train skoru): state ~0.90x0.35=0.315, queue ~0.30x0.30=0.09, physics ~0.35x0.15=0.05, ODME ~0.84x0.20=0.17
- Kalan açık: queue (0.30 ağırlık, persistence ~0.30) ve physics (0.15, tavan ~0.95). LB 1. 0.924
- Onset pencerelerinde kuyruk sadece T+30'da (organizatör onaylı) -> persistence onset'te hep 0 IoU

### Task 1 interpolasyon (src/task1_interp.py, 2026-10-19)
- Yöntem: kalıntı r = gözlem - profil (maskeli train'den link x gün x saat ortalaması). r aynı gün içinde zamanda (link başına) ve uzayda (order_index sırası, km mesafe) lineer interpole edilir; ters mesafe ağırlığı w = 1/(d + t0)^p, uzay mesafesi s_km ile adıma çevrilir. Tahmin = profil + ağırlıklı r. Hız [5, vf+4.5 ve 120.75] arası kırpılır
- Grid (D12_I5_N, 72 kombinasyon, src/tune_task1.py): en iyi t0=2, p=1, s_km=0.25, w_prof=0 -> 0.8978. Sadece zaman 0.8923; profil ağırlığı eklemek kötüleştiriyor
- Tüm train resmi scorer: S_state 0.7320 -> 0.9050. En zayıf D7_I405_N 0.878, en iyi D7_I10_W 0.937. Kalan hata: hız RMSE 1.8-4.2 km/h, akış 26-48 veh/h/şerit
- Profil maskeli train'den, yani hedef hücreler profile girmiyor (sızıntı yok)
- Validation/private çıktısı henüz üretilmedi, submit yok

### Veri kalite profili (2026-10-16, src/profile_quality.py, notebooks/quality_D12_I5_N/profil.txt)
- Yapı temiz: (timestamp, link_id) tekrar yok, her gün 258x288 tam ızgara, bayraklar (is_imputed/is_score_eligible/is_missing) pct_observed ile %100 tutarlı, maskesiz hücreler maskeli dosyayla birebir aynı
- Eksik: pct_observed=0 hücreler (%16.5) tamamen boş; maske train'de ek %24.8; validation %43, private %42 boş
- pct 1-74 hücreler (%7) değerli ama skorlanmıyor; dağılımı eligible ile aynı -> girdi olarak kullanılabilir
- Hız tavanı 120.75 km/h (clip); 34k ardışık tekrar bundan, takılı sensör değil
- occupancy 0.75'te tavanlanıyor (%1.8, sıkışıkta) -> density_occ (max 213.75) sıkışıkta güvenilmez; k = q/v kullan
- Ağ: links.csv = fd_parameters.csv; lwr_mainline_topology.csv farklı (vf 16-27 km/h, kapasite, şerit sayısı). Scorer hangisini kullanıyor kontrol et
- has_sensor=0 linklerde de veri var (oran aynı)
- flow > kapasite %0.18 (max 1.09x), v > vf+5 yok, k > k_jam yok
- Ramp akışları çok küçük (medyan 15-19 veh/h, ana hat 4588) -> birim/ölçek belirsiz; Task 3 için kritik
- Validation/private: Task2 pencereleri 10 x 18 adım = 180 adım tüm linkler boş (beklenen)
- v1_baseline/submission.csv doğrulandı: 6,980,503 satır, şablonla aynı id sırası, null/dup yok

### Discussion bulguları
- Organizatör onaylı (#742350): queue_onset pencerelerinde kuyruk HER ZAMAN sadece T+30'da; T+5..T+25 boş. Kullanmak serbest
- Bir katılımcı onset pencerelerinin ~0.7 IoU tavanı olduğunu iddia ediyor (#742761)
- Dataset bir kez sızıntı yüzünden yeniden kuruldu; eski notebook'lar eski veriye göre olabilir

### LB durumu (2026-10-16)
- 482 takım, 8744 submission. 1.: 0.92406; top-10 ~0.900+; top-20 ~0.890
- Öne çıkan public notebook'lar: lamhuy8904/traffic-flow-bench-pipeline (44 oy), soukeaizenz/0-809-lb-top-50-open-sourcing-physics-pipeline, yeashusemwal/trafficflowbench-trees-u-net-cnn-lb-0-87, anubisyy/catboost-lgbm-nnls-pipeline-top-50-0-86517

### EDA notebook (2026-10-19, notebooks/eda_kaggle/)
- Kaggle Code'da paylaşmak için İngilizce, sadece betimsel EDA (kullanıcı kararı: model/skor ipucu yok — onset T+30, hangi parametre dosyası, kendi yöntemlerimiz notebook'ta YOK)
- Dosyalar: build_notebook.py (nbformat ile .ipynb üretir; düzenleme buradan, sonra `python build_notebook.py`), traffic-flow-bench-eda.ipynb (push edilecek), traffic-flow-bench-eda.executed.ipynb (yerel çıktılı kopya), kernel-metadata.json (id andacdemirdelen/traffic-flow-bench-guided-eda-of-all-10-corridors, is_private true, GPU/internet kapalı, competition_sources)
- Veri kökü otomatik: TFB_ROOT env, /kaggle/input/*/.../config/corridors.json (1-4 seviye), yoksa ../../data/kaggle_public
- Bellek: koridor başına her 3. train günü (91 gün), D12_I5_N tam (4.1 GB RAM). Yerelde 64-73 sn
- 8 bölüm + başta 10 maddelik "Key findings" kutusu; grafiklerde bulgu başlıkları (headline/subtitle yardımcıları)
- Yeni bulgular: validation/private'ta her Task 2 origin sonrası T+5..T+90 (18 adım) tüm linklerde gizli, öncesindeki 1 saat maskesiz. Kuyruk saatleri tüm koridorlarda 13-20; D7_I10_W ve D7_I210_E neredeyse hiç kuyruk yok. Task 4 weak prior A·b ile link count korelasyonu 0.1-0.4, toplam oranı 0.7-2.3x. Sıkışık FD dalı ayrık kümeler
- Durum: Kaggle'a push EDİLMEDİ (kullanıcı "şimdilik yeterli" dedi). Push ve public yapma kararı kullanıcıda; önce private push + çıktı kontrolü
