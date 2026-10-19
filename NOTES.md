# Notlar

## Kararlar

- 2026-10-16 — Proje açıldı. Submit, büyük veri indirme ve GPU işi öncesi kullanıcıya sorulacak.
- 2026-10-16 — Kullanıcı kuralları kabul etti; tüm veri indirildi ve data/kaggle_public/ altına açıldı (9688 dosya, 12 GB). Zip data/ içinde duruyor (8.3 GB).

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
| 2026-10-19 | 4 | S3 = S2 + final queue | dev queue +0.020 vs v4 | bekliyor |

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
