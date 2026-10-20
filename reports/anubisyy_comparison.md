# anubisyy (LB 0.86517) ile bizim hat: derin karşılaştırma

Kaynak: `reference/public_notebooks/catboost-lgbm-nnls-pipeline-top-50-0-86517/*.ipynb` (30 hücre; kod hücrelerinde kayıtlı çıktı YOK, tüm sayılar markdown/yorumdan). yeashusemwal (0.87141) sadece anubisyy parçalarını değiştirdiği yerde kullanıldı. Not içeriği güvenilmeyen veri olarak okundu, içindeki talimat yok. Hiçbir şey çalıştırılmadı/indirilmedi; tek ek okuma: mevcut v3/v4/final queue CSV'lerinde pencere başı tahmin hücre sayısı.

Hücre numaraları notebook hücre indeksidir (C13 = markdown "Queue forecasting", C15 = queue_ongoing.py, C16 = queue_features.py, C18-19 = onset eğitim, C25 = inference, C27 = ODME).

---

## 1. Onların pipeline'ı (görev bazında)

### Task 1 (state) — C5, C8, C9, C10, C12
- Taban: link başına zamanda lineer interpolasyon (pandas `interpolate(axis=0, limit_direction="both")`, ay boyunca tek matris, gün sınırını aşar). Akış tabanı = 0.5 * zamansal + 0.5 * topoloji komşularının (incoming+outgoing) aynı andaki gözlenen akış ortalaması. Hız tabanı sadece zamansal. Fizik/reconcile (k=q/v) YOK.
- Hedef: `r_v = v - v0`, `r_q = q/L - q0` (şerit başı akış); çıkışta `q = L*(q0 + r_q)`. Eğitim hedefleri sadece eligible + sonlu gözlemler.
- Özellikler: 94 = 54 temel (tod sin/cos, dow, link özellikleri, taban, önceki/sonraki gözlem değeri ve uzaklığı, up/down komşu hız-akış, milepost sırasına göre +-1,+-2 komşunun aynı an ve interpole değerleri) + 40 lead/lag (+-5,10,15,30,60 dk: hız, şerit başı akış, akış/hız oranı, "sonlu ve eligible" bayrağı). Bayrak: sonlu ama ineligible hücreyi modele söylüyor.
- Eğitim: her gün için 20/30/50 % rejiminden biri (gün hash'i), eligible+sonlu hücreler kendi simüle maskesiyle saklanır, öznitelikler maskeli diziden hesaplanır, etiket değişmez. İki bağımsız maske seed'i, gün başı 750 hedef -> 3.675 M satır (Haz-Oca). Şubat: tek sabit maske, 420k hedef (erken durdurma + seçim).
- Model: CatBoost RMSE, derinlik 8 (9000 tur) ve derinlik 10 (4500 tur), lr 0.06, l2=10, Bernoulli subsample 0.85, early stop 150 (Şubat), GPU. Hız ve akış için ayrı, 4 model, eşit ortalama. Final model Şubat'sız (Haz-Oca).
- Kırpma: hız `clip(1,180)`, akış `clip(0,150000)`.
- CV: zaman bölmesi, tek ay (Şubat), Şubat aynı zamanda erken durdurma verisi.

```python
# C9 prepare_state.py: lead/lag bloğu (kalite bayrağı dahil)
for offset in (-12,-6,-3,-2,-1,1,2,3,6,12):
    rr=rows+offset; ...
    extra.extend([np.where(valid,vv,np.nan), np.where(valid,qq,np.nan),
                  np.where(valid,qq/np.maximum(vv,1),np.nan),
                  np.where(valid&e[rr,cols]&np.isfinite(vv),1,0)])
```
Physics için ayrı modelleme yok: "There is no separate physics model or conservation penalty" (C5).

### Task 2 ortak (C13)
- Girdi: sadece pencere geçmişi (T-60..T-5, 12 satır). T satırı HİÇ kullanılmıyor (`assert history.timestamp.max() < origin`, C25). Gün düzeyi/koridor geneli özellik yok, frekans (link x saat kuyruk oranı) özelliği yok.
- Ufuk: 6 adım, T+5..T+30. Kuyruk = `speed <= cut` (cut = 0.6*vf; v_cut yoksa 0.6*free_speed).
- Eğitim verisi: train/mainline_states ham (maskesiz) günler, Haz-Şub; Şubat doğrulama.
- Etiket/truth: GÖZLENEN hız (sonlu) <= cut; kayıp ve doğrulama sadece `is_score_eligible & isfinite` hücrelerde.

```python
# C15 collect()
y = ((future_speed <= panel.cut[None, :]) & np.isfinite(future_speed)).reshape(-1)
valid = (eligible[t+1:t+HORIZON+1] & np.isfinite(future_speed)).reshape(-1)
if valid.sum() < .6 * len(valid): continue
...
train_X.append(X[valid]); train_w.append(np.full(valid.sum(), 1000 / valid.sum()))
```

### Task 2 aday pencere / origin tanımı (C15 `candidate_origins`, `choose_origins`)
```python
queued = (speed <= panel.cut[None, :]) & eligible & np.isfinite(speed)   # SADECE eligible kuyruk
active = queued.any(axis=1)
if np.isfinite(speed[t-HISTORY:t]).mean() < .7: continue                 # geçmiş kapsaması
if not active[t+1:t+HORIZON+1].any(): continue                            # ufukta kuyruk şart
onset  : not active[t-12:t].any() and not active[t]
ongoing: active[t]
```
- Ongoing eğitiminde pencere başına en fazla 2 pencere / gün / koşul, aralarında >= 72 adım (6 saat), `rng.permutation` ile rastgele. Persistence <= 0.9 gibi bir selector benzetmesi YOK.
- Ongoing modeli hem onset hem ongoing pencerelerinden eğitildi (özellik 10: koşul bayrağı 0/1). Satır ağırlığı `1000/eligible_hücre_sayısı` -> her pencere eşit ağırlık (skor pencere-IoU ortalaması).
- Onset eğitimi (C18/C19): `cand` içinden sadece "t-1 aday değil" olan origin (bir koşunun ilki) -> kuyruk tam T+30'da ilk kez görünür. Etiket `v[t+6] <= cut`, satırlar `e[t+6]` eligible hücreler, ağırlıksız.

### Ongoing (C13, C15, C25)
- LightGBM, 36 özellik + ufuk adımı h (link x h satırı; 6 ufuk bağımsız doğrudan). Özellikler: panel_no, link indeksi (`arange`), vf, kapasite/şerit, şerit, uzunluk, saat/sin/cos/dow, koşul bayrağı, son gözlenen hız/akış (vf, cap ile normalize), son gözlemin yaşı, son3/önceki3 ort. hız-akış ve farkları, ort/min/max, geçerli oran, `queue_history.sum`, `last_queue`, incoming/outgoing komşu ortalaması (son hız, son akış, son kuyruk), pencere-geneli geçerlilik oranı ve min hız.
- Parametre: lr 0.045, leaves 31, max_depth 8, min_data 250, ff 0.85, bagging 0.85, l2 8, max_bin 127, 700 tur, early stop 60 (Şubat ağırlıklı logloss). Binary logloss.
- Eşik 0.40 (Şubat IoU ile). Yorum: "Later ongoing models and a lower threshold looked better locally but lost public-LB score" (C13). Düzeltme/boşluk doldurma yok.

### Onset (C13, C16, C18, C19, C25)
- Sadece T+30 tahmin edilir (T+5..T+25 sıfır). 83 özellik = ongoing 36 + kısa lag'ler (1,2,3,6,9,12: hız ve akış) + min hız(son3/6), son3-önceki3 farkları, komşu incoming/outgoing son3 ort., +-1,+-2 milepost komşularının (uzaklık, son/recent hız, son/recent akış, son kuyruk) + milepost.
- LightGBM: 2 aşamalı (lr 0.04, leaves 15, min_data 100, l2 10, ff 0.9; 900 tur early stop 70, sonra +1500 tur devam), tek T+30 etiketiyle. CatBoost: derinlik 7, lr 0.05, l2 10, 3000 tur early stop 150 (GPU), eğitim geçmişinin bir kopyasında hücrelerin %5'i silinir (hız/akış NaN, kalite bayrağı 0), etiket aynı.
- Olasılık `0.5*lgb + 0.5*cat`, eşik 0.55, hiçbiri geçmezse en yüksek tek link (top-1). Şubat'ta tam 6 adım IoU ile seçildi.

```python
# C25
probability = ((1-weight)*onset_model.predict(X) + weight*onset_cat.predict_proba(X)[:, 1])
selected = probability >= 0.55
if not selected.any(): selected[np.argmax(probability)] = True
prediction[-1] = selected
```

### Task 4 (C26-C28)
- Aynı amaç: 1/2||Af-c||^2 + lambda/2||f-b||^2, f>=0, lambda=0.05; A sadece sayımı olan linkleri içerir; her split'in KENDİ sayımı ve prior'ı (`task4/<panel>/<split>/synthetic_link_counts.csv`, `synthetic_weak_prior.csv`). Prior'a doğrudan çıpa, ağırlık/ölçek yok.
- Çözüm: link-uzayı kapalı form ridge `u = b + A^T (AA^T + lam I)^-1 (c - A b)`, negatifler kırpılıp L-BFGS-B'ye başlangıç (maxiter 600, ftol 1e-12, gtol 1e-5) -> sonra bu başlangıçtan devam: `maxiter=8000, ftol=1e-14, gtol=1e-6, maxcor=30`. Warm start asset'ten (`odme_initial.parquet`), `REBUILD_WARM_START` ile yeniden üretilebilir. Yazar kendisi "success certificate değil" diyor.

### Merge (C30)
- state.csv / queue.csv / odme.csv anahtarlarla `submission_key.csv` sırasına birleşiyor; eksik eşleşme assert; Task 3 satırı yok. Ek postprocess yok.

### yeashusemwal'ın anubisyy üstündeki değişiklikleri (sadece bunlar)
- Task 2: anubisyy onset aynen. Ongoing: kendi ensemble'ının olasılığı ile anubisyy LightGBM olasılığı 50/50, eşik 0.45 (0.40 değil).
- Task 1: kendi (LGBM+U-Net) ile anubisyy CatBoost 50/50 ortalama.
- Task 4: kendi, FISTA + sıkı L-BFGS-B (anubisyy'nin 8000-iter çözümünden +0.0006 daha yakınsak).

---

## 2. Rapor edilen skorlar (bileşen ayrıştırmamız için)

anubisyy (hepsi public LB, "isolated" ekler):
| Adım | LB | Delta | Not |
|---|---:|---:|---|
| ODME 600 iter (yakınsamamış) | 0.84833 | - | |
| ODME maxiter 8000 | 0.86025 | +0.01192 | Task 1/2 sabit (bu bizde zaten kesin çözülüyor) |
| + kalite özellikli state + CatBoost 8/10 ensemble | 0.86446 | +0.00421 | Şubat state proxy 0.934905 -> 0.941802, 10/10 panel iyileşti |
| + onset blend (LGBM + CatBoost-dropout, thr 0.55) | 0.86517 | +0.00071 | onset IoU kazancı = 0.00071/0.15 = +0.0047; Şubat proxy (tam 6-adım IoU) 0.881986 -> 0.892520 |
- Rank 48/320 (5 Eki). Ongoing/onset için ayrı LB mutlak değeri YOK. Persistence proxy'si kodda yazdırılıyor ama çıktı kayıtlı değil.
- Şubat onset proxy 0.8925 (gözlenen-eligible etiket, sadece T+30 hücreleri) bizim 'obs' dev onset 0.91-0.92 ile aynı seviyede -> yerelde onlar bizden iyi DEĞİL.

yeashusemwal (anubisyy bileşenleri için dolaylı bilgi, hepsi LB):
- Kendi Task 2'sini anubisyy Task 2'siyle değiştirmek +0.0095. Kendi CNN onset'ini anubisyy onset'iyle değiştirmek +0.0103 -> anubisyy onset IoU'su yeashusemwal CNN onset'inden ~0.069 yüksek (0.0103/0.15). Yani anubisyy'nin gücü onset.
- Ongoing: iki modelin olasılık ortalaması +0.0023 (ongoing IoU +0.015), eşik 0.40 -> 0.45 +0.0011 (ongoing IoU +0.0073). "At 0.40 the average predicts too many queued cells" ve "queues in scored months smaller than in training".
- Task 1: anubisyy CatBoost tek başına onların Task 1'inden -0.0005; 50/50 ortalama +0.0023 (farklı aileler, farklı hatalar). Held-out S_state: LGBM 0.9377, U-Net 0.9312, 70/30 0.9397 (conservation proxy 0.58-0.59, bizimkiyle karşılaştırılamaz).
- Task 4 kendi sayım +0.031 (bizim S1 ile birebir), yakınsama +0.0006.
- Negatifler: top-2 fallback + eşik 0.52 -0.0070; top-3 fallback -0.0003; 1-link boşluk doldurma -0.0042; CNN ongoing -0.0049; uzun-bağlam 4 tohum -0.0012.
- Selector-benzeri ongoing eğitimi (persistence <= 0.9) ensemble'a %20 payla +0.0003 (küçük, gürültü).

Bizim bileşen çıkarımımız (ÇIKARIM, ölçüm değil):
- ODME: ikisi de aynı amaç, aynı sayım/prior, ikisi de neredeyse yakınsamış. Bizimki kesin; onlarınki en fazla ~0.0006 gevşek. Fark ~0.
- State+physics: Şubat proxy 0.9418 (onlar) vs CV 0.9397 (bizim LE, farklı aylar/maske) -> onlar <= ~0.002 toplam avantajlı veya eşit; yeashusemwal anubis Task 1'ini bizim seviyemizde (-0.0005) buluyor. Bizim fizik taraf avantajımız (v:0.6 reconcile, A/LE) onlarda yok.
- Toplam fark 0.00588 -> geriye queue'da ~0.004-0.008 toplam = **queue IoU farkı ≈ 0.013-0.027** onlar lehine. Bizim v4 queue LB ≈ 0.77 (dev 0.793 - 0.023, v3->v5 kalıntısından ~0.77; güven orta) ise anubisyy queue ≈ 0.78-0.80.

---

## 3. Yan yana tablo

| Boyut | Onlar | Biz | Olası etki |
|---|---|---|---|
| T satırı | Hiç yok (geçmiş T-60..T-5) | Task 1 maskeli dosyadan T satırı kullanıyoruz (%46-69 dolu), hibrit | Bizim avantajımız olmalı; kötüleştirmez ama dağılım farkı riski (val maskesi) |
| Gün-düzeyi / koridor özellikleri | Yok | day_min_r, corr_* var (no_day dev'de zarar) | Dev'de fayda, LB'de belirsiz (adv AUC ongoing 0.98 kaynağı) |
| Frekans özelliği | Yok. Ama link indeksi + milepost + panel_no özellik -> darboğaz kimliği tüm aylardan örtük öğrenilir | v4: freq YOK (LOMO), pos/panel/lane var; v3 freq vardı (LB'de v4 ~+0.05 iyi) | Link kimliği ay-bağımsız yumuşak freq gibi çalışıyor; bizde `pos` var ama link-kimlik çözünürlüğü daha düşük olabilir |
| Etiket (truth) | Gözlenen hız <= cut, kayıp sadece eligible&sonlu | Gözlenen hız <= cut, tüm sonlu hücreler (pct 1-74 ineligible dahil), eligible filtresi yok; dev'de ayrıca 'fill' truth | Eligible filtresi eksikliği: ~%7 hücre gürültülü etiket; LB sadece eligible sayıyor. ETKİ BİLİNMİYOR |
| Aday origin | Eligible kuyruk, geçmiş >= %70 sonlu, ufukta kuyruk şart, onset = bir koşunun ilk origin'i; max 2 pencere/gün/koşul, 6 saat ara | Onset: sessiz + T+30 kuyruk, skip 18 adım. Ongoing: T'de kuyruk (her türlü gözlenen), ufuk şartı YOK, rastgele 2/gün | Ufuk-şartsız ongoing: kuyruğu tamamen kaybolan pencereleri de içerir, selector'ın hiç seçmeyeceği tür |
| Satır ağırlığı | 1000/eligible_hücre (pencere eşit) | Yok (hücre eşit) | Pencere-IoU metriğiyle uyum; küçük/büyük kuyruk dengesi |
| Onset modeli | LGBM(2 aşama, 15 leaf)+CatBoost(d7, %5 dropout) 50/50, thr 0.55, top-1 | Tek LGBM 63 leaf/min50/400 tur, thr 0.30, top-1 | Onlar regülarize + erken durdurma + blend; biz yüksek kapasite, eşik düşük |
| Onset özellikleri | 83: kısa lag'ler (1,2,3,6,9,12), milepost +-2 komşu last/recent hız-akış-kuyruk, yaş, geçerlilik | r_last/mean/min/slope, +-3 link r_last/fc_last/q_now, up/dn min5, T satırı | Onlarda şok dalgası zamanlaması (raw lag) var, bizde özet istatistik |
| Ongoing modeli | Tek LGBM 700 tur es60, 31 leaf, min250, l2 8; 36 özellik + h | LGBM 400 tur sabit, 63 leaf, min50, 0.8 ff; ufuk + freq_h(v4'te yok) | Biz overfit'e daha yatkın; dev-LB açığı büyük |
| Ongoing eşik | 0.40 (Şubat IoU), "düşük eşik LB'de kötü" | 0.35 | İki yazar da ~0.40-0.45'i LB'de tercih etmiş; dev'imiz eşiğe düz (0.35: 0.827, 0.40: 0.829) |
| Doğrulama | Tek ay Şubat, thr ve early stop aynı ay | 3 ay CV dev seti (3360 pencere) seçici-taklit | Bizimki daha sağlam ama LB'ye aktarılmıyor (dev-LB +0.07-0.11) |
| Onset/ongoing eğitim karışımı | Ongoing modeli onset pencereleri de görüyor | Ayrı | Küçük |
| State taban | Zamansal lineer interp; akış 0.5 zaman + 0.5 komşu; hız tabanı fizik yok | Zaman+uzay IDW interp, v:0.6 reconcile | Bizim taban daha güçlü |
| State model | CatBoost d8 (9000) + d10 (4500) GPU, 3.675M satır, 2 maske kopyası, Şubat'sız final | LGBM 400 tur (A), L (+-5..60 dk lead/lag, 122 öz.) + M2 ortalaması (LE), resmi train maskeleri, 9 ay | Onlar çok daha fazla tur ve erken durdurma; biz sabit 400 tur (olası underfit) |
| Kalite bayrağı | sonlu&eligible bayrağı her ofsette | Yok (ineligible sonlu hücre düz gözlem) | Küçük/orta, proxy ile ölçülebilir |
| Mask augmentation | 2 bağımsız simüle maske/gün | Tek resmi maske | Orta |
| Physics | Ayrı yok | A/LE ile 0.712 proxy, v:0.6 | Biz önde |
| ODME | L-BFGS-B 8000 it (warm start asset) | Dual semismooth-Newton, kesin | Eşit/biz biraz önde |
| Merge/postprocess | Yok | run_all.py manifest + byte-özdeş yeniden üretim | Biz önde (süreç) |

---

## 4. Aktarılabilir fikirler (öncelik sırasıyla, en fazla 8)

Genel kural: state/physics için yerel proxy güvenilir (LB'ye %78-100 aktarım). Queue için dev ölçütü SAĞLANMADI (dev-LB +0.07-0.11) -> her queue değişikliği tek değişkenli LB probu ister (günlük 5 hak). LB queue gürültüsü ±0.015 (+-0.0045 S_total).

### 1. Ongoing eşiği 0.35 -> 0.40/0.45 (v4 üzerinde, tek değişken)
- Etki: Task 2 ongoing. İki yazarın bağımsız bulgusu (anubisyy: düşük eşik LB'de kaybettirdi, 0.40'ta kaldı; yeashusemwal: 0.40->0.45 +0.0011 = ongoing IoU +0.007, gerekçe val/private kuyrukları eğitimden küçük). Bizim adversarial bulgumuz aynı yönde (val ongoing geçmiş kuyruk 211/196 vs dev 285).
- Ölçüm: dev eşiğe düz olduğu için dev'de zarar vermez (0.35 -> 0.40 dev ~+0.002). LB: v4'ün kayıtlı olasılıklarından yeniden eşikleyip tek probe. Not: bizim kalibrasyonumuz (ağırlıksız, 63 leaf) onlarınkinden farklı; 0.40 ile 0.45 iki probe gerekebilir.
- Risk: düşük (ongoing 40 pencere, +-0.011 SE ile 0.001-0.003 düzeyindeki etki tek probe'da görünmeyebilir). Efor: düşük (sadece predict + eşik).

### 2. E1 ongoing + fill etiketi + thr 0.40'ı v4 onset'in ÜZERİNE test et (S3 yeniden yorum)
- Çıkarım: S3 queue = v4'ten -0.0455. Ama S3'ün onset'i v3 onset'iydi (freq özellikli, `full` girdi, LOMO değil) ve v3'ün LB queue'su 0.7187 iken v4 ~0.77. S3 queue ~0.7245 = v3 + 0.006 -> final'in ongoing kısmı (E1 + fill + 0.40) LB'de v3 ongoing'e göre ~+0.012 IoU kazandırmış olabilir; kayıp "v3 onset + v3 girdi tarifi"nden gelmiş olabilir, ongoing değişikliklerinden değil. Bu hipotez S3'ü açıklayan en sade okuma (v3->v4 farkı dev'de -0.009, LB'de +0.05: dev sırayı freq/LOMO/hibrit girdi için ters çevirdi).
- Ölçüm: tek probe: S5 (v4) onset + final ongoing (E1) -> LB. Tek değişken (sadece ongoing satırları).
- Risk: orta (ongoing fill etiketi + E1 + thr 0.40 üç değişken birlikte; ayrıştırılamaz). Efor: düşük-orta (kod mevcut, v4 ongoing'i queue_final ongoing'iyle değiştir).

### 3. Eligible-only kayıp + pencere-eşit ağırlık (1000/eligible_hücre) ve aday seçimi eligible kuyrukla
- Etki: Task 2 onset+ongoing. Etiket tanımı farkı (aşağıdaki "Bayrak" bölümü). Onlar bunu kullanıyor, biz kullanmıyoruz.
- Ölçüm: dev'de eligible truth versiyonumuz var ('obs'/'fill' + eligible). Ama dev göstermez (açığın nedeni belki bu); LB probu şart. Ayrı probe: sadece ağırlık, sonra sadece eligible filtre.
- Risk: düşük-orta. Efor: düşük (fit içinde `y.notna()` yerine eligible maske + ağırlık).

### 4. Regülarize LGBM + erken durdurma (leaves 31, min_data 250, l2 8, lr 0.045, early stop held-out ay) hem onset hem ongoing; iki farklı modelin olasılık ortalaması
- Etki: Task 2. Model kapasitemiz yüksek (63 leaf, min_data 50, sabit 400 tur), dev-LB açığı model'e özgü (persistence tutuyor, model tutmuyor): daha basit/regülarize model genelleme farkını azaltabilir. yeashusemwal iki farklı ailenin ortalaması +0.0023 (ongoing IoU +0.015).
- Ölçüm: dev kullan ama sadece "kötüleşmiyor" kontrolü olarak; asıl karar LB probu. Eşik yeniden seçilmeli (kalibrasyon değişir) -> Fikir 1 ile karışmasın, ayrı probe.
- Risk: orta (eşik/kalibrasyon kayması). Efor: orta.

### 5. Onset özellikleri: raw kısa lag'ler + milepost +-2 komşu last/recent/kuyruk + yaş/geçerlilik + link indeksi ve milepost
- Etki: Task 2 onset (anubisyy gücü onset). Şok dalgası zamanlaması bizde özet istatistiklerde kayboluyor; link indeksi/milepost ay-bağımsız darboğaz kimliği (freq'in LOMO'suz zararlı hali olmadan).
- Ölçüm: dev onset hem 'obs' hem 'fill' ile; sadece kötüleşmediğini kontrol; LB probu. Onların kendi Şubat proxy farkı küçük (0.8820 -> 0.8925 tüm blend için).
- Risk: orta. Efor: orta.

### 6. Onset: CatBoost + %5 hücre dropout kopyası, 50/50 blend, thr 0.55
- Etki: onset IoU +0.0047 (LB +0.0007 toplam, gürültü altında). Ek: onset eşiğimiz 0.30 vs onların 0.55, ama veri kalibrasyonları farklı. Pencere başı tahmin hücresi bizim v4: ortalama 3.9 (medyan 4, %15 tek hücre) vs dev truth ufuk hücresi ortalama 3.66 (medyan 3) -> sayı olarak aşırı tahmin kanıtı yok.
- Ölçüm: LB probu tek başına anlamlı çıkmaz (+0.0007 << ±0.0045). Tek başına yapma; diğerleriyle birleştir.
- Risk: düşük. Efor: orta-yüksek (CatBoost, GPU gerekirse kullanıcıdan izin).

### 7. State: eligibility bayrakları + mask augmentation + daha çok tur/erken durdurma + farklı model ailesi (CatBoost veya MLP) üçüncü üye
- Etki: Task 1 + Task 3. anubisyy kalite özellikleri + ensemble +0.0069 Şubat proxy (+0.0042 LB); yeashusemwal iki farklı Task 1 ailesi ortalaması +0.0023 LB. Bizim LE iki LGBM ortalaması (fark küçük). Bizde sabit 400 tur var, onlarda 9000/4500 tur es150 (muhtemel underfit).
- Ölçüm: yerel 3-kat ay-CV, S_state + physics proxy, güvenilir (S4/S5 beklenti ile birebir). Sıralı deneyler: (a) bayraklar, (b) 1500 tur + es, (c) ek maske kopyası, (d) CatBoost.
- Risk: düşük. Efor: orta-yüksek; CatBoost/GPU/ağır yerel işlem için ÖNCE kullanıcıdan izin (kural). Beklenen +0.001-0.003 toplam.

### 8. Ongoing aday penceresine "ufukta kuyruk var" + "geçmiş >= %70 sonlu" şartı, onset pencereleri de ongoing modelinde
- Etki: Task 2 ongoing. Selector ufukta kuyruk şartı koyuyor; bizim v4 ongoing'i rastgele T'de-kuyruklu origin'ler (ufuk şartsız) kullanıyor. E1 (selector-taklit) bunun daha katı hali; Fikir 2 ile örtüşür.
- Ölçüm: dev (selector benzeri) + LB probu. Risk: düşük. Efor: düşük.

---

## BAYRAK: queue etiket tanımı ve eligibility farkları (dev-LB açığı hipotezimiz)

1. Onlar eğitimde ve seçimde sadece `is_score_eligible & isfinite` hücreleri sayıyor (aday kuyruk tanımı, etiket, kayıp, doğrulama, pencere ağırlığı). Biz `isnan` dışındaki tüm gözlenen hücreleri (pct 1-74 ineligible dahil, ~%7) hem kuyruk aday tanımında hem etikette kullanıyoruz.
2. Onlarda pencere başına `valid.sum() >= 0.6*len` (ufuk eligible kapsaması) şartı var; bizde yok. (LB future_coverage ölçümümüz 1.0 çıkmıştı.)
3. Onlar gözlenen etiketi bilerek kullanıyor ve "latent state ile uyumsuzluk" riskini yazıyor. Biz 'fill' truth'u dev ölçütü yaptık; E3 fill etiketle eğitimi onset'te bozdu (-0.020).
4. Onlarda ongoing ve onset ayrı etiket/aday kuralıyla aynı modelde (koşul bayrağıyla), pencere başına eşit ağırlık. Bizde hücre eşit.
5. Onların Şubat onset proxy'si (gözlenen-eligible) 0.8925, bizim 'obs' dev 0.91-0.92: yerelde fark yok; LB'de fark varsa yerelde görünmeyen bir şeyde (kapasite/eşik/girdi biçimi/dağılım) yatıyor, etiket tanımı tek başına açıklamayabilir.
6. Girdi farkı: onlarda T satırı yok, bizde maskeli T satırı. Val girdilerinde T satırı %46-69 dolu, eğitimde (hibrit) taklit ediliyor, ama bu onlara kıyasla ek bir dağılım kayması kaynağıdır (v4'ün gün-düzeyi özellikleri adv AUC 0.98 ongoing).
7. Hipotez testi için en ucuz probe: etiket+ağırlık değişikliği (Fikir 3) tek başına. Tek değişkenli ve onların tarifine en yakın nokta.

---

## 5. Onların bizden kötü yaptığı şeyler (gerilemeyelim)

- Task 1 taban: pure zamansal interpolasyon, akış tabanı 0.5/0.5; bizim zaman+uzay IDW + v:0.6 reconcile daha güçlü. Physics için hiçbir şey yok; bizim 0.712 proxy / S_LWR 0.574 onlarda kör.
- Final state modeli Şubat'sız (Haz-Oca); biz 9 ayın hepsiyle eğitiyoruz.
- Task 1 eğitim maskeleri kendi simülasyonları, bizimkiler resmi train maskeleri (test dağılımı birebir).
- Queue doğrulaması tek ay (Şubat), eşik ve early stop aynı ayda -> eşik gürültülü; bizim 3-ay CV dev seti ve tanılarımız (adversarial, LOMO, LB probu) daha sağlam. Kendileri "local gains need careful interpretation" diyor; LB'ye bakıp seçtikleri ayarlar var.
- ODME: L-BFGS-B warm start'ı kapalı asset'ten, başarı sertifikası yok, yeashusemwal'a göre ~0.0006 gevşek. Bizimki kesin, aynı yeniden üretilebilir (run_all byte-özdeş).
- T satırı yok (ongoing'de en güncel durum T-5'te). Bizim ongoing özellikleri T satırından fayda sağlıyor olmalı; bunu bozma.
- Süreç: GPU yeniden üretim bit-özdeş değil, üç ayrı script'e dağılmış; bizde tek giriş + manifest.
- Ayarlar LB'ye bakarak: ongoing 0.40 ("lower threshold lost LB"), blend ağırlıkları; körü körüne kopyalama (bizim kalibrasyonumuz farklı).

## Özet karar önerisi (tek cümle)
En düşük riskli kazanç: v4 üzerinde (1) ongoing eşik probu ve (2) E1 ongoing probu; ardından etiket/ağırlık (3). State tarafında yerel proxy ile CatBoost/üçüncü üye + bayraklar + daha uzun eğitim denenebilir (GPU için izin gerekli).
