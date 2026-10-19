# Public notebook özeti: Task 2 (queue) ve Task 4 (ODME)

Kaynak: `reference/public_notebooks/<slug>/` altındaki 4 notebook (yalnızca kaynak; çalıştırılmadı, veri indirilmedi). Fikir çıkarma amaçlı, kod kopyalanmadı.

## Notebook 1: yeashusemwal / trafficflowbench-trees-u-net-cnn-lb-0-87
Rapor edilen LB: **0.87141** (final birleşik; tablo 0.85556 -> 0.87141 aşamalarını gösteriyor). Final Task 2 = anubisyy'nin onset modeli + ongoing'de kendi ensemble'ı ile anubisyy LightGBM'in 50/50 olasılık ortalaması.

1. **Onset sadece T+30:** onset pencerelerinde yalnız h=6 tahmin ediliyor; tek başına +0.027 (bu bizim yaptığımızla aynı, tek en büyük kazanç). Onset'te eşik P>0.50, hiçbir link geçmezse en olası 2 link (top-2 fallback). Eşik 0.52 + top-2 geri tepti (-0.0070), top-3 fallback -0.0003.
2. **Ongoing ensemble + eşik:** 7 LightGBM üyesi (iki tanesi 3 saatlik ek bağlam: kuyruk yaşı, kuyruk önü hızı, rampa akışları), eşik P>0.40; anubisyy ile olasılık ortalaması alınıp eşik 0.45 uygulanınca +0.0023 (ortalama) ve +0.0011 (0.40 -> 0.45). Gerekçe: skorlanan aylardaki kuyruklar eğitimdekinden küçük, 0.40 fazla hücre işaretliyor.
3. **Leakage uyarısı:** hedef günü de içeren (tüm günlerin ortalaması) profiller leak ediyor; leave-one-day-out profillerle yerel skor düştü ama LB yükseldi (Task 2 +0.008, Task 1 +0.004). Yerel doğrulama modeli yanlış sıraladı (CNN ongoing'de yerelde iyi, LB'de -0.0049).
4. **Selector'a benzer eğitim penceresi:** resmi ongoing pencerelerin persistence IoU'su hiç 0.9 üstü değil (0/40, rastgele anlarda %8.7); sadece selector'ın seçebileceği pencerelerle eğitilen üye (v20, ensemble'da %20 pay) +0.0003. Bagging/leak-free üyeler +0.008, ensemble en iyi tek üyeden iyi (+0.0005). Boş kalan 1-link boşluğu doldurma -0.0042 (boşluklar çoğunlukla gerçek), maske temizliği/expected-IoU decoding yerelde kazanç vermedi, hücre-iletim simülasyonu persistence'ın altında kaldı.
5. **ODME:** min ||Af-c||^2 + lambda||f-b||^2, f>=0, lambda=0.05, **her split'in kendi sayım ve prior'ı** ile (+0.031, kendi en büyük tek kazançları). Çözüm yakınsamaya kadar: FISTA warm start, sonra sıkı toleranslı L-BFGS-B (+0.0006). Ölçekleme/ağırlık yok; doğrulama şeması yok (S_od/S_attr yerelde ölçülemiyor).

## Notebook 2: anubisyy / catboost-lgbm-nnls-pipeline-top-50-0-86517
Rapor edilen LB: **0.86517** (Task 1 CatBoost depth 8+10 residual, 94 özellik; ODME çözüm sıkılaştırma 0.84833 -> 0.86025; state özellikleri 0.86446; onset blend 0.86517).

1. **Eğitim penceresi seçimi:** aday origin = geçmiş 60 dk'nın >=%70'i sonlu, gelecek 6 adımda bir yerde kuyruk. Onset: önceki saatte ve T'de kuyruk yok; ardışık aday origin'lerin yalnızca ilki (episode başlangıcı). Ongoing: T'de kuyruk var. Satır ağırlığı 1000/uygun_hücre_sayısı; kayıp sadece skorlanabilir (is_score_eligible) hücrelerde. Kuyruk etiketi: speed <= cut (free-flow'un %60'ı).
2. **Ongoing modeli (basit, tek LightGBM):** 700 ağaç, 36 özellik, horizon adımı özellik olarak (link x adım satırı); özellikler: son gözlenen hız/akış ve yaşı, son 3 vs önceki 3 adım ortalama farkı, min/mean/max, kuyruk geçmişi sayısı, son kuyruk bayrağı, gelen/giden komşu link ortalamaları, tod sin/cos, dow, geçerlilik oranı. Eşik 0.40 (February IoU ile seçildi). "Daha sonraki ongoing modeller ve düşük eşik yerelde iyi, LB'de kötü" notu var.
3. **Onset modeli:** 83 özellik (36 + lag 1,2,3,6,9,12 hız/akış, son 3/6 min hız, komşu ortalamaları, milepost sırasına göre +-1,+-2 komşu hız/akış/kuyruk). Yalnız T+30, LightGBM (1900 ağaç, 2 aşamalı) + CatBoost depth 7 (eğitim kopyasında %5 hücre dropout), 50/50 olasılık, eşik 0.55, boşsa argmax tek link. February'de tam 6 adım IoU ile seçildi: proxy 0.8820 -> 0.8925, LB +0.0007.
4. **Doğrulama:** eğitim Haziran-Ocak, doğrulama Şubat (zaman bölmeli); proxy IoU persistence ile kıyaslanıyor. Yazar, gözlenen-hız etiketi ile gizli durum arasındaki uyumsuzluk yüzünden küçük yerel kazançların güvenilmez olduğunu belirtiyor.
5. **ODME:** aynı objektif (1/2||Af-c||^2 + lambda/2||f-b||^2, lambda=0.05, f>=0). A yalnızca ölçülen linkleri içerir (eksik sayım 0 sayılmaz). Link-uzayında kapalı form ridge başlangıcı f=b+A^T(AA^T+lambda I)^-1(c-Ab), negatifler 0'a kırpılıp L-BFGS-B'ye veriliyor; analitik gradyan; **maxiter=8000, ftol=1e-14, gtol=1e-6, maxcor=30**: sadece bu LB'yi 0.84833 -> 0.86025 yaptı (600 iterasyon yakınsamamıştı). Prior'a doğrudan çıpa; ağırlık/ölçekleme yok.

## Ek notebook'lar (kısa)
- **soukeaizenz (LB 0.80939):** Task 2 saf persistence (ongoing) + onset'te yalnız T+30, korridor başına ampirik top-2 darboğaz; BFS ile yukarı akım genişletme IoU'yu çökertti. ODME lambda=0.05, prior'a doğrudan çıpa (v1'de prior'u 0.95*wp+0.05*base ile sulandırmak ve 0 akışları değiştirmek S_od/S_attr'ı bozdu), scipy nnls.
- **sujanmajhisuzan ("0.90 target", LB raporlamıyor):** anubisyy checkpoint'lerinin türevi; ek: ongoing'de horizon bazlı eşik vektörü ve 1B morfolojik kapama. Kapama yeashusemwal'ın negatif bulgusuyla (-0.0042) çelişiyor, kanıtsız; ODME ayarları (maxiter 10000, ftol 1e-15, gtol 1e-7) aynı formülasyon.

## Task 1/3 notu (en fazla iki madde)
- Task 1: CatBoost depth 8+10 residual + +-5..60 dk lead/lag (hız, şerit-başı akış, oran, eligible bayrağı) özellikleri; ayrıca LightGBM + U-Net (yoğunluk terimli kayıp) 70/30 karışımı ve iki bağımsız Task 1 modelinin 50/50 ortalaması (+0.0023). Leave-one-day-out profil (+0.004).
- Task 3: hız >=5 km/h kırpma ve U-Net kaybına yoğunluk (akış/hız) terimi eklemek; Task 3 yalnız tahmin hücrelerindeki yoğunluğa duyarlı.

## Bize uygulanabilir fikirler (beklenen değere göre)
1. **ODME çözümünü yakınsamaya kadar sür (YÜKSEK EV, DÜŞÜK RİSK):** iki notebook da +0.0006 ile +0.012 arası kazanç gösteriyor (anubisyy 0.84833 -> 0.86025). Bizim NNLS solver'ın gerçekten yakınsadığını gradyan/objektif ile doğrula; değilse sıkı L-BFGS-B. Doğrulama yerel S_link ile mümkün; overfit riski yok (konveks problem).
2. **Ongoing'de ensemble + ikinci model olasılık ortalaması, eşik 0.45 civarı (ORTA-YÜKSEK EV, ORTA RİSK):** +0.0023 ve +0.0011. Risk: eşik kazancı LB'ye bakılarak ayarlanmış; yerel IoU eşiği 0.40 gösteriyor. Eşiği tek başına LB'ye göre oynama; hedef kuyruk hacminin eğitimden küçük olduğunu varsay ve pozitif oranla kalibre et.
3. **Leak-free (leave-one-day-out) profil/özellikler (ORTA-YÜKSEK EV, DÜŞÜK RİSK):** hedef günü içeren agregasyon varsa kaldır; yerel skor düşebilir, LB yükselir.
4. **Onset: komşu +-2 milepost sırası özellikleri, kısa lag'ler (1,2,3,6,9,12), 2 aşamalı LightGBM + CatBoost 50/50, %5 hücre dropout kopyası (ORTA EV, ORTA RİSK):** onset model kalitesi (hangi darboğaz ilk kırılır) en büyük açık problem; bizim LightGBM tek model. Küçük February proxy farkı gürültülü olabilir.
5. **Onset eğitim penceresi: yalnız episode başlangıç origin'i, T'de ve önceki saatte kuyruk yok; ongoing'de selector-benzeri (persistence IoU <=0.9) pencereler (ORTA EV, DÜŞÜK-ORTA RİSK):** bu zaten bizim selector bilgimizle örtüşüyor; kazançlar küçük (+0.0003).
6. **Ensemble çeşitliliği: bagging/farklı bağlam uzunlukları (ORTA-DÜŞÜK EV, DÜŞÜK RİSK):** +0.008 (leak-free üyelerle).
7. **Onset eşik/fallback:** top-2 fallback ve eşik 0.50-0.55; top-k'yı büyütme zarar veriyor (-0.007, -0.0003). Mevcut ayarı koru, yeni ayara LB ile yakınsama.
8. **Yapma:** 1-link boşluk doldurma/morfolojik kapama (-0.0042), BFS ile yukarı akım genişletme, hücre-iletim simülasyonu, CNN'i ongoing'de kullanma (-0.0049), U-Net payını %50'ye çıkarma.

Genel overfit uyarısı: iki yazar da "yerel doğrulama LB'yi yanlış sıraladı" diyor; ongoing eşik, blend ağırlığı ve fallback k değerleri LB'ye göre seçilmiş. Bunları körü körüne kopyalama; kendi doğrulamamızda (selector-benzeri pencereler, zaman bölmeli) test et.
