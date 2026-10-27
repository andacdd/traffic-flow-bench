# Liderler ne yapıyor? Task 2 odaklı araştırma (2026-10-26)

Kapsam: Kaggle "2026 IEEE Big Data - Traffic Flow Bench" discussion forumu (19 başlığın tamamı), leaderboard, Code sekmesi (kaggle CLI listesi, 21 notebook), resmi repo `reference/trafficflowbench-public` (HEAD f9e3fbc, 2026-10-14; `git fetch` yeni commit getirmedi).
Notebook kaynakları `reference/public_notebooks/<ref>/` altına çekildi, hiçbiri çalıştırılmadı.

İşaretler:
- **[OLGU]**: kaynakta açıkça yazıyor. Tırnak içindekiler doğrudan alıntıdır (15 kelimenin altında).
- **[ÇIKARIM]**: benim yorumum ya da hesabım. Kaynakta yazmıyor.
- **[KENDİ KONTROL]**: bu araştırma sırasında train verisi üzerinde yaptığım küçük ölçüm.

Tarihler: forum zaman damgası (sayfadaki `title` özniteliği, UTC'ye çevrildi) okunabildiyse kesin tarih verildi. Okunamadıysa "~" ile göreli tarihten hesaplandı.

---

## 1. Özet

1. **[OLGU] 0.92'lik skorlar halka açık bir kaynakla açıklanmıyor.** Forumda ve Code sekmesinde 0.89'un üstünde skor bildiren hiçbir yazı ya da notebook yok. Herkese açık en iyi kaynaklar 0.885 (Paul Larmuseau, LB'de) ve 0.871 (yeashusemwal / honghanhhh). Liderlerin yöntemi bilinmiyor.
2. **[OLGU + ÇIKARIM] İlk 21'deki bazı skorlar veri yeniden kurulmadan önceki sürümden kalma.** Host, eski skorların LB'de kaldığını ve kaldıramadığını yazdı ([#740844](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740844), 2026-09-11). Kushal Khemani (7., 0.91004), Kanak Khemani (21., 0.89449) ve minghao mei (80.) son kez veri yeniden açılmadan önce submit etti (yeniden açılış: 2026-09-11 16:25 UTC). Pardheev Krishna (3., son submit 09-12) ve Jason Karpeles (6., son submit 09-19) belirsiz. **Ama 1., 2., 4. ve 5. sıradaki takımlar (0.9275 / 0.9251 / 0.9223 / 0.9198) son birkaç gün içinde submit etti.** 2. (tyz123456, ID 17027267) ve 5. (Ashkan Zare, ID 17034479) takımların ID'leri, 21 Eylül'de forumda yazan Dank Axon'un ID'sinden (16933247) büyük: bu ikisi rebuild'den sonra katıldı, skorları kesin olarak yeni veriye ait. Yani ~0.925 gerçek ve güncel veri üzerinde alınmış bir skor. Fark gerçek.
3. **[OLGU] Task 2 bilgi kuralı kesin.** Origin'i T olan pencere için girdi, model fit'i, model seçimi ve eşik seçimi yalnızca zaman damgası ≤ T olan veriyi kullanabilir. Buna aynı split'in önceki günleri, train'in tamamı ve April pencereleri için March'ın tamamı dahil. **T anındaki satırın kendisi de kullanılabilir.** Host'un ödül incelemesinde kodu "every row stamped after each window's origin removed" ile tekrar koşup tahminlerin değişmediğini kontrol edeceği yazılı ([#749601](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749601), [#748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789), [#745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700)). Bu yüzden T sonrası veriye dayanan bir hile ödül incelemesinde düşer. **[ÇIKARIM]** Liderlerin 0.92'si ödül için geçerli olmayabilir de. Bizim incident memory refit'imiz bu teste göre kurulmalı (bkz. 5.1).
4. **[OLGU] Window selector'a göre koşullamak kurallara uygun.** Host'un ifadesi: 360 dakika aralık ve maksimum persistence IoU (0.9) üzerinden koşullamak "within the rules". Tek şart: sadece T'ye kadar olan `window_index.csv` origin'leri kullanılabilir ([#745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700), ~2026-10-15). Onset'te kuyruğun yalnızca T+30'da çıkması resmi duyuruyla herkese açıklandı ([#742350](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350), 2026-09-21). Bu yüzden bizim "onset sadece T+30" avantajımız yok, bu artık herkesin tabanı. Host'a göre sıralamayı ayıran şey "which links queue".
5. **[OLGU] Sıralamayı kuyruğun kenarları belirliyor.** Paul Larmuseau (LB 0.88543) LB probe'ları ile ölçtü: kendi kuyruk tahminleri public LGBM/CatBoost kuyruğuyla pozitif hücrelerin %76-91'inde örtüşüyor, ama S_queue'da 0.16 geride kalıyorlar. Ayrıca kendi ürettikleri pencereler ongoing kazancını ~8 kat abartmış ([plarmuseau notebook](https://www.kaggle.com/code/plarmuseau/trafficflowbench-the-physics-of-a-synthetic-jam), 2026-10-21). E-Max AI'da da aynısı oldu: yerelde +0.0135 S_queue, LB'de -0.0028 ([#748784](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748784)). Bizim "dev yönü LB'de ters döndü" gözlemimizle tutarlı.
6. **[OLGU] Ayların darboğaz haritası değişiyor, March April'ı öngörmüyor.** Larmuseau'nun bulgusu: bazı darboğazlar March ve April'da tamamen kapanıyor (%24 → %0), yenileri açılıyor (%0 → %35). March'ın kaymaları ile April'ın kaymaları arasındaki korelasyon -0.07. Önerisi: train + puanlanan ayın kendi önceki günleri (nedensel). **[ÇIKARIM]** Bizim "olay hafızası / ay-içi refit" yönümüzü doğruluyor. April'a March'tan özellik taşımanın riskli olduğunu da gösteriyor.
7. **[OLGU] Kusursuz tahmin physics'te ~0.96 alıyor, biz ~0.72'deyiz. Bu kuyruktan sonraki en büyük boşluk.** Kaynak: kendi NOTES.md; repo README'deki organizatör perfect-answer satırı. **[ÇIKARIM]** Physics farkı 0.24 × 0.15 = 0.036 S_total ediyor. Queue farkı (1.0 − 0.80) × 0.30 = 0.06. State farkı (1 − 0.935) × 0.35 = 0.023. 0.0556'lık açık tek bir task'tan gelemez. Liderler büyük ihtimalle S_queue ≈ 0.88-0.92 ile birlikte daha iyi bir state/physics kombinasyonuna sahip.
8. **[OLGU] Larmuseau'nun ongoing için en iyi modeli:** ham uzay-zaman yaması (21 link × 9 adım, 378 özellik) üzerinde LightGBM. Feb'de IoU 0.864, 52 elle üretilmiş özellik 0.85. Doğrusal yöntemler (ARIMA, Kalman, CTM, SVD) persistence seviyesinde kaldı ([plarmuseau notebook](https://www.kaggle.com/code/plarmuseau/trafficflowbench-the-physics-of-a-synthetic-jam)).
9. **[KENDİ KONTROL] "Ongoing için de selector son adımı işaret ediyor" hipotezi reddedildi.** 40 train ongoing penceresinde kuyruk yaşı medyan 180+ dakika. Değişen hücre sayısı ufuk boyunca artıyor (ort. 9.3 → 16.8) ama değişim son adımda toplanmıyor. Persistence IoU ≤ 0.9 sınırı neredeyse hiç bağlayıcı değil (gözlem etiketiyle 40/40 pencere < 0.9). Onset tipi bir zaman hilesi ongoing'de yok.

---

## 2. Liderlik tablosu snapshot (public = March, 2026-10-26 ~15:00 UTC)

Kaynak: [leaderboard](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/leaderboard) ve `kaggle competitions leaderboard -s` (submissionDate = takımın SON submit zamanı, en iyi skorun alındığı zaman değil). LB sayfasındaki banner'a göre public skor yalnızca validation ayı (March). Final April'dan gelir ve "A gain on this score does not always carry over".

| # | Takım | Skor | Son submit (UTC) | Not |
|---|---|---|---|---|
| 1 | Max Pollard | 0.92750 | 2026-10-26 | güncel |
| 2 | tyz123456 | 0.92505 | 2026-10-26 | güncel; takım rebuild sonrası katıldı (ID > Dank Axon) |
| 3 | Pardheev Krishna | 0.92406 | 2026-09-12 | rebuild'den ~12 saat sonra; en iyi skor eski olabilir **[ÇIKARIM]** |
| 4 | Musa Peker | 0.92226 | 2026-10-24 | güncel |
| 5 | Ashkan Zare | 0.91982 | 2026-10-25 | güncel; takım rebuild sonrası katıldı |
| 6 | Jason Karpeles | 0.91665 | 2026-09-19 | belirsiz |
| 7 | Kushal Khemani | 0.91004 | 2026-09-10 | **rebuild öncesi** (eski veri) |
| 8 | Lukas | 0.90937 | 2026-10-26 | |
| 9 | Amin Mohamed | 0.90494 | 2026-10-20 | |
| 10 | Anton | 0.90303 | 2026-10-23 | |
| 21 | Kanak Khemani | 0.89449 | 2026-09-10 | **rebuild öncesi** |
| 33 | Dank Axon | 0.89128 | 2026-10-23 | onset T+30 bulgusunu forumda paylaşan takım |
| 52 | Paul Larmuseau | 0.88543 | 2026-10-26 | public notebook'un sahibi |
| 67 | Anubisyy | 0.88101 | 2026-10-26 | public notebook'u 0.86517'de; sonra gizli olarak 0.881'e çıkmış |
| ~105-115 | (biz) | 0.87192 | | 0.8737-0.8720 bandında |

Toplam ~633 takım. 50. sıra 0.8858, 100. sıra 0.8751.

---

## 3. Discussion bulguları (19 başlığın hepsi okundu)

| Başlık | Tarih | Yazar (rol) | İddia / karar | URL |
|---|---|---|---|---|
| Task 2 future cells are present in the released mainline_states_masked files | 2026-09-09 | Luigi ishii (katılımcı) | Ufuk hücrelerinin ~%63-69'u maskeli state'te gözlemlenmiş durumdaydı. Train'de bu yolla S_queue ~0.9 çıkıyor. Kendisi kullanmadı. Host doğruladı: val S_queue 0.641 vs persistence 0.252 | [740522](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740522) |
| Submissions paused: Task 2 labels were readable | 2026-09-09 | Organizer | Ufuk hücreleri boşaltıldı. Kushal Khemani (o tarihte 7.) sızıntıyı 8 Eylül'de e-posta ile bildirdiğini yazdı. Chonghao Peng `window_history.parquet` içinde ikinci bir sızıntı yolu buldu. Host submit'leri yeniden durdurdu | [740536](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740536) |
| Submissions are open again, on a rebuilt dataset | 2026-09-11 | Organizer | val ve private yeni seed'lerle yeniden üretildi, train aynı kaldı. Eski skorlar LB'de kalıyor ("we cannot remove them"). Ödül için yalnızca bugünden sonraki submit'ler sayılıyor. Musa Peker (4.): "most of the top entries have not been resubmitted". Host Kaggle'a ticket açtı. Sonucu forumda yok | [740844](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740844) |
| Task 2: which released data may a forecast use? | 2026-09-19 | Ankiy / Organizer | Task 2 online bir task: ≤ T her dosya serbest (aynı gün daha erken, önceki günler, train'in tamamı, `ramp_states` dahil). Ufuktaki rampa değerleri ve buffer sonrası ana hat değerleri yasak. Task 1 offline: split'teki her gözlem kullanılabilir | [742068](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068) |
| Task 2: in every queue_onset train window the queue first appears exactly at T+30 | 2026-09-21 | Dank Axon (şu an 33.) | 39/40 train onset penceresinde kuyruk ilk kez T+30'da çıkıyor. Aynı özelliklerle genel model IoU 0.57, sadece T+30 için eğitilen model 0.93 (train onset, tarih dışarıda bırakılarak). Host: "You can keep using it." | [742328](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742328) |
| Task 2 onset windows: the queue is always at the last step | 2026-09-21 | Organizer (duyuru) | 3 split'te 120/120 onset penceresinde kuyruk yalnızca T+30'da. Sebep: selector en erken uygun origin'i seçiyor. Kullanmak serbest. Host'a göre ayırt edici şey "which links queue" | [742350](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742350) |
| Is S_queue above 0.9 achievable using only data ≤ T? (onset ~0.7 tavanı) | 2026-09-23 (cevap 09-28) | Ankiy / Organizer | Host hedef açıklamıyor. Etiketler gürültüsüz alttaki state'ten hesaplanıyor, gürültü tabanı yok | [742761](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742761) |
| Clarification: Does the Task 2 timestamp cutoff apply to external data? | 2026-09-24 | Organizer | Zaman kuralı yarışma dosyaları için geçerli. Veri sentetik olduğu için dış kaynakta T sonrasına ait bilgi yok. Cevapları bulmaya yönelik kaynak yasak | [742910](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742910) |
| [0.809 LB] Open-Sourcing our Physics-Consistent 4-Task Pipeline | 2026-09-24 | Sōsuke Aizen (katılımcı) | Yumuşak FD projeksiyonu (α=0.15) ve 5 geçişli korunum. ODME λ 20 → 0.05 ile S_link 0.987 → 0.9999. Onset'te hız tabanlı bastırma IoU'yu 0.74'ten 0.17'ye düşürdü | [742980](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742980) |
| How to reconstruct the topology of corridors | ~2026-09-25 | Organizer cevabı | `from_node`/`to_node` kullanılmıyor. Boyuna sıra için milepost, korunum için `incoming/outgoing_link_ids`, rampalar için `ramp_attachment_map` kullanılıyor | [743819](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/743819) |
| Important update: award eligibility, final evaluation | ~2026-09-26 | Organizer | Kod 6 Kasım'a (submit), rapor 10 Kasım'a kadar. Tekrar üretilen skor private skorun %1'i içinde kalmalı. Ödül puanı %85 private LB + %15 rapor | [744187](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/744187) |
| Accidental submission from secondary account | ~2026-09-25 | katılımcı | (Task 2 ile ilgisiz) | [743549](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/743549) |
| About Student / Newcomer Award | ~2026-09-20 | katılımcı / Organizer | (Task 2 ile ilgisiz) | [741898](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/741898) |
| Clarification on Task 2 forecast origin T | soru ~2026-10-03, cevaplar ~2026-10-15 | Alejandro Rendon B. / Organizer | **T'deki satır kullanılabilir** (geçmiş T-60..T-5'i kapsıyor, artı T satırı). Resmi persistence baseline T-5 kullanıyor, yani gerekenden muhafazakâr. İkinci cevap: selector koşullaması (360 dk, maks persistence IoU) serbest. Ancak sadece ≤ T `window_index` origin'leri kullanılabilir | [745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700) |
| How to validate queue changes when the horizon is blanked | ~2026-10-13, cevap ~10-15 | E-Max AI (112.) / Organizer | 9 train ayından ~1300 ongoing pencere üretilmiş. 0.40 eşiği zaten argmax, eğri düz. Ortalama p train'de 0.169, val/private'ta 0.116. Yerelde +0.0135, LB'de -0.0028. Host: 80 resmi train penceresi Haziran 2030'da çünkü "selector keeps the earliest qualifying origins". Resmi etiket alttaki state'ten hesaplanıyor | [748784](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748784) |
| Task 2 is online: does "≤ T" include the validation split for private windows? | ~2026-10-13, cevap ~10-15 | huyml911 / Organizer | Evet. T'den önceki veriyle refit serbest (aynı split'in önceki günleri dahil). April pencereleri March'ı fit/seçim için kullanabilir, tersi yasak. Tüm split'e fit edilip aynı split'in erken pencerelerine uygulanan model yasak | [748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789) |
| Task 2 - Protected Window Histories vs. Calendar Masked-State Coverage | ~2026-10-14, cevap ~10-15 | Rahul Bharadwaj / Organizer | Maskeli takvim geçmişinde kapsama ~%51, çünkü 60 dk'lık pencere geçmişleri Task 1 maskesinden muaf. Girdi dağılımı için referans `window_history` olmalı. Train'in maskesiz state'iyle istenen kapsamaya göre kendi maskeni uygulamak serbest | [749278](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749278) |
| Task 2 clarification: As-of-origin requirements for model selection | ~2026-10-15, cevap ~10-18 | Rahul Bharadwaj / Organizer | Eksik veri, kapsama ve pencere metaverisine bakmak serbest. Kısıt trafik gözlemlerine uygulanıyor. Ödül incelemesinde Task 2 kodu T sonrası satırlar silinerek tekrar koşulacak, tahminler değişmemeli. Log gerekmiyor | [749601](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749601) |
| Reminder: register by October 25 | ~2026-10-15 | Organizer | 460'tan fazla takım var. Kayıt son tarihi 25 Ekim'di (geçti) | [749480](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749480) |

Sonuç: hiçbir üst sıra takımı (ilk 30) yöntemini forumda paylaşmamış. Paylaşan en yüksek sıralı katılımcılar Musa Peker (4.; yöntem değil LB önerisi) ve Dank Axon (33.; onset T+30).

---

## 4. Notebook'lar (2026-10-19 sonrası yeni olanlar ve önceden özetlenmemiş eski olanlar)

Liste: `kaggle kernels list --competition ... --sort-by dateRun` (21 notebook). Yeni çekilenler `reference/public_notebooks/<kullanıcı>_<slug>/` altında.

| Ref (son çalışma) | Rapor edilen LB | Yeni olan ne | URL |
|---|---|---|---|
| plarmuseau/trafficflowbench-the-physics-of-a-synthetic-jam (2026-10-21) | Notebook'ta 0.8423'e kadar ayrışım; takımın LB'si şu an 0.88543 | **En değerli yeni kaynak.** (a) LB probe ayrışımı: place × time-of-day onset hafızası (T+30'da top-4) ile +0.067. Public queue'ya geçince +0.048. Kuyruk kenarları belirleyici. (b) Onset öncülleri (aşağı link yoğunluğu AUC 0.82) seçilmiş pencerelerin içinde "useless". Zaman bilgisi zaten selector'dan geliyor. (c) Ongoing: ham 21×9 yaması + LGBM 0.864. (d) Aylar arası darboğaz değişimi; March April'ı öngörmüyor. (e) Kuyruk uzunluğu ≈ (N − N₀)/(k_jam − k_free), March/April'da R² 0.94-0.95. (f) Dağılma (dissolution) hiçbir sinyalle haber verilmiyor (AUC ≤ 0.58) | [link](https://www.kaggle.com/code/plarmuseau/trafficflowbench-the-physics-of-a-synthetic-jam) |
| yangkuangou/traffic-task-2-forecast-time-availability-audit (2026-10-26) | yok | Önceki sürüme göre sadece bakım (aşama kümesi kontrolü). İçerik değişmemiş | [link](https://www.kaggle.com/code/yangkuangou/traffic-task-2-forecast-time-availability-audit) |
| robinamirbahar/trafficflowbench-physics-guided-state-odme (2026-10-26) | Notebook'ta yok; takımın LB'si 0.87282 | Daha önce biliniyordu (FD projeksiyonu). Kendi physics S1'imiz buradan geldi | [link](https://www.kaggle.com/code/robinamirbahar/trafficflowbench-physics-guided-state-odme) |
| honghanhhh/trafficflowbench-lb-0-87141 (2026-10-25) | 0.87141 | Daha önce biliniyordu | [link](https://www.kaggle.com/code/honghanhhh/trafficflowbench-lb-0-87141) |
| paragpu/trafficflow-v1 (2026-10-19) | yok | Sadece EDA, değeri yok | [link](https://www.kaggle.com/code/paragpu/trafficflow-v1) |
| rushikeshpundkar/notebooke511edcdea (2026-10-19) | yok | Hız-yoğunluk monotonluk denetimi ve son işlem. Task 2 ile ilgisiz | [link](https://www.kaggle.com/code/rushikeshpundkar/notebooke511edcdea) |
| yunimiaomiaobei/ieeee (2026-10-17) | yok | Basit pipeline. Onset kuralı: hız ≤ 1.15 × v_cut ise kuyruk. Bizim seviyemizin altında | [link](https://www.kaggle.com/code/yunimiaomiaobei/ieeee) |
| shuangranse/trafficflowbench-06970-solution-writeup (2026-09-27) | 0.7269 | Onset'te flow/capacity ≥ ~1.0 kapısı: kuyruk yokken kapasite üstü akış ufkun boş kalacağını gösteriyor (train ayrımı 0.994 vs 1.003). Ongoing'de iki kuyruklu sensörlü linkin arasındaki sensörsüz linkleri doldurma | [link](https://www.kaggle.com/code/shuangranse/trafficflowbench-06970-solution-writeup) |
| lamhuy8904/traffic-flow-bench-pipeline (2026-09-08) | yok | "S_physics ≈ 0.975" iddiası var. **[ÇIKARIM]** Rebuild ve physics anchoring öncesi. SCORING_SPEC'e göre o dönemde sınır akışlarını projekte ederek S_LWR 1.0 almak mümkündü. Bugün geçersiz | [link](https://www.kaggle.com/code/lamhuy8904/traffic-flow-bench-pipeline) |
| sachith7, yanamiiannaa, avikdas567 (09-24 .. 10-05) | yok | Task 1 tanılama (temporal vs spatial dolgu, yönlü özellikler + CNN harmanı Feb S_state 0.930 → 0.937), fizik temalı pipeline. Task 2 için yeni bir şey yok | [sachith7](https://www.kaggle.com/code/sachith7/trafficflowbench-when-does-temporal-repair-fail), [yanamii](https://www.kaggle.com/code/yanamiiannaa/trafficflow-directional-features-safe-profiles), [avikdas](https://www.kaggle.com/code/avikdas567/hydrodynamic-traffic-reconstruction-queue-odme) |

Mevcut özetle (`public_notebooks_summary.md`) karşılaştırma:
- **Doğrulanan:** onset sadece T+30 (host + Dank Axon + Larmuseau). Ongoing eşiği 0.40-0.45 düz bir tepe (E-Max AI). Yerel doğrulamanın LB'yi yanlış sıralaması (Larmuseau, E-Max AI).
- **Çelişen:** shuangranse sensörsüz ara linkleri doldurmanın "worth a lot" olduğunu söylüyor. yeashusemwal ise 1-link boşluk doldurmada -0.0042 bulmuştu. **[ÇIKARIM]** Bu iki şey farklı: biri sensörsüz linkler, diğeri gözlemlenmemiş boşluklar.
- **Bilgi:** Anubisyy takımı LB'de 0.88101'e çıktı. Bu public notebook'unun (0.86517) 0.016 üstü. Yeni sürümü yayınlanmadı.
- **Kayıt:** Larmuseau, anubisyy state'i için 0.9416 ve 50/50 harman için 0.9349 rakamlarını veriyor. Bunların LB'den mi yerelden mi türetildiği açık değil.

---

## 5. Bize uygulanabilir fikirler (beklenen S_total kazancı ve riske göre sıralı)

Kazanç tahminleri **[ÇIKARIM]**. Kaynak satırı fikrin nereden geldiğini gösteriyor.

### 5.1 Uyum güvencesi: refit/hafıza "T sonrası satırlar silinince değişmez" testini geçmeli
- Kaynak: [#749601](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/749601), [#748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789).
- Kazanç: skora doğrudan etkisi yok. Ama ödül uygunluğu için koşul: tekrar üretim private'ın %1'i içinde olmalı ([#744187](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/744187)).
- Risk (yüksek önem): herhangi bir queue özelliği T sonrası veriye bakıyorsa tahmin değişir ve takım elenir. Örnekler: offline Task 1 rekonstrüksiyonu (gelecek gözlemleri kullanır), tüm ay üzerinden hesaplanmış profil, haftalık refit'te T'nin gününden sonraki günler.
- Öneri: bir "T-kesme tekrar koşusu" testi yazılıp tüm private pencerelerde tahminlerin birebir aynı kaldığı doğrulanmalı.

### 5.2 Ongoing: ham uzay-zaman yaması modeli, mevcut harmana üçüncü üye olarak
- Kaynak: Larmuseau'nun tablosu (Feb'de 52 özellik 0.85, ham 21×9 yaması 0.864).
- Kazanç: ongoing +0.01-0.015 IoU → S_total +0.0015-0.002.
- Risk: orta. Larmuseau'nun kendi uyarısı ve bizim kendi deneyimimiz, üretilmiş pencerelerdeki kazancın LB'ye ~1/8 oranında taşındığını gösteriyor. Bu yüzden ağırlıklı dev ile ölçülmeli, tek değişkenli bir LB probe'u ile doğrulanmalı.

### 5.3 Onset konumu: place × time-of-day hafızası ve ay-içi darboğaz değişimi
- Kaynak: Larmuseau (top-4 hafıza onset'te LB'de +0.067; darboğazlar ay içinde açılıp kapanıyor, kapatma kuralı ≥10 gün ve oran < 0.05). shuangranse (flow/cap ≥ 1.0 kapısı).
- Kazanç: bizim en büyük kayıp kalemimiz onset-nadir (0.029). Ay-içi refit (q9) zaten bu yönde. Ek olarak "kaybolan darboğazı kapatma" kuralı ve flow/cap kapısı denenebilir. Beklenen +0.002-0.005 (private'ta daha fazla).
- Risk: orta. public LB'de val onset'leri 1-7 Mart'ta, yani ölçülemez. Train'de geri test yapılmalı.

### 5.4 Physics boşluğu (~0.036 S_total potansiyel)
- Kaynak: NOTES'taki organizatör perfect answer (physics ~0.96). SCORING_SPEC (S_LWR yalnızca Task 1 değerlerinden, anchoring var). Kendi S1 operatörümüz yerelde +0.031 physics verdi, LB'ye ~1/4'ü yansıdı.
- Kazanç: belirsiz. Liderlerin ~0.925'i büyük ihtimalle physics'te de bizden ileride. **[ÇIKARIM]** Queue ~0.90 + physics ~0.72 + state 0.94 ≈ 0.905 eder, 0.925'e yetmez.
- Risk: düşük-orta. State proxy'leri LB'ye iyi taşınıyor (NOTES: S4 beklentiyle birebir tuttu).

### 5.5 State: anubisyy-yalnız state'i tek değişkenli LB probe ile kontrol etmek
- Kaynak: Larmuseau'nun 0.9416 (anubisyy) vs 0.9349 (50/50) rakamları.
- Kazanç: ancak bizim ~0.935 tahminimiz yanlışsa var. O durumda +0.002-0.0025 S_total.
- Risk: düşük, tek submit.

### 5.6 Selector koşullaması (360 dk aralık, persistence IoU ≤ 0.9)
- Kaynak: [#745700](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/745700), config `maximum_persistence_iou: 0.9`.
- **[KENDİ KONTROL]** Train ongoing'de persistence IoU sınırı bağlayıcı değil, son adım yoğunlaşması yok. yeashusemwal'ın "sadece selector'ın seçebileceği pencereler" üyesi +0.0003 vermişti.
- Kazanç: ≈ 0. Önceliksiz.

### 5.7 Kullanılmaması gerekenler
- Ufuktaki rampa değerleri ve buffer sonrası ana hat değerleri ([#742068](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/742068)).
- March pencereleri için April verisi ([#748789](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/748789)).
- `window_history` ikinci sızıntı yolu: rebuild ile kapatıldı ([#740536](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740536), [#740844](https://www.kaggle.com/competitions/2026-ieee-big-data-traffic-flow-bench/discussion/740844)).

---

## 6. Erişilemeyenler ve sınırlar

- **Üst takımların yöntemi:** ilk 30'dan hiçbiri yöntemini forumda ya da Code sekmesinde paylaşmamış. 0.92'nin nereden geldiği doğrudan bir kaynakla açıklanamıyor. Bu raporun 1.7 ve 5.4 maddeleri çıkarım.
- **Leaderboard'da en iyi skorun tarihi:** CLI ve sayfa sadece son submit tarihini veriyor. Pardheev Krishna ve Jason Karpeles'in skorlarının rebuild öncesi mi sonrası mı olduğu belirlenemedi. Host'un Kaggle ticket'ının sonucu forumda yok.
- **Forum tarihleri:** bazı başlıklarda yalnızca göreli tarih ("11d ago") okunabildi. Bunlar "~" ile işaretlendi.
- **Code sekmesi:** giriş yapılmadan liste sayfası render olmadı. Liste kaggle CLI ile alındı (21 notebook). Notebook çıktıları ve veri setleri indirilmedi.
- **Kendi kontrolüm (madde 1.9):** etiketleri gözlenen hızdan türettim (`speed ≤ 0.6 × free_speed`, bütün linkler). Resmi etiket alttaki state'ten ve uygun hücrelerden hesaplanıyor. Persistence IoU değerleri bu yüzden gürültüyle aşağı çekilmiş olabilir. Niteliksel sonucu (son adım yoğunlaşması yok, kuyruklar eski) değiştirmesi beklenmez. Script: `/private/tmp/claude-501/-Users-andac-Claude-Personal-Projects-ieee-traffic-flow-bench/92e3eab7-1b76-4705-901f-2fd070006512/scratchpad/tools/ongoing_steps.py` (geçici).
