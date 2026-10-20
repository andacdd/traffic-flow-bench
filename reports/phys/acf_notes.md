# Faz 4a — Physics kapanış testi: kalıntı otokorelasyonu (src/phys_acf.py, 2026-10-19)

Veri: 10 panel train, eligible hücreler, 1583 link, 9 ay. r1 = gözlem - merkezli med5 (ve mean3) düzleştirme; r2 = gözlem - v:0.6 interpolasyon (sadece Task 1 hedef hücreleri). Null'lar: (a) link içi karıştırma, (b) aynı maske + aynı filtreden geçmiş iid beyaz gürültü. Merkezli filtre beyaz gürültüye de yapay ACF verir (med5: lag1 -0.225, lag2 -0.290; mean3: lag1 -0.667), bu yüzden doğru karşılaştırma (b).

## Bulgular
- Hız: med5 r1 ACF beyaz gürültü null'ıyla neredeyse aynı (lag1 -0.220 vs -0.225, lag2 -0.286 vs -0.290; panel bazında fark <= 0.026). Hız gürültüsü zamanda bağımsız.
- N (= k*L): lag1 -0.191 vs -0.225 (fark +0.034; serbest +0.029, kuyruklu +0.073), lag3 +0.051 vs +0.103. Fark pozitif ve filtrenin sızdırdığı yavaş sinyalle uyumlu; renkli gürültü deseni (ör. AR(1) pozitif yapı) yok.
- Hız-akış gürültüsü aynı hücrede ilişkisiz: corr(r_v, r_q) medyan 0.001 (serbest), 0.067 (kuyruklu, IQR 0.005-0.33). k ile ilişki sadece q/v tanımından gelen mekanik ilişki.
- AR(6) açıklanan varyans (havuzlanmış, med5): N 0.090, q 0.107, v 0.113; aynı filtreden geçmiş beyaz gürültüde 0.182. Gerçek kalıntı beyaz gürültüden DAHA az tahmin edilebilir -> AR'nin yakaladığı her şey filtre artefaktı (merkezli filtre geleceği sızdırıyor), gerçek gürültü öngörüsü değil.
- S_LWR vekili: AR ile mean|ΔN kalıntı| oranı 0.912 (hedef hücreler 0.912); beyaz gürültü null'ı 0.819. Gerçek veride kazanç null'dan küçük -> öngörülebilir pay ~0.
- r2 (interpolasyon hatası) pozitif ve yavaş sönen ACF (N lag1 0.33, lag3 0.13; karıştırma null |acf| q95 0.024). Bu ölçüm gürültüsü değil yumuşak sinyal hatası; model A bunu zaten kâhin seviyesine (0.695 vs 0.699) kadar aldı.

## Karar
Physics kapatılsın. Kalan açık zamanda bağımsız, hız-akış arasında ilişkisiz hücre başı ölçüm gürültüsü; beyaz gürültü null'ına göre öngörülebilir pay ölçülemeyecek kadar küçük (AR R2 gerçek 0.09 < null 0.18). 0.70 üstü için yeni modelleme yapılmasın; physics işi Faz 4 üretim eğitimine (A, 9 ay) indirgensin.

Çıktılar: acf_by_corridor.csv, acf_summary.csv, acf_xcorr0.csv, acf_ar.csv, acf_delta.csv, acf_plot.png, acf_run.log. Süre 535 s, tepe bellek 2.71 GB.
