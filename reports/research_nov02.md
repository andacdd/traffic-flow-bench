# Kod + discussion taraması (2 Kas 2026)

Discussion: 22 Ekim'den sonra yeni teknik başlık yok (son: ödül kaydı hatırlatması). Önceki bulgular reports/queue/research_leaders.md.

Code (27 Eki sonrası 7 notebook, kaynaklar reference/kernels_nov/, git dışı):

| Notebook | LB | Bize yeni olan |
|---|---|---|
| autokagon/public-fast-gpu-inference (Robina v3 yeniden üretimi) | 0.87212 | physics: yoğunluk zamansal düzleştirme k=0.94k+0.03(k-1+k+1) (komşular gözlenmiş ve eligible ise), q=k*v; zarf v<0.58vf ve q>q_max ise q=0.96q+0.04q_max. State 1.04 yeash - 0.04 anubis. Ongoing: v_T ile promote/clear kuralı (0.39/0.48; 0.48/0.22/0.38, cut+12) |
| cacinie/tfb-v5orig | 0.86535 | anubis + ongoing eşik 0.50 (0.40: 0.86520, 0.60: -0.0028). Döküm iddiası state 0.941, queue 0.887, physics 0.66, ODME 0.854 (bizim ODME ~0.994 tahminiyle çelişir) |
| plarmuseau/eleven-anomalies | - | on-ramp akışı jam anında ~0 (yasal geçmişle ongoing +0.008); vf gizli sabit, 120.7 tavan; low-rank tensör state LB +0.003 |
| faresabbasai2022/2026-74 | ~0.74 | link × dow × slot kuyruk frekans profili (LOO) |
| yangkuangou/availability-audit | - | T zaman damgalı gözlem kullanılabilir (host 745700) |
| cacinie/tfb-report, prvsiyan | 0.72 | yeni bir şey yok |

Sonuç: public en iyi 0.87212 < bizim 0.87346. Liderlerin (0.931) yöntemi public değil. Uygulanabilir yeni fikirler: physics zamansal düzleştirme (LB A/B), ongoing eşik kayması, low-rank tensör state üyesi.
