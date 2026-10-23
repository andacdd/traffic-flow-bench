# Glossary

**Pencere (window)**: Task 2'de bir koridor-gün için T anındaki tahmin noktası; ufuk T+5..T+30 dk. Onset ve ongoing olmak üzere iki koşulu var.
_Avoid_: örnek, sample

**T**: Pencerenin başlangıç anı. T'den sonraki hiçbir gözlem o pencerenin tahmininde kullanılamaz.

**Onset penceresi**: T'de kuyruk yok, ufukta kuyruk başlıyor.

**Ongoing penceresi**: T'de kuyruk zaten var, ufukta sürüyor ya da değişiyor.

**Değerlendirme ayı**: Skorlanan split'in ayı. Validation için Mart, private için Nisan. Train bunların dışında kalan aylardır.

**Nadir konum**: Train'de kuyruk sıklığı 0.02'nin altında olan hücre (link × zaman dilimi).

**Nadir pencere**: Ufuktaki kuyruk hücrelerinin yarıdan fazlası nadir konumda olan pencere. Olay kaynaklı kuyruğu temsil eder.
_Avoid_: incident window

**Olay oranı**: Bir aydaki kuyruk hücrelerinin nadir konuma düşen payı. Train %0.8-2.9, validation %5.2, private %8.5.

**Ağırlıklı dev**: Dev pencere skorunun, nadir pencerelere ρ ağırlığı verilerek hesaplanan ortalaması (validation için ρ≈0.28, private için ρ≈0.4). LB'yi tahmin etmek için kullanılan yerel ölçüt.

**Online refit**: Bir pencerenin modelinin, train'e ek olarak aynı değerlendirme ayının T'den önceki günleriyle yeniden eğitilmesi.

**Olay tekrarı**: Bir linkte nadir kuyruk görülen günden sonraki günlerde aynı linkte yine nadir kuyruk görülmesi. Değerlendirme aylarında var (validation lag-1 lift 3.9, private 2.8), train'de yok (1.25).

**Olay hafızası**: Bir hücrenin linkinde, T'nin gününden önceki 1/3/7 günde nadir kuyruk olup olmadığını özetleyen bilgi.
