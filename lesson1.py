# ====================================================
# KIRMIZI NESNE TAKİP PROGRAMI
#
# Bu program ne yapar?
#   1. Kameradan sürekli görüntü alır (video = art arda gelen resimler)
#   2. Her resimdeki kırmızı renkli bölgeyi bulur
#   3. O bölgenin etrafına yeşil bir kutu çizer
#
# Çıkmak için: klavyeden 'q' tuşuna bas
# ====================================================

import cv2       # Görüntü işleme için kullanılan ana kütüphane
import numpy as np  # Sayı dizileriyle (array) çalışmak için

# ----------------------------------------------------
# 1) KAMERAYI AÇ
# ----------------------------------------------------
# VideoCapture(0) -> bilgisayarın kendi (dahili) kamerasını açar.
# Eğer harici bir kamera kullanıyorsan 0 yerine 1 yazman gerekebilir.
kamera = cv2.VideoCapture(0)

# ----------------------------------------------------
# KIRMIZI RENGİN SINIRLARINI TANIMLA (HSV formatında)
# ----------------------------------------------------
# HSV üç değerden oluşur:
#   H (Hue)        -> Rengin türü (kırmızı, mavi, yeşil...)
#   S (Saturation) -> Rengin canlılığı (0 = soluk/gri, 255 = çok canlı)
#   V (Value)      -> Parlaklık (0 = karanlık, 255 = çok parlak)
#
# Kırmızı renk, renk çemberinde HEM EN BAŞTA HEM DE EN SONDA yer alır
# (0'a yakın VE 180'e yakın). Bu yüzden kırmızıyı yakalamak için
# TEK bir aralık yetmez, İKİ AYRI aralık tanımlayıp bunları birleştiririz.
#
# Saturation'ı (150) ve Value'yu (90) yüksek tuttuk, çünkü:
# İnsan teni de düşük doygunlukta kırmızımsı bir renktir. Bu sınırları
# düşük tutarsak program YÜZÜ VEYA ELİ "kırmızı nesne" sanıp yanlış
# kutu çizebilir. Yüksek tutunca sadece GERÇEKTEN CANLI VE PARLAK
# kırmızılar (kırmızı bir kupa, kırmızı bir top gibi) yakalanır.

# 1. kırmızı aralığı (0-10 arası ton değerleri)
kirmizi_alt_1 = np.array([0, 150, 90])     # [Ton, Doygunluk, Parlaklık] alt sınır
kirmizi_ust_1 = np.array([10, 255, 255])   # [Ton, Doygunluk, Parlaklık] üst sınır

# 2. kırmızı aralığı (170-180 arası ton değerleri)
kirmizi_alt_2 = np.array([170, 150, 90])
kirmizi_ust_2 = np.array([180, 255, 255])

# ----------------------------------------------------
# GÜRÜLTÜ FİLTRESİ AYARI
# ----------------------------------------------------
# Kameradaki küçük ışık yansımaları da bazen "kırmızı" görünebilir.
# Bunların alanı (piksel sayısı) genelde küçüktür. Bu yüzden alanı
# bu sayıdan KÜÇÜK olan şekilleri "gürültü" sayıp yok sayacağız.
kucuk_alan_siniri = 800

# ----------------------------------------------------
# GÜRÜLTÜ TEMİZLEME ARACI (KERNEL)
# ----------------------------------------------------
# Aşağıda kullanacağımız morfolojik işlemler (temizleme), maskeyi
# küçük bir 5x5'lik "pencere" ile tarayarak çalışır. Bu diziye
# "çekirdek" (kernel) denir.
temizleme_araci = np.ones((5, 5), np.uint8)


# ======================================================
# 2) ANA DÖNGÜ - HER KAREYİ (RESMİ) SÜREKLİ OKU VE İŞLE
# ======================================================
# "while True" -> program 'q' tuşuna basılana kadar bu bloğu
# sonsuz şekilde tekrar tekrar çalıştırır. Video dediğimiz şey
# aslında saniyede onlarca resmin art arda gösterilmesidir; bu
# yüzden "canlı görüntü" elde etmek için sürekli yeni kare okumamız
# gerekir.
while True:

    # ------------------------------------------------
    # 3) KAMERADAN BİR KARE (RESİM) OKU
    # ------------------------------------------------
    # kamera.read() iki şey döndürür:
    #   basarili -> okuma başarılı oldu mu? (True / False)
    #   kare     -> okunan resmin kendisi (piksellerden oluşan dizi)
    basarili, kare = kamera.read()

    # Eğer kare okunamadıysa (örneğin kamera koptuysa),
    # döngüden çıkıp programı durduruyoruz.
    if not basarili:
        break

    # ------------------------------------------------
    # 4) RENGİ HSV FORMATINA ÇEVİR
    # ------------------------------------------------
    # Kamera görüntüsü normalde BGR (Mavi-Yeşil-Kırmızı) formatındadır.
    # Renk tespiti için HSV formatı çok daha uygundur çünkü ışık
    # değişse bile (oda karanlıklaşsa/aydınlansa bile) bir rengin
    # Ton (Hue) değeri fazla değişmez. Bu yüzden her kareyi HSV'ye
    # çeviriyoruz.
    hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------
    # 5) KIRMIZI RENGİ MASKELE
    # ------------------------------------------------
    # cv2.inRange(): belirttiğimiz alt-üst sınırlar arasında kalan
    # pikselleri BEYAZ (255), aralık dışında kalanları SİYAH (0)
    # yapar. Sonuç, sadece siyah-beyazdan oluşan bir "maske" resmidir.

    # İlk kırmızı aralığı için maske
    maske_1 = cv2.inRange(hsv, kirmizi_alt_1, kirmizi_ust_1)
    # İkinci kırmızı aralığı için maske
    maske_2 = cv2.inRange(hsv, kirmizi_alt_2, kirmizi_ust_2)

    # cv2.bitwise_or(): iki maskeyi birleştirir. Bir piksel,
    # maske_1'de VEYA maske_2'de beyazsa, sonuçta da beyaz olur.
    # (Böylece kırmızının HER İKİ ucunu da yakalamış oluruz.)
    maske = cv2.bitwise_or(maske_1, maske_2)

    # ------------------------------------------------
    # 6) MASKEDEKİ GÜRÜLTÜYÜ TEMİZLE
    # ------------------------------------------------
    # MORPH_OPEN: önce küçültür (erozyon) sonra büyütür (genişletme).
    # Bu, tek tek küçük beyaz noktacıkları (gürültüyü) tamamen
    # siler ama büyük nesneyi korur.
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, temizleme_araci)

    # MORPH_CLOSE: önce büyütür sonra küçültür.
    # Bu da nesnenin İÇİNDE kalan küçük siyah boşlukları/delikleri
    # doldurur, böylece nesne daha düzgün/dolu bir şekil olur.
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, temizleme_araci)

    # ------------------------------------------------
    # 7) MASKEDEKİ ŞEKİLLERİN (KONTURLARIN) SINIRLARINI BUL
    # ------------------------------------------------
    # Kontur = beyaz bir bölgenin dış hattı/sınır çizgisi.
    # cv2.findContours() maskedeki TÜM beyaz bölgelerin konturlarını
    # bir liste olarak döndürür.
    #   RETR_EXTERNAL      -> sadece en dıştaki sınırları al
    #   CHAIN_APPROX_SIMPLE -> konturu daha az noktayla, sade bir
    #                          şekilde sakla (bellekten tasarruf)
    konturlar, _ = cv2.findContours(
        maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # ------------------------------------------------
    # 8) EN BÜYÜK KIRMIZI ŞEKLİ BUL VE ETRAFINA KUTU ÇİZ
    # ------------------------------------------------
    # Ekranda birden fazla kırmızımsı bölge olabilir. Genelde asıl
    # takip etmek istediğimiz nesne, ARADAKİ EN BÜYÜK ŞEKİLDİR.
    if konturlar:  # Liste boş değilse (en az bir şekil bulunduysa)

        # max(..., key=cv2.contourArea): listedeki her şeklin
        # alanını hesaplayıp en büyük alana sahip olanı seçer.
        en_buyuk_sekil = max(konturlar, key=cv2.contourArea)

        # Bulduğumuz "en büyük" şekil bile çok küçükse, bu muhtemelen
        # gerçek bir nesne değil, gürültüdür. Bu yüzden alan
        # kontrolü yapıyoruz.
        if cv2.contourArea(en_buyuk_sekil) > kucuk_alan_siniri:

            # cv2.boundingRect(): şekli tam olarak çevreleyen,
            # kenarları düz (eğik olmayan) bir dikdörtgen hesaplar.
            # x, y -> dikdörtgenin SOL ÜST köşesinin konumu
            # genislik, yukseklik -> dikdörtgenin boyutları
            x, y, genislik, yukseklik = cv2.boundingRect(en_buyuk_sekil)

            # cv2.rectangle(resim, sol_ust_nokta, sag_alt_nokta,
            #                renk, kalinlik)
            # Renkler OpenCV'de BGR sırasıyladır: (0, 255, 0) = YEŞİL
            # (Blue=0, Green=255, Red=0)
            cv2.rectangle(
                kare,
                (x, y),
                (x + genislik, y + yukseklik),
                (0, 255, 0),   # yeşil renk
                2              # çizgi kalınlığı (piksel)
            )

            # cv2.putText(): dikdörtgenin biraz üstüne açıklayıcı
            # bir yazı ekler, kutuyla çakışmasın diye 10 piksel
            # yukarı kaydırıyoruz (y - 10).
            cv2.putText(
                kare,
                "Kirmizi Nesne",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,  # yazı tipi
                0.6,                       # yazı boyutu
                (0, 255, 0),               # yeşil renk
                2                          # kalınlık
            )

    # ------------------------------------------------
    # 9) SONUÇLARI EKRANDA GÖSTER
    # ------------------------------------------------
    # İki ayrı pencere açıyoruz:
    #   "Kirmizi Nesne Takibi" -> normal renkli görüntü + kutu
    #                             (kullanıcının asıl gördüğü pencere)
    #   "Maske"                -> siyah/beyaz maske görüntüsü
    #                             (HSV ayarlarının doğru çalışıp
    #                             çalışmadığını KONTROL etmek için
    #                             faydalıdır; nesne beyaz, geri
    #                             kalan her şey siyah görünmelidir)
    cv2.imshow("Kirmizi Nesne Takibi", kare)
    cv2.imshow("Maske", maske)

    # ------------------------------------------------
    # 10) 'q' TUŞUNA BASILDI MI KONTROL ET
    # ------------------------------------------------
    # cv2.waitKey(1): 1 milisaniye boyunca klavyeden bir tuşa
    # basılmasını bekler. Bu satır aynı zamanda pencerelerin
    # ekranda düzgün güncellenmesi için de gereklidir; bu satır
    # olmadan pencereler donuk görünür.
    # ord("q"): 'q' harfinin kod karşılığıdır. Basılan tuş bu ise
    # döngüyü kırıp programı bitiriyoruz.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------------------------------
# 11) KAMERAYI KAPAT VE PENCERELERİ TEMİZLE
# ----------------------------------------------------
# Bu adım ÇOK ÖNEMLİDİR. Kamerayı serbest bırakmazsak (release
# etmezsek), kamera "meşgul" görünmeye devam edebilir ve program
# kapansa bile başka uygulamalar (hatta bu programı tekrar
# çalıştırdığında kendisi) kamerayı açamayabilir.
kamera.release()

# Açık olan tüm OpenCV pencerelerini kapatır.
cv2.destroyAllWindows()
