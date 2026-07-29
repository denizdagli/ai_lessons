"""
DERS 2: Yapay Zeka ile Nesne Algılama (YOLOv8)
================================================
Bu programda:
  1. Ultralytics kütüphanesinden hazır (pretrained) bir YOLOv8 modeli yükleyeceğiz.
     Bu model, COCO veri setiyle önceden eğitilmiştir ve 80 farklı nesne sınıfını
     (insan, bardak, telefon, sandalye, köpek, araba, ...) tanıyabilir.
  2. Webcam'den canlı görüntü alacağız (Ders 1'deki kamera döngüsünü hatırla).
  3. Her karede modeli çalıştırıp bulunan nesnelerin etrafına kutu, üstüne de
     sınıf adı ve güven skorunu (confidence) çizeceğiz.

Gereksinim:
    pip install ultralytics

NOT: Programı ilk çalıştırdığında model dosyası (yolov8n.pt) otomatik olarak
internetten indirilecek. Bu birkaç saniye ile bir dakika arası sürebilir,
sonraki çalıştırmalarda tekrar indirilmez.
"""

import cv2
from ultralytics import YOLO

# ------------------------------------------------------------------
# 1) MODELİ YÜKLE
# ------------------------------------------------------------------
# "yolov8n.pt" -> "n" = nano, en küçük ve en hızlı YOLOv8 sürümü.
# Daha yüksek doğruluk isteyen ama daha yavaş çalışan sürümler de var:
# yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large), yolov8x.pt (extra large)
# Eğitim/demo amaçlı, hızlı çalışması için "n" sürümünü kullanıyoruz.
model = YOLO("yolov8n.pt")

# ------------------------------------------------------------------
# 2) GÜVEN SKORU (CONFIDENCE) EŞİĞİ
# ------------------------------------------------------------------
# Modelin bir tespiti "gerçek" saymak için ne kadar emin olması gerektiğini
# belirler (0.0 - 1.0 arası).
#   - Düşük değer (örn. 0.1): Daha fazla nesne tespit eder ama yanlış
#     tespitler (false positive) artar.
#   - Yüksek değer (örn. 0.7): Daha az ama daha güvenilir tespitler yapar,
#     bazı gerçek nesneleri kaçırabilir.
#
# EGZERSİZ: Bu değeri değiştirip (örn. önce 0.25, sonra 0.7 dene) kamerada
# neyin değiştiğini gözlemle.
GUVEN_ESIGI = 0.4


def ana_program():
    # 0 -> bilgisayarın varsayılan (dahili) kamerası.
    kamera = cv2.VideoCapture(0)

    if not kamera.isOpened():
        print("HATA: Kamera açılamadı. Başka bir uygulama kamerayı kullanıyor olabilir.")
        return

    print("Program başladı. Çıkmak için 'q' tuşuna basın.")

    while True:
        basarili, kare = kamera.read()
        if not basarili:
            print("Kameradan görüntü alınamadı.")
            break

        # Ayna etkisi için kareyi yatayda çeviriyoruz (isteğe bağlı, doğal görünür)
        kare = cv2.flip(kare, 1)

        # ------------------------------------------------------------------
        # 3) MODELİ ÇALIŞTIR
        # ------------------------------------------------------------------
        # model(kare, ...) bize o karedeki TÜM tespitleri (kutu, sınıf, skor)
        # tek bir çağrıda döndürür. verbose=False -> konsola gereksiz log
        # basmasını engeller (her karede satırlarca çıktı istemiyoruz).
        sonuclar = model(kare, conf=GUVEN_ESIGI, verbose=False)

        # sonuclar[0] -> bu karenin sonucu (model tek kareyle çalıştığı için
        # liste içinde tek eleman döner)
        tespitler = sonuclar[0]

        # ------------------------------------------------------------------
        # 4) HER TESPİT İÇİN KUTU VE ETİKET ÇİZ
        # ------------------------------------------------------------------
        for kutu in tespitler.boxes:
            # Kutunun koordinatları (sol-üst ve sağ-alt köşe)
            x1, y1, x2, y2 = map(int, kutu.xyxy[0])

            # Sınıf numarası -> sınıf ismi (örn. 0 -> "person", 41 -> "cup")
            sinif_id = int(kutu.cls[0])
            sinif_adi = model.names[sinif_id]

            # Güven skoru (0.0 - 1.0)
            guven = float(kutu.conf[0])

            # Kutuyu çiz
            cv2.rectangle(kare, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Etiketi (sınıf adı + yüzde olarak güven skoru) kutunun üstüne yaz
            etiket = f"{sinif_adi} %{guven * 100:.0f}"
            cv2.putText(
                kare, etiket, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

        # Kaç nesne bulunduğunu ekranın sol üstüne yazalım (küçük ekstra bilgi)
        cv2.putText(
            kare, f"Bulunan nesne: {len(tespitler.boxes)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("YOLOv8 - Nesne Tespiti", kare)

        # 'q' tuşuna basılırsa döngüden çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ana_program()
