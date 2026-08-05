import cv2  # Görüntü işleme işlemleri ve kamera erişimi için OpenCV kütüphanesini içe aktarıyoruz.
import mediapipe as mp  # El takibi ve eklem tespiti yapabilmek için MediaPipe kütüphanesini içe aktarıyoruz.

# MediaPipe El Takibi Modülünü Başlatma
mp_hands = mp.solutions.hands  # MediaPipe içindeki el tespiti ve takip modülüne erişim sağlıyoruz.
mp_drawing = mp.solutions.drawing_utils  # El eklem noktalarını ekrana çizdirebilmek için çizim araçlarını yüklüyoruz.

hands = mp_hands.Hands(  # El takip modelini belirttiğimiz parametrelerle yapılandırıp başlatıyoruz.
    static_image_mode=False,  # Video akışı kullanacağımız için sürekli tespit yerine takip modunu aktif ediyoruz.
    max_num_hands=1,  # Performansı artırmak ve çakışmaları önlemek için sadece 1 eli takip edecek şekilde sınırlıyoruz.
    min_detection_confidence=0.7,  # Bir elin ilk tespit edilmesi için gereken minimum güven oranını (%70) belirliyoruz.
    min_tracking_confidence=0.7  # Tespit edilen elin takibinin sürdürülmesi için gereken minimum güven oranını (%70) belirliyoruz.
)

# Kamera Başlatma
cap = cv2.VideoCapture(0)  # Bilgisayarın varsayılan kamerasını (0 numaralı indeks) başlatıyoruz.

def finger_states(landmarks):  # El eklem noktalarını alıp parmakların açık/kapalı durumunu döndüren fonksiyonu tanımlıyoruz.
    """Parmakların açık mı kapalı mı olduğunu tespit eder (Y eksenine göre)"""  # Fonksiyonun amacını belirten açıklama metni (docstring).
    # Eklemlerin uç ve alt noktaları (Tip vs PIP)
    tips = [8, 12, 16, 20]  # İşaret, Orta, Yüzük ve Serçe parmaklarının uç noktalarının (Landmark) indeks numaraları.
    pips = [6, 10, 14, 18]  # İşaret, Orta, Yüzük ve Serçe parmaklarının orta boğum eklem noktalarının indeks numaraları.
    
    fingers = []  # Parmak durumlarını (1: Açık, 0: Kapalı) saklayacağımız boş bir liste oluşturuyoruz.
    
    # Dört ana parmağın kontrolü (Açık = 1, Kapalı = 0)
    for tip, pip in zip(tips, pips):  # Her bir parmağın uç ve orta eklem indekslerini eşleştirip döngüye sokuyoruz.
        if landmarks[tip].y < landmarks[pip].y:  # Ekranda Y ekseni yukarıdan aşağı arttığı için, uç nokta orta eklemden yukarıdaysa parmak açıktır.
            fingers.append(1)  # Parmak açıksa listeye 1 ekliyoruz.
        else:  # Uç nokta orta eklemin altındaysa parmak kapalıdır.
            fingers.append(0)  # Parmak kapalıysa listeye 0 ekliyoruz.
            
    return fingers  # Dört parmağın durumunu içeren listeyi [İşaret, Orta, Yüzük, Serçe] fonksiyon çıktısı olarak döndürüyoruz.

while cap.isOpened():  # Kamera açık ve erişilebilir olduğu sürece döngüyü sürdürüyoruz.
    success, frame = cap.read()  # Kameradan bir kare görüntü okuyoruz; 'success' durum bilgisini, 'frame' ise resmi tutar.
    if not success:  # Eğer kameradan görüntü alınamazsa (örneğin kamera bağlantısı koptuysa):
        print("Kamera görüntüsü alınamadı.")  # Kullanıcıya konsolda hata mesajı gösteriyoruz.
        continue  # Döngünün başına dönerek sonraki kareyi okumayı tekrar deniyoruz.

    # Aynalama ve BGR -> RGB Dönüşümü
    frame = cv2.flip(frame, 1)  # Görüntüyü yatay eksende çevirerek (aynalama) doğal bir ayna etkisi elde ediyoruz.
    h, w, c = frame.shape  # Alınan kare görüntünün yükseklik (h), genişlik (w) ve kanal sayısı (c) değerlerini çekiyoruz.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV varsayılan BGR formatını, MediaPipe'ın işleyebildiği RGB formatına dönüştürüyoruz.
    
    # MediaPipe İşleme
    results = hands.process(rgb_frame)  # RGB formatındaki kareyi MediaPipe modeline göndererek el ve eklem tespiti yapıyoruz.
    
    command = "SERBEST"  # Varsayılan durumu/komutu "SERBEST" olarak belirliyoruz.

    if results.multi_hand_landmarks:  # Eğer karede en az bir el tespit edildiyse:
        for hand_landmarks in results.multi_hand_landmarks:  # Tespit edilen her bir el için eklem noktaları üzerinde döngü başlatıyoruz.
            # Eklemleri ve bağlantıları ekrana çiz
            mp_drawing.draw_landmarks(  # MediaPipe çizim aracını kullanarak tespit edilen eli çizdiriyoruz.
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS  # Orijinal kare üzerine, el noktalarını ve aralarındaki bağlantı çizgilerini ekliyoruz.
            )
            
            lm = hand_landmarks.landmark  # El üzerindeki 21 adet eklem noktasının normalize edilmiş (0-1 arası) koordinat listesini alıyoruz.
            open_fingers = finger_states(lm)  # Parmakların açık/kapalı durumunu belirlemek için fonksiyonumuzu çağırıyoruz.
            
            # --- HAREKET KONTROL ALGORİTMASI ---
            
            # 1. DUR Komutu: Tüm parmaklar açık (Açık el / Avuç içi)
            if open_fingers == [1, 1, 1, 1]:  # İşaret, Orta, Yüzük ve Serçe parmaklarının tümü açıksa:
                command = "DUR"  # Ekrana yazdırılacak komutu "DUR" olarak güncelliyoruz.
                
            # 2. SAĞA DÖN / SOLA DÖN Komutu: Sadece işaret parmağı açık ve yatay yönlenmiş
            elif open_fingers == [1, 0, 0, 0]:  # Sadece işaret parmağı açık, diğer 3 parmak kapalıysa:
                index_tip_x = lm[8].x  # İşaret parmağı ucunun (Landmark 8) yatay (X) koordinatını alıyoruz.
                index_mcp_x = lm[5].x  # İşaret parmağı kökünün (Landmark 5) yatay (X) koordinatını alıyoruz.
                
                # İşaret parmağı belirgin şekilde sağa veya sola bakıyorsa
                if index_tip_x - index_mcp_x > 0.1:  # Parmak ucu, kök noktasına göre belirgin şekilde sağdaysa:
                    command = "SAGA DON"  # Komutu "SAGA DON" olarak güncelliyoruz.
                elif index_mcp_x - index_tip_x > 0.1:  # Parmak ucu, kök noktasına göre belirgin şekilde soldaysa:
                    command = "SOLA DON"  # Komutu "SOLA DON" olarak güncelliyoruz.
                else:  # Parmak dik duruyorsa veya yatay eğim yetersizse:
                    command = "YON BELIRTIN"  # Kullanıcıdan parmağı yana yatırmasını isteyecek metni atıyoruz.
                    
            # 3. YOL AL / İLERİ Komutu: Yumruk yapılmış (Tüm parmaklar kapalı)
            elif open_fingers == [0, 0, 0, 0]:  # Dört parmağın tümü kapalıysa (yumruk yapıldıysa):
                command = "ILERI"  # Komutu "ILERI" olarak güncelliyoruz.

            # Koordinat çıkarma örneği (İşaret parmağı ucu - Landmark 8)
            cx, cy = int(lm[8].x * w), int(lm[8].y * h)  # Normalize edilmiş X ve Y koordinatlarını piksellere dönüştürüyoruz.
            cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # İşaret parmağının ucuna 10 piksel yarıçapında mavi dolu bir daire çiziyoruz.

    # Ekrana Komutu Yazdırma (Bilgi Paneli)
    cv2.rectangle(frame, (20, 20), (350, 80), (0, 0, 0), cv2.FILLED)  # Sol üst köşeye komut metninin arka planı için siyah bir dikdörtgen çiziyoruz.
    cv2.putText(frame, f"KOMUT: {command}", (30, 60),  # Dikdörtgenin içine o anki aktif komutu yazdırıyoruz.
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Yeşil renkte, 1 büyüklüğünde ve yumuşatılmış kenarlı yazı tipi kullanıyoruz.

    # Görüntüyü Göster
    cv2.imshow("MediaPipe El ve Hareket Takibi", frame)  # İşlenmiş kareden oluşan pencereyi ekranda gösteriyoruz.

    # 'q' tuşuna basarak çıkış yapabilirsiniz
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Klavye girdisini 1 milisaniye bekliyoruz; 'q' tuşuna basıldıysa:
        break  # Sonsuz kamera döngüsünden çıkıyoruz.

cap.release()  # Kamerayı serbest bırakarak kaynağı sistem kullanımına geri veriyoruz.
cv2.destroyAllWindows()  # OpenCV tarafından açılmış olan tüm pencereleri kapatıyoruz.
