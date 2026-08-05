# Gerekli kütüphaneleri projeye dahil ediyoruz
import json     # Vosk çıktısını (JSON formatında gelir) Python sözlüğüne çevirmek için
import pyaudio  # Mikrofondan canlı ses akışı (stream) almak için
from vosk import Model, KaldiRecognizer # Vosk'un dil modeli ve tanıma motoru sınıfları

# 1. DİL MODELİ VE TANIMA MOTORUNUN HAZIRLANMASI
# -------------------------------------------------------------------------
# Proje klasöründeki indirilmiş Türkçe Vosk model dosyasını belleğe yüklüyoruz.
model = Model("vosk-model-small-tr-0.3")

# KaldiRecognizer, tanımlanan modeli ve ses frekansını (16000 Hz) kullanarak 
# sesi metne dönüştürecek ana işleme motorunu oluşturur.
recognizer = KaldiRecognizer(model, 16000)

# 2. MİKROFON AKIŞININ (AUDIO STREAM) BAŞLATILMASI
# -------------------------------------------------------------------------
# PyAudio nesnesini başlatıyoruz (donanım sürücüleriyle bağlantı kurar)
p = pyaudio.PyAudio()

# Mikrofondan ses verisi okuyabilmek için bir akış (stream) açıyoruz
stream = p.open(
    format=pyaudio.paInt16,  # Sesi 16-bit tam sayı formatında al (Vosk'un beklediği standart)
    channels=1,              # Mono ses kanalı (tek kanal, Vosk için yeterlidir)
    rate=16000,              # Örnekleme hızı (16 kHz - KaldiRecognizer ile aynı olmalıdır)
    input=True,              # Cihazı ses ÇIKIŞI (hoparlör) değil, ses GİRİŞİ (mikrofon) olarak seçer
    frames_per_buffer=8000   # Her okumada belleğe alınacak olan ses dilimi büyüklüğü
)

# Ses akışını aktif hale getiriyoruz (mikrofonu dinlemeye başlar)
stream.start_stream()

print("Vosk dinliyor (Çevrimdışı)... Konuşabilirsiniz.")

# 3. CANLI SES AKIŞINI SÜREKLİ DİNLEME VE ÇÖZÜMLEME DÖNGÜSÜ
# -------------------------------------------------------------------------
while True:
    # Mikrofondan 4000 karelik (frame) ham ses verisini okuyoruz.
    # exception_on_overflow=False: İşlemci yetişemezse uygulamanın çökmesini önler, fazla veriyi atlar.
    data = stream.read(4000, exception_on_overflow=False)
    
    # AcceptWaveform(data): Gelen ses verisini işler.
    # Eğer konuşmacı bir duraksama yaptıysa veya cümle bittiyse 'True' döner.
    if recognizer.AcceptWaveform(data):
        # Result(): Tamamlanmış cümlenin sonucunu string (JSON) olarak verir.
        # json.loads(): String olan JSON verisini Python sözlük (dict) yapısına çevirir.
        result = json.loads(recognizer.Result())
        
        # 'text' anahtarındaki tamamlanmış cümleyi ekrana yazdırır.
        print(f"Cümle: {result['text']}")
    else:
        # Konuşma henüz bitmediyse, devam ediyorsa bu blok çalışır.
        # PartialResult(): Konuşma anındaki geçici/anlık tahminleri getirir.
        partial = json.loads(recognizer.PartialResult())
        
        # 'partial' anahtarında anlık bir tahmin varsa bunu aynı satırda güncelleyerek yazdırır.
        # end='\r': İmleci satırın başına döndürerek anlık yazının üst üste güncellenmesini sağlar.
        if partial['partial']:
            print(f"Anlık: {partial['partial']}", end='\r')
