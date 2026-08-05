import speech_recognition as sr

# 1. Tanımlayıcı (Recognizer) nesnesini oluşturuyoruz
r = sr.Recognizer()

# 2. Mikrofonu ses kaynağı olarak tanımlıyoruz
with sr.Microphone() as source:
    print("Konuşabilirsiniz, sizi dinliyorum...")
    
    # Ortamdaki gürültüyü engellemek için ses seviyesini ayarlıyoruz (opsiyonel ama faydalı)
    r.adjust_for_ambient_noise(source)
    
    # Mikrofondan gelen sesi dinliyoruz
    audio = r.listen(source)
    print("Ses kaydedildi, metne dönüştürülüyor...")

# 3. Alınan sesi metne çevirme aşaması
try:
    # Google Speech Recognition servisi ile sesi Türkçe (tr-TR) olarak çözümleme
    text = r.recognize_google(audio, language="tr-TR")
    print(f"Söylediğiniz metin: {text}")

except sr.UnknownValueError:
    # Ses anlaşılamadığında çalışan kısım
    print("Ses anlaşılamadı, lütfen tekrar deneyin.")

except sr.RequestError as e:
    # İnternet bağlantısı veya Google servisiyle ilgili sorun olduğunda
    print(f"Google servislerine ulaşılamadı; {e}")
