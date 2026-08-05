import pyttsx3


def dinamik_seslendir(metin, hiz=160, ses_seviyesi=0.9):
    """Sistemdeki mevcut sesleri listeleyen, kullanıcıya seçim yaptıran ve

    verilen metni seçilen ses ile seslendiren fonksiyon.
    """

    # 1. ADIM: pyttsx3 ses motorunu başlatıyoruz
    engine = pyttsx3.init()

    # 2. ADIM: Konuşma hızı (rate) ve ses seviyesi (volume) parametrelerini ayarlıyoruz
    # hiz: Varsayılan değer genelde 200'dür, 160 daha doğal bir okuma sunar.
    # ses_seviyesi: 0.0 (sessiz) ile 1.0 (maksimum) arasında bir değer alır.
    engine.setProperty("rate", hiz)
    engine.setProperty("volume", ses_seviyesi)

    # 3. ADIM: İşletim sisteminde yüklü olan tüm TTS (Text-to-Speech) seslerini alıyoruz
    voices = engine.getProperty("voices")

    # Sistemdeki sesleri kullanıcıya düzenli bir listede gösteriyoruz
    print("\n--- Sistemde Yüklü Sesler ---")
    for index, voice in enumerate(voices):
        # Eğer ses için tanımlı dil varsa aralarına virgül koyarak yazdırıyoruz, yoksa 'Bilinmiyor' yazıyoruz
        diller = (
            ", ".join(voice.languages) if voice.languages else "Bilinmiyor"
        )
        print(f"[{index}] İsim: {voice.name} | Dil: {diller}")
    print("-----------------------------\n")

    # 4. ADIM: Kullanıcı geçerli bir indeks girene kadar döngüde kalıyoruz
    while True:
        try:
            # Kullanıcıdan konsol üzerinden bir sayı alıyoruz
            secim_girdisi = input(
                f"Lütfen kullanmak istediğiniz sesin numarasını girin (0 - {len(voices)-1}): "
            )
            secim = int(secim_girdisi)

            # Girilen sayının liste aralığında olup olmadığını kontrol ediyoruz
            if 0 <= secim < len(voices):
                # Seçilen sesi alıp motorun 'voice' özelliğine atıyoruz
                secilen_voice = voices[secim]
                engine.setProperty("voice", secilen_voice.id)
                print(f"\nSeçilen Ses: {secilen_voice.name}\n")
                break  # Geçerli seçim yapıldığı için döngüden çıkıyoruz
            else:
                print(
                    "Geçersiz bir numara girdiniz, lütfen listedeki aralıkta bir sayı seçin."
                )

        except ValueError:
            # Kullanıcı harf veya özel karakter girerse hatayı yakalayıp uyarırız
            print("Hata: Lütfen geçerli bir tamsayı girin!")

    # 5. ADIM: Metni okuma kuyruğuna ekliyoruz ve seslendirmeyi başlatıyoruz
    engine.say(metin)

    # engine.runAndWait(): Seslendirme bitene kadar Python betiğinin kapanmasını engeller ve bekletir
    engine.runAndWait()


# Betik doğrudan çalıştırıldığında bu blok devreye girer
if __name__ == "__main__":
    # Seslendirilecek örnek metin
    okunacak_metin = "Merhaba! Seçtiğiniz ses ile metin başarıyla okunuyor."

    # Fonksiyonu çağırıyoruz
    dinamik_seslendir(okunacak_metin)
