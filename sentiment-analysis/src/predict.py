"""
Tek yorum tahmini modülü.
Kaydedilmiş en iyi modeli yükler ve ham Türkçe metni işleyerek tahmin döndürür.
"""

import os
import joblib
from preprocess import metin_temizle

VARSAYILAN_MODEL = os.path.join("models", "lojistik_regresyon.pkl")


# ---------------------------------------------------------------------------
# Model Yükleme
# ---------------------------------------------------------------------------

def model_yukle(model_yolu: str = VARSAYILAN_MODEL):
    """
    Diske kaydedilmiş sklearn Pipeline'ı yükler.

    Parameters
    ----------
    model_yolu : str
        .pkl dosyasının yolu.

    Returns
    -------
    sklearn Pipeline
        Yüklenmiş model.
    """
    try:
        return joblib.load(model_yolu)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model bulunamadı: {model_yolu}\n"
            "Önce 'python train.py' komutuyla modeli eğitin."
        )


# ---------------------------------------------------------------------------
# Tek Yorum Tahmini
# ---------------------------------------------------------------------------

def yorumu_tahmin_et(yorum: str, model=None, model_yolu: str = VARSAYILAN_MODEL) -> dict:
    """
    Ham Türkçe yorumu temizleyip duygu tahmini yapar.

    Parameters
    ----------
    yorum : str
        Ham Türkçe yorum metni.
    model : sklearn Pipeline, optional
        Önceden yüklenmiş model. None ise diskten yüklenir.
    model_yolu : str
        Model yüklenecekse kullanılacak .pkl yolu.

    Returns
    -------
    dict
        duygu, guven_skoru, temiz_metin anahtarlarını içerir.
    """
    if model is None:
        model = model_yukle(model_yolu)

    temiz = metin_temizle(yorum)
    tahmin = model.predict([temiz])[0]

    # Olasılık skoru varsa güven skoru hesapla
    try:
        olasiliklar = model.predict_proba([temiz])[0]
        siniflar = list(model.classes_)
        guven = float(olasiliklar[siniflar.index(tahmin)])
    except AttributeError:
        guven = 1.0   # predict_proba desteklenmiyorsa (NB bazı durumlarda)

    return {
        "duygu": tahmin,
        "guven_skoru": round(guven, 4),
        "temiz_metin": temiz,
    }


# ---------------------------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    yorumlar = [
        "Bu ürün gerçekten harika, çok memnun kaldım!",
        "Kargo çok geç geldi ve ürün hasarlıydı. Berbat.",
        "Fiyatına göre idare eder, ne iyi ne kötü.",
        "Servis personeli çok ilgiliydi, teşekkürler.",
        "Ürün kalitesi düşük, para ziyanı.",
    ]

    print("=== Tahmin Demo ===")
    for yorum in yorumlar:
        try:
            sonuc = yorumu_tahmin_et(yorum)
            print(f"\n  Yorum  : {yorum}")
            print(f"  Duygu  : {sonuc['duygu']}  (güven: {sonuc['guven_skoru']:.2%})")
        except FileNotFoundError as e:
            print(f"\n  [UYARI] {e}")
            break
