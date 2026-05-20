"""
Türkçe müşteri yorumları için veri ön işleme modülü.
Gerçek veri sütunları: text (yorum metni), label (Positive/Notr/Negative)
Veri zaten train/test olarak ayrılmış; train.csv ve test.csv ayrı okunur.
"""

import re
import os
import pandas as pd
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK stopword listesini indir (ilk çalıştırmada gerekli)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

# Türkçe özel stopword listesi (NLTK'ya ek)
OZEL_STOPWORD = {
    "bir", "bu", "ve", "ile", "de", "da", "mi", "mu", "mı", "mü",
    "için", "çok", "daha", "en", "ne", "ki", "ama", "fakat", "ya",
    "hem", "hiç", "her", "biz", "siz", "onlar", "ben", "sen", "o",
    "bunu", "şunu", "bunun", "şunun", "gibi", "kadar", "bile",
}

try:
    STOPWORDS = set(stopwords.words("turkish")) | OZEL_STOPWORD
except OSError:
    STOPWORDS = OZEL_STOPWORD

# Etiket eşleme: İngilizce/büyük harf → Türkçe küçük harf
ETIKET_ESLE = {
    "positive": "olumlu",
    "negative": "olumsuz",
    "notr":     "notr",
    "Positive": "olumlu",
    "Negative": "olumsuz",
    "Notr":     "notr",
}


# ---------------------------------------------------------------------------
# 1. Veri Okuma
# ---------------------------------------------------------------------------

def veri_oku(dosya_yolu: str) -> pd.DataFrame:
    """
    CSV dosyasını okur; 'text' ve 'label' sütunlarını döndürür.

    Parameters
    ----------
    dosya_yolu : str
        train.csv veya test.csv dosyasının yolu.

    Returns
    -------
    pd.DataFrame
        Sadece 'text' ve 'label' sütunlarını içeren DataFrame.
    """
    try:
        df = pd.read_csv(dosya_yolu, encoding="utf-8")
        gerekli = {"text", "label"}
        eksik = gerekli - set(df.columns)
        if eksik:
            raise ValueError(f"CSV'de eksik sütunlar: {eksik}")
        return df[["text", "label"]].copy()
    except FileNotFoundError:
        raise FileNotFoundError(f"Dosya bulunamadı: {dosya_yolu}")
    except Exception as e:
        raise RuntimeError(f"CSV okunurken hata: {e}") from e


# ---------------------------------------------------------------------------
# 2. Boş Satır Silme
# ---------------------------------------------------------------------------

def bos_satir_sil(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boş veya sadece boşluk içeren text/label satırlarını siler.

    Parameters
    ----------
    df : pd.DataFrame
        Ham DataFrame.

    Returns
    -------
    pd.DataFrame
        Boş satırlar çıkarılmış DataFrame.
    """
    baslangic = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  Boş satır silindi: {baslangic - len(df)} satır kaldırıldı.")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3–6. Metin Temizleme
# ---------------------------------------------------------------------------

def metin_temizle(metin: str) -> str:
    """
    Tek bir metin üzerinde tüm temizleme adımlarını uygular:
    küçük harf → URL/sayı temizle → noktalama sil → stopword çıkar.

    Parameters
    ----------
    metin : str
        Ham yorum metni.

    Returns
    -------
    str
        Temizlenmiş metin.
    """
    metin = str(metin).lower()
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)          # URL
    metin = re.sub(r"\d+", " ", metin)                        # Sayılar
    metin = re.sub(r"[^\w\sçğışöüÇĞİŞÖÜ]", " ", metin)      # Noktalama
    metin = re.sub(r"\s+", " ", metin).strip()
    kelimeler = [k for k in metin.split() if k not in STOPWORDS and len(k) > 1]
    return " ".join(kelimeler)


def metinleri_temizle(df: pd.DataFrame) -> pd.DataFrame:
    """
    'text' sütununa temizleme pipeline'ını uygular; 'temiz_metin' sütunu ekler.

    Parameters
    ----------
    df : pd.DataFrame
        'text' sütunu olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'temiz_metin' sütunu eklenmiş DataFrame.
    """
    df["temiz_metin"] = df["text"].apply(metin_temizle)
    df = df[df["temiz_metin"].str.strip() != ""].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 7. Etiket Dönüşümü  (Positive/Notr/Negative → olumlu/notr/olumsuz)
# ---------------------------------------------------------------------------

def etiket_donustur(df: pd.DataFrame) -> pd.DataFrame:
    """
    'label' sütunundaki İngilizce etiketleri Türkçe 'duygu' sütununa çevirir.

    Positive  → olumlu
    Negative  → olumsuz
    Notr      → notr

    Parameters
    ----------
    df : pd.DataFrame
        'label' sütunu olan DataFrame.

    Returns
    -------
    pd.DataFrame
        'duygu' sütunu eklenmiş DataFrame.
    """
    df["duygu"] = df["label"].map(ETIKET_ESLE)
    bilinmeyen = df["duygu"].isna().sum()
    if bilinmeyen > 0:
        print(f"  [UYARI] Eşlenemeyen {bilinmeyen} etiket NaN olarak işaretlendi.")
        df = df.dropna(subset=["duygu"]).reset_index(drop=True)
    dagılım = df["duygu"].value_counts()
    print(f"  Etiket dağılımı:\n{dagılım.to_string()}")
    return df


# ---------------------------------------------------------------------------
# 8. TF-IDF Vektörizasyonu
# ---------------------------------------------------------------------------

def tfidf_vektorize(X_train, X_test, max_features: int = 50000, kaydet_yolu: str = None):
    """
    Eğitim verisine TF-IDF fit eder; eğitim ve test matrislerini döndürür.

    Parameters
    ----------
    X_train : Series
        Eğitim metinleri.
    X_test : Series
        Test metinleri.
    max_features : int
        Maksimum özellik sayısı.
    kaydet_yolu : str, optional
        Vektörizer dosya yolu.

    Returns
    -------
    tuple
        (X_train_tfidf, X_test_tfidf, vectorizer)
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        analyzer="word",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)
    print(f"  TF-IDF boyutu: {X_train_tfidf.shape}")

    if kaydet_yolu:
        os.makedirs(os.path.dirname(kaydet_yolu), exist_ok=True)
        joblib.dump(vectorizer, kaydet_yolu)
        print(f"  Vektörizer kaydedildi: {kaydet_yolu}")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ---------------------------------------------------------------------------
# Ana Pipeline
# ---------------------------------------------------------------------------

def on_isleme_pipeline(
    train_yolu: str,
    test_yolu: str,
    model_dizini: str = "models",
) -> dict:
    """
    Ham CSV çiftinden TF-IDF matrislerine kadar tüm ön işleme adımlarını çalıştırır.
    Veri zaten train/test olarak ayrılmış olduğu için kendi bölme yapmaz.

    Parameters
    ----------
    train_yolu : str
        train.csv dosyasının yolu.
    test_yolu : str
        test.csv dosyasının yolu.
    model_dizini : str
        Vektörizerin kaydedileceği dizin.

    Returns
    -------
    dict
        X_train, X_test, y_train, y_test, vectorizer, df_train, df_test
    """
    print("=" * 55)
    print("VERİ ÖN İŞLEME BAŞLADI")
    print("=" * 55)

    for yol, etiket in [(train_yolu, "Train"), (test_yolu, "Test")]:
        print(f"\n[{etiket}] CSV okunuyor: {yol}")
        if etiket == "Train":
            df_train = veri_oku(yol)
            df_train = bos_satir_sil(df_train)
            df_train = metinleri_temizle(df_train)
            df_train = etiket_donustur(df_train)
        else:
            df_test = veri_oku(yol)
            df_test  = bos_satir_sil(df_test)
            df_test  = metinleri_temizle(df_test)
            df_test  = etiket_donustur(df_test)

    print(f"\n  Train: {len(df_train):,}  |  Test: {len(df_test):,} satır")

    print("\nTF-IDF vektörizasyonu...")
    vectorizer_yolu = os.path.join(model_dizini, "tfidf_vectorizer.pkl")
    X_train_tfidf, X_test_tfidf, vectorizer = tfidf_vektorize(
        df_train["temiz_metin"], df_test["temiz_metin"],
        kaydet_yolu=vectorizer_yolu,
    )

    print("\n" + "=" * 55)
    print("ÖN İŞLEME TAMAMLANDI")
    print("=" * 55)

    return {
        "X_train": X_train_tfidf,
        "X_test":  X_test_tfidf,
        "y_train": df_train["duygu"],
        "y_test":  df_test["duygu"],
        "vectorizer": vectorizer,
        "df_train": df_train,
        "df_test":  df_test,
    }


# ---------------------------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ornek = [
        "Bu ürün gerçekten çok güzel, kesinlikle tavsiye ederim!",
        "Kargo çok geç geldi ve paket hasarlıydı. Berbat.",
        "Fiyatına göre fena değil.",
    ]
    print("=== Metin Temizleme Demo ===")
    for m in ornek:
        print(f"  Ham  : {m}")
        print(f"  Temiz: {metin_temizle(m)}")
        print()
