"""
Türkçe Müşteri Yorumları — Duygu & Konu Analizi Eğitim Scripti
===============================================================
Çalıştırma : python train.py
Gereksinimler:
    pip install pandas scikit-learn imbalanced-learn joblib nltk TurkishStemmer

Üretilen dosyalar (models/ klasörüne):
    sentiment_vectorizer.joblib
    sentiment_model.joblib
    topic_vectorizer.joblib
    topic_model.joblib
    label_info.joblib
"""

# ── Standart kütüphaneler ────────────────────────────────────────────────────
import os
import re
import sys
import logging
import inspect

# ── Üçüncü parti ─────────────────────────────────────────────────────────────
try:
    import joblib
except ImportError:
    try:
        from sklearn.externals import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required to run this script. Install it with `pip install joblib`"
        ) from exc

import numpy as np
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── NLTK Stopwords ───────────────────────────────────────────────────────────
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

OZEL_STOPWORD = {
    "bir", "bu", "ve", "ile", "de", "da", "mi", "mu", "mı", "mü",
    "için", "ne", "ki", "ya",
    "hem", "her", "biz", "siz", "onlar", "ben", "sen", "o",
    "bunu", "şunu", "bunun", "şunun", "gibi", "kadar", "bile",
    # KALDIRILDI — duygu kuvvetlendiriciler (app.py ile senkron):
    # "çok"  → "çok kötü", "çok güzel" ifadelerinde anlam taşır
    # "hiç"  → "hiç beğenmedim", "hiç iyi değil" gibi güçlü olumsuzlamalarda kritik
    # "daha" → "daha kötü", "daha iyi" karşılaştırmalarında gerekli
    # "en"   → "en berbat", "en güzel" üstünlük ifadelerinde gerekli
}
# Duygu kuvvetlendiriciler: NLTK listesinde de olsa kesinlikle korunmalı.
# "çok kötü", "hiç beğenmedim", "en berbat", "daha iyi" gibi ifadelerde
# bu kelimeler silinirse model sinyali zayıflar veya tersine döner.
DUYGU_KUVVETLENDIRICILER = {"çok", "hiç", "daha", "en", "ama", "fakat"}

try:
    STOPWORDS = (set(stopwords.words("turkish")) | OZEL_STOPWORD) - DUYGU_KUVVETLENDIRICILER
except OSError:
    STOPWORDS = OZEL_STOPWORD - DUYGU_KUVVETLENDIRICILER

# ── TurkishStemmer (opsiyonel) ────────────────────────────────────────────────
try:
    from TurkishStemmer import TurkishStemmer as _TS
    _stemmer = _TS()
    USE_STEMMER = True
    log.info("TurkishStemmer yüklendi — kök bulma aktif.")
except ImportError:
    USE_STEMMER = False
    log.warning("TurkishStemmer bulunamadı. Kök bulma devre dışı.")
    log.warning("Kurmak için: pip install TurkishStemmer")

# ════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ════════════════════════════════════════════════════════════════════════════
PROJE_KOKU   = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV    = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "train.csv")
TEST_CSV     = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "test.csv")
MODEL_DIR    = os.path.join(PROJE_KOKU, "models")

# ──────────────────────────────────────────────────────────────────────────────
# YILDIZ → DUYGU EŞLEMESİ (Ham Hepsiburada verisi için)
#   1–2 yıldız → olumsuz
#   3   yıldız → nötr
#   4–5 yıldız → olumlu
# ──────────────────────────────────────────────────────────────────────────────
YILDIZ_DUYGU = {
    1: "olumsuz",
    2: "olumsuz",
    3: "nötr",
    4: "olumlu",
    5: "olumlu",
}

# Mevcut etiketleri normalize et (Positive / Notr / Negative → Türkçe)
ETIKET_NORM = {
    "positive": "olumlu",
    "Positive": "olumlu",
    "POSITIVE": "olumlu",
    "negative": "olumsuz",
    "Negative": "olumsuz",
    "NEGATIVE": "olumsuz",
    "notr":     "nötr",
    "Notr":     "nötr",
    "NOTR":     "nötr",
    "nötr":     "nötr",
    "olumlu":   "olumlu",
    "olumsuz":  "olumsuz",
}

# ──────────────────────────────────────────────────────────────────────────────
# KONU ANAHTAR KELİMELERİ (otomatik etiketleme için)
# ──────────────────────────────────────────────────────────────────────────────
KONU_ANAHTARLARI = {
    "teslimat": [
        "kargo", "teslimat", "teslim", "gecikme", "kurye",
        "gönderim", "paket", "hızlı", "yavaş", "geç", "gönderi",
        "ertesi", "kargolama", "teslimaat",
    ],
    "fiyat": [
        "fiyat", "ücret", "pahalı", "ucuz", "para", "değer",
        "kampanya", "indirim", "ekonomik", "hesaplı", "bütçe",
        "piyasa", "değmez", "değiyor", "tutarında",
    ],
    "urun": [
        "ürün", "kalite", "malzeme", "bozuk", "kusurlu", "sağlam",
        "dayanıklı", "renk", "boyut", "beden", "model", "görünüm",
        "kullanım", "performans", "özellik", "işlevsel", "orijinal",
    ],
    "hizmet": [
        "müşteri", "hizmet", "destek", "yardım", "çözüm", "iade",
        "satıcı", "iletişim", "ilgili", "yardımcı", "servis",
        "personel", "garson", "çalışan", "teknik",
    ],
    "hijyen": [
        "hijyen", "temiz", "kirli", "steril", "ambalaj", "sağlık",
        "bakteri", "dezenfektan", "pis", "bakımsız", "sağlıksız",
    ],
}


# ════════════════════════════════════════════════════════════════════════════
# METİN TEMİZLEME
# ════════════════════════════════════════════════════════════════════════════

def metin_temizle(metin: str) -> str:
    """
    Ham Türkçe metni adım adım temizler:
        1. Küçük harfe çevir
        2. URL & e-posta sil
        3. Sayıları sil
        4. Noktalama / özel karakterleri sil (Türkçe harfler korunur)
        5. Fazla boşlukları temizle
        6. Stopword çıkarımı (NLTK Türkçe + özel liste)
        7. TurkishStemmer ile kök bulma (varsa)
    """
    if not isinstance(metin, str):
        return ""

    metin = metin.lower()
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)          # URL
    metin = re.sub(r"\S+@\S+\.\S+", " ", metin)              # e-posta
    metin = re.sub(r"\d+", " ", metin)                        # sayı
    metin = re.sub(r"[^\w\sçğışöüÇĞİŞÖÜ]", " ", metin)      # noktalama
    metin = re.sub(r"\s+", " ", metin).strip()

    kelimeler = [
        k for k in metin.split()
        if k not in STOPWORDS and len(k) > 1
    ]

    if USE_STEMMER:
        try:
            kelimeler = [_stemmer.stem(k) for k in kelimeler]
        except Exception:
            pass  # hata durumunda köksüz devam et

    return " ".join(kelimeler)


# ════════════════════════════════════════════════════════════════════════════
# VERİ OKUMA
# ════════════════════════════════════════════════════════════════════════════

def veri_oku(train_yolu: str, test_yolu: str) -> pd.DataFrame:
    """
    train.csv + test.csv okur; 'text' ve 'label' sütunlarını kullanır.
    Eğer veride 'rating' / 'puan' sütunu varsa, yıldız → duygu eşlemesi uygulanır.
    Her iki kaynak da tek bir DataFrame olarak döndürülür.
    """
    parcalar = []
    for yol, etiket in [(train_yolu, "train"), (test_yolu, "test")]:
        log.info(f"Okunuyor: {yol}")
        for enc in ("utf-8", "utf-8-sig", "latin-1", "iso-8859-9"):
            try:
                df = pd.read_csv(yol, encoding=enc)
                log.info(f"  {etiket}: {len(df):,} satır ({enc})")
                break
            except (UnicodeDecodeError, FileNotFoundError):
                df = None
        if df is None:
            log.error(f"  Dosya okunamadı: {yol}")
            continue

        df.columns = df.columns.str.lower().str.strip()

        # ── Metin sütununu bul ──────────────────────────────────────────────
        metin_adaylar = ["text", "reviewtext", "yorum", "review", "comment", "metin", "içerik"]
        metin_col = next((c for c in metin_adaylar if c in df.columns), df.columns[0])

        # ── Etiket sütununu bul ────────────────────────────────────────────
        # Önce hazır etiket (label/duygu), yoksa yıldız dönüşümü
        etiket_col = None
        yildiz_col = None

        if "label" in df.columns:
            etiket_col = "label"
        elif "duygu" in df.columns:
            etiket_col = "duygu"
        else:
            yildiz_adaylar = ["rating", "puan", "star", "yildiz", "stars", "overall"]
            yildiz_col = next((c for c in yildiz_adaylar if c in df.columns), None)

        df = df[[metin_col] + ([etiket_col] if etiket_col else [yildiz_col])].copy()
        df.columns = ["text", "raw_label"]
        df = df.dropna(subset=["text", "raw_label"])
        df["text"] = df["text"].astype(str).str.strip()
        df = df[df["text"] != ""]

        # ── Duygu etiketi oluştur ──────────────────────────────────────────
        if yildiz_col:
            # Ham yıldız verisi: 1-2 → olumsuz / 3 → nötr / 4-5 → olumlu
            df["raw_label"] = pd.to_numeric(df["raw_label"], errors="coerce")
            df = df.dropna(subset=["raw_label"])
            df["raw_label"] = df["raw_label"].astype(int).clip(1, 5)
            df["duygu"] = df["raw_label"].map(YILDIZ_DUYGU)
            log.info(
                f"  Yıldız → Duygu eşlemesi uygulandı "
                f"(1-2=olumsuz, 3=nötr, 4-5=olumlu)"
            )
        else:
            # Etiket zaten mevcut: normalize et
            df["duygu"] = df["raw_label"].map(ETIKET_NORM)
            gecersiz = df["duygu"].isna().sum()
            if gecersiz:
                log.warning(f"  {gecersiz} satır tanımsız etiketle silindi.")
                df = df.dropna(subset=["duygu"])

        parcalar.append(df[["text", "duygu"]])

    if not parcalar:
        raise RuntimeError("Hiç veri yüklenemedi. CSV yollarını kontrol edin.")

    tam = pd.concat(parcalar, ignore_index=True)
    log.info(f"\nToplam satır: {len(tam):,}")
    log.info(f"Duygu dağılımı:\n{tam['duygu'].value_counts().to_string()}")
    return tam


# ════════════════════════════════════════════════════════════════════════════
# OTOMATİK KONU ETİKETLEME
# ════════════════════════════════════════════════════════════════════════════

def konu_ata(metin: str) -> str:
    """Anahtar kelime skorlamasıyla en uygun konuyu seçer. Eşitsizlikte 'urun'."""
    metin_lower = metin.lower()
    skorlar = {
        konu: sum(1 for kw in kwler if kw in metin_lower)
        for konu, kwler in KONU_ANAHTARLARI.items()
    }
    en_iyi = max(skorlar, key=skorlar.get)
    return en_iyi if skorlar[en_iyi] > 0 else "urun"


# ════════════════════════════════════════════════════════════════════════════
# MODEL EĞİTİMİ (TF-IDF + SMOTE + Lojistik Regresyon)
# ════════════════════════════════════════════════════════════════════════════

def model_egit(
    X_train: list,
    y_train: np.ndarray,
    model_adi: str,
) -> tuple:
    """
    Tek bir hedef için TF-IDF + SMOTE + Lojistik Regresyon pipeline'ı.

    Parametreler
    ------------
    X_train  : Temizlenmiş eğitim metinleri (liste)
    y_train  : Etiketler (numpy dizisi)
    model_adi: Loglama için isim

    Döndürür
    --------
    (vectorizer, classifier)
    """
    log.info(f"\n{'=' * 55}")
    log.info(f"[{model_adi.upper()}] EĞİTİM BAŞLIYOR")
    log.info(f"{'=' * 55}")

    # 1. TF-IDF Vektörizasyon ────────────────────────────────────────────────
    log.info("TF-IDF vektörizasyonu...")
    vektorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),      # unigram + bigram
        sublinear_tf=True,       # log(tf+1) — sık kelimeleri bastırır
        min_df=2,                # en az 2 belgede geç
        max_df=0.95,             # çok yaygın kelimeleri ele
        analyzer="word",
    )
    X_tfidf = vektorizer.fit_transform(X_train)
    log.info(f"TF-IDF boyutu: {X_tfidf.shape}")

    # 2. SMOTE — Sınıf Dengesizliğini Gider ──────────────────────────────────
    #    "Sürekli nötr verme" hatasının TEMELİ budur.
    #    SMOTE, azınlık sınıflar için sentetik örnekler üretir.
    log.info("SMOTE uygulanıyor (azınlık sınıflar dengeleniyor)...")
    log.info(f"SMOTE öncesi: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    smote = SMOTE(random_state=42, k_neighbors=5)
    try:
        # imbalanced-learn >= 0.8: sparse matris desteği
        X_dengeli, y_dengeli = smote.fit_resample(X_tfidf, y_train)
    except TypeError:
        # Eski sürüm: dense'e çevirerek dene
        log.warning("Sparse SMOTE başarısız, dense'e dönüştürülüyor...")
        X_dengeli, y_dengeli = smote.fit_resample(X_tfidf.toarray(), y_train)

    log.info(f"SMOTE sonrası: {dict(zip(*np.unique(y_dengeli, return_counts=True)))}")
    log.info(f"SMOTE sonrası boyut: {X_dengeli.shape}")

    # 3. Lojistik Regresyon ──────────────────────────────────────────────────
    log.info("Lojistik Regresyon eğitiliyor...")
    clf = LogisticRegression(
        C=1.0,
        max_iter=2_000,
        solver="saga",           # büyük veri setleri için ideal
        random_state=42,
    )
    clf.fit(X_dengeli, y_dengeli)
    log.info(f"Eğitim tamamlandı. Sınıflar: {clf.classes_}")

    return vektorizer, clf


# ════════════════════════════════════════════════════════════════════════════
# DEĞERLENDİRME
# ════════════════════════════════════════════════════════════════════════════

def modeli_degerlendir(
    vektorizer: TfidfVectorizer,
    clf: LogisticRegression,
    X_test: list,
    y_test: np.ndarray,
    model_adi: str,
) -> None:
    X_tfidf = vektorizer.transform(X_test)
    y_pred  = clf.predict(X_tfidf)
    log.info(f"\n[{model_adi.upper()}] Sınıflandırma Raporu:")
    log.info(f"\n{classification_report(y_test, y_pred, zero_division=0)}")


# ════════════════════════════════════════════════════════════════════════════
# ANA AKIŞ
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── 1. Veri Yükle ────────────────────────────────────────────────────────
    df = veri_oku(TRAIN_CSV, TEST_CSV)

    # ── 2. Metin Temizleme ───────────────────────────────────────────────────
    log.info("\nMetinler temizleniyor...")
    df["temiz"] = df["text"].apply(metin_temizle)
    df = df[df["temiz"].str.len() > 3].reset_index(drop=True)
    log.info(f"Temizleme sonrası: {len(df):,} geçerli satır")

    # ── 3. Konu Otomatik Etiketle ────────────────────────────────────────────
    log.info("\nKonu etiketleri atanıyor...")
    df["konu"] = df["text"].apply(konu_ata)
    log.info(f"Konu dağılımı:\n{df['konu'].value_counts().to_string()}")

    # ── 4. Train / Test Bölme ─────────────────────────────────────────────────
    X = df["temiz"].values
    y_duygu = df["duygu"].values
    y_konu  = df["konu"].values

    X_train, X_test, yd_train, yd_test, yk_train, yk_test = train_test_split(
        X, y_duygu, y_konu,
        test_size=0.20,
        random_state=42,
        stratify=y_duygu,        # sınıf oranlarını koru
    )
    log.info(f"\nEğitim: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 5. Duygu Modeli ───────────────────────────────────────────────────────
    duygu_vec, duygu_clf = model_egit(X_train.tolist(), yd_train, "Duygu")
    modeli_degerlendir(duygu_vec, duygu_clf, X_test.tolist(), yd_test, "Duygu")

    # ── 6. Konu Modeli ────────────────────────────────────────────────────────
    konu_vec, konu_clf = model_egit(X_train.tolist(), yk_train, "Konu")
    modeli_degerlendir(konu_vec, konu_clf, X_test.tolist(), yk_test, "Konu")

    # ── 7. Modelleri Kaydet ───────────────────────────────────────────────────
    log.info("\nModeller kaydediliyor...")
    kayitlar = {
        "sentiment_vectorizer.joblib": duygu_vec,
        "sentiment_model.joblib":      duygu_clf,
        "topic_vectorizer.joblib":     konu_vec,
        "topic_model.joblib":          konu_clf,
    }
    for dosya, nesne in kayitlar.items():
        yol = os.path.join(MODEL_DIR, dosya)
        joblib.dump(nesne, yol)
        log.info(f"  Kaydedildi: {yol}")

    # Etiket bilgisi (API için)
    joblib.dump(
        {
            "duygu_siniflari": duygu_clf.classes_.tolist(),
            "konu_siniflari":  konu_clf.classes_.tolist(),
        },
        os.path.join(MODEL_DIR, "label_info.joblib"),
    )

    log.info("\n" + "=" * 55)
    log.info("EĞİTİM BAŞARIYLA TAMAMLANDI")
    log.info(f"Duygu sınıfları : {duygu_clf.classes_}")
    log.info(f"Konu sınıfları  : {konu_clf.classes_}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
