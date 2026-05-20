"""
Model eğitim modülü.
Ön işlemden gelen TF-IDF sparse matrislerini alır;
Naive Bayes ve Lojistik Regresyon modellerini eğitip joblib ile kaydeder.
"""

import os
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score


# ---------------------------------------------------------------------------
# Model Tanımları (TF-IDF preprocess'te uygulandığından sadece sınıflandırıcı)
# ---------------------------------------------------------------------------

def naive_bayes_modeli() -> MultinomialNB:
    """
    Multinomial Naive Bayes modelini döndürür.
    Metin sınıflandırmasında güçlü baseline; çok hızlı ve az bellekli.

    Returns
    -------
    MultinomialNB
        Eğitime hazır model.
    """
    return MultinomialNB(alpha=0.1)   # alpha=0.1: Laplace yumuşatma


def lojistik_regresyon_modeli() -> LogisticRegression:
    """
    Lojistik Regresyon modelini döndürür.
    TF-IDF ile birlikte genellikle Naive Bayes'ten daha yüksek F1 verir.

    Returns
    -------
    LogisticRegression
        Eğitime hazır model.
    """
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="saga",           # Büyük veri için saga daha hızlı
        class_weight="balanced", # Dengesiz sınıflar için otomatik ağırlık
        random_state=42,
    )


# ---------------------------------------------------------------------------
# Eğitim & Kayıt
# ---------------------------------------------------------------------------

def modeli_egit(model, X_train, y_train):
    """
    Verilen modeli eğitim verisi üzerinde eğitir.

    Parameters
    ----------
    model : sklearn estimator
        Eğitilecek model.
    X_train : sparse matrix
        TF-IDF eğitim matrisi.
    y_train : array-like
        Eğitim etiketleri.

    Returns
    -------
    Eğitilmiş model.
    """
    model.fit(X_train, y_train)
    return model


def modeli_kaydet(model, model_adi: str, dizin: str = "models") -> str:
    """
    Eğitilmiş modeli joblib ile diske kaydeder.

    Parameters
    ----------
    model : sklearn estimator
        Kaydedilecek model.
    model_adi : str
        Dosya adı (uzantısız).
    dizin : str
        Hedef dizin.

    Returns
    -------
    str
        Kaydedilen dosyanın tam yolu.
    """
    os.makedirs(dizin, exist_ok=True)
    dosya_yolu = os.path.join(dizin, f"{model_adi}.pkl")
    joblib.dump(model, dosya_yolu)
    print(f"  Model kaydedildi: {dosya_yolu}")
    return dosya_yolu


# ---------------------------------------------------------------------------
# Hızlı Doğrulama
# ---------------------------------------------------------------------------

def egitim_skorlari(model, X_train, y_train, X_test, y_test) -> dict:
    """
    Eğitim ve test kümesinde Accuracy ile Makro F1 hesaplar.

    Parameters
    ----------
    model : eğitilmiş model
    X_train, y_train : eğitim verisi
    X_test, y_test   : test verisi

    Returns
    -------
    dict
        train_acc, test_acc, train_f1, test_f1 anahtarlarını içerir.
    """
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    return {
        "train_acc": accuracy_score(y_train, y_pred_train),
        "test_acc":  accuracy_score(y_test,  y_pred_test),
        "train_f1":  f1_score(y_train, y_pred_train, average="macro"),
        "test_f1":   f1_score(y_test,  y_pred_test,  average="macro"),
    }


# ---------------------------------------------------------------------------
# Ana Eğitim Akışı
# ---------------------------------------------------------------------------

def egitim_pipeline(X_train, y_train, X_test, y_test, model_dizini: str = "models") -> dict:
    """
    Tüm modelleri sırayla eğitir, kaydeder ve özet sonuçları döndürür.

    Parameters
    ----------
    X_train : sparse matrix
        TF-IDF eğitim matrisi.
    y_train : array-like
        Eğitim etiketleri.
    X_test : sparse matrix
        TF-IDF test matrisi.
    y_test : array-like
        Test etiketleri.
    model_dizini : str
        Modellerin kaydedileceği dizin.

    Returns
    -------
    dict
        Her modelin model nesnesi ve skor bilgilerini içerir.
    """
    print("=" * 55)
    print("MODEL EĞİTİMİ BAŞLADI")
    print("=" * 55)

    modeller = {
        "naive_bayes":       naive_bayes_modeli(),
        "lojistik_regresyon": lojistik_regresyon_modeli(),
    }

    sonuclar = {}

    for isim, model in modeller.items():
        print(f"\n[{isim.upper()}] Eğitim başlıyor...")
        model  = modeli_egit(model, X_train, y_train)
        yol    = modeli_kaydet(model, isim, model_dizini)
        skorlar = egitim_skorlari(model, X_train, y_train, X_test, y_test)

        print(
            f"  Train Acc={skorlar['train_acc']:.4f} | "
            f"Test  Acc={skorlar['test_acc']:.4f} | "
            f"Makro F1={skorlar['test_f1']:.4f}"
        )
        hedef = "HEDEF KARŞILANDI ✓" if skorlar["test_f1"] >= 0.80 else "Hedef altında"
        print(f"  Makro F1 >= 0.80 → {hedef}")

        sonuclar[isim] = {"model": model, "dosya_yolu": yol, "skorlar": skorlar}

    print("\n" + "=" * 55)
    print("EĞİTİM TAMAMLANDI")
    print("=" * 55)
    return sonuclar


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import on_isleme_pipeline

    PROJE_KOKU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    TRAIN_CSV  = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "train.csv")
    TEST_CSV   = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "test.csv")
    MODEL_DIR  = os.path.join(PROJE_KOKU, "models")

    print("=== Veri ön işleniyor... ===")
    veri = on_isleme_pipeline(TRAIN_CSV, TEST_CSV, model_dizini=MODEL_DIR)

    print("\n=== Modeller eğitiliyor... ===")
    egitim_pipeline(
        veri["X_train"], veri["y_train"],
        veri["X_test"],  veri["y_test"],
        model_dizini=MODEL_DIR,
    )
