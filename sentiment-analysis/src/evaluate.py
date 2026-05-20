"""
Model değerlendirme modülü.
Her model için Accuracy, Makro F1, Classification Report ve
Confusion Matrix üretir; modeller arası karşılaştırma tablosu oluşturur.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Sunucu ortamında ekran gerektirmeyen backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

HEDEF_F1 = 0.80   # Proje hedef Makro F1 skoru
SINIFLAR = ["olumsuz", "notr", "olumlu"]


# ---------------------------------------------------------------------------
# Tekil Model Değerlendirme
# ---------------------------------------------------------------------------

def model_yukle(dosya_yolu: str):
    """
    Diskten joblib ile kaydedilmiş bir modeli yükler.

    Parameters
    ----------
    dosya_yolu : str
        .pkl dosyasının tam yolu.

    Returns
    -------
    sklearn Pipeline
        Yüklenmiş model pipeline'ı.
    """
    try:
        return joblib.load(dosya_yolu)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model dosyası bulunamadı: {dosya_yolu}")


def metrikleri_hesapla(y_gercek, y_tahmin) -> dict:
    """
    Verilen gerçek ve tahmin etiketleri için temel metrikleri hesaplar.

    Parameters
    ----------
    y_gercek : array-like
        Gerçek sınıf etiketleri.
    y_tahmin : array-like
        Model tahminleri.

    Returns
    -------
    dict
        accuracy, makro_f1, rapor anahtarlarını içerir.
    """
    return {
        "accuracy": accuracy_score(y_gercek, y_tahmin),
        "makro_f1": f1_score(y_gercek, y_tahmin, average="macro"),
        "rapor": classification_report(y_gercek, y_tahmin, target_names=SINIFLAR, zero_division=0),
    }


def metrikleri_yazdir(model_adi: str, metrikler: dict) -> None:
    """
    Hesaplanan metrikleri formatlanmış şekilde konsola yazdırır.

    Parameters
    ----------
    model_adi : str
        Modelin okunabilir adı.
    metrikler : dict
        metrikleri_hesapla() çıktısı.
    """
    ayrac = "=" * 55
    hedef = "HEDEF KARŞILANDI ✓" if metrikler["makro_f1"] >= HEDEF_F1 else f"Hedef altında (≥ {HEDEF_F1})"

    print(f"\n{ayrac}")
    print(f"  MODEL: {model_adi.upper()}")
    print(ayrac)
    print(f"  Accuracy   : {metrikler['accuracy']:.4f}")
    print(f"  Makro F1   : {metrikler['makro_f1']:.4f}  ← {hedef}")
    print(f"\n  Sınıf Bazlı Rapor:")
    for satir in metrikler["rapor"].splitlines():
        print(f"    {satir}")


# ---------------------------------------------------------------------------
# Confusion Matrix Görselleştirme
# ---------------------------------------------------------------------------

def confusion_matrix_kaydet(
    y_gercek,
    y_tahmin,
    model_adi: str,
    kaydet_dizini: str = "figures",
) -> str:
    """
    Confusion matrix'i seaborn heatmap olarak PNG dosyasına kaydeder.

    Parameters
    ----------
    y_gercek : array-like
        Gerçek etiketler.
    y_tahmin : array-like
        Tahmin edilen etiketler.
    model_adi : str
        Başlıkta ve dosya adında kullanılacak model adı.
    kaydet_dizini : str
        PNG dosyasının kaydedileceği dizin.

    Returns
    -------
    str
        Kaydedilen dosyanın yolu.
    """
    os.makedirs(kaydet_dizini, exist_ok=True)
    dosya_yolu = os.path.join(kaydet_dizini, f"confusion_matrix_{model_adi}.png")

    cm = confusion_matrix(y_gercek, y_tahmin, labels=SINIFLAR)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=SINIFLAR,
        yticklabels=SINIFLAR,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Tahmin Edilen", fontsize=12)
    ax.set_ylabel("Gerçek", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_adi}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(dosya_yolu, dpi=150)
    plt.close(fig)

    print(f"  Confusion matrix kaydedildi: {dosya_yolu}")
    return dosya_yolu


# ---------------------------------------------------------------------------
# Modeller Arası Karşılaştırma Tablosu
# ---------------------------------------------------------------------------

def karsilastirma_tablosu(sonuclar: dict) -> pd.DataFrame:
    """
    Tüm modeller için Accuracy ve Makro F1 skorlarını tek tabloda gösterir.

    Parameters
    ----------
    sonuclar : dict
        Her model için {"accuracy", "makro_f1"} içeren sözlük.
        Örn.: {"naive_bayes": {"accuracy": 0.85, "makro_f1": 0.83}, ...}

    Returns
    -------
    pd.DataFrame
        Karşılaştırma tablosu; Makro F1'e göre azalan sırada sıralı.
    """
    satirlar = []
    for isim, m in sonuclar.items():
        hedef = "Evet" if m["makro_f1"] >= HEDEF_F1 else "Hayır"
        satirlar.append({
            "Model": isim,
            "Accuracy": round(m["accuracy"], 4),
            "Makro F1 (Ana Metrik)": round(m["makro_f1"], 4),
            f"F1 ≥ {HEDEF_F1}": hedef,
        })

    tablo = pd.DataFrame(satirlar).sort_values("Makro F1 (Ana Metrik)", ascending=False)
    tablo = tablo.reset_index(drop=True)

    print("\n" + "=" * 55)
    print("  MODELLER ARASI KARŞILAŞTIRMA")
    print("=" * 55)
    print(tablo.to_string(index=False))
    print("=" * 55)
    return tablo


def karsilastirma_grafigi_kaydet(
    sonuclar: dict,
    kaydet_dizini: str = "figures",
) -> str:
    """
    Modellerin Accuracy ve Makro F1 skorlarını grouped bar chart olarak kaydeder.

    Parameters
    ----------
    sonuclar : dict
        Her model için {"accuracy", "makro_f1"} içeren sözlük.
    kaydet_dizini : str
        PNG dosyasının kaydedileceği dizin.

    Returns
    -------
    str
        Kaydedilen dosyanın yolu.
    """
    os.makedirs(kaydet_dizini, exist_ok=True)
    dosya_yolu = os.path.join(kaydet_dizini, "model_karsilastirma.png")

    modeller = list(sonuclar.keys())
    accuracy_skorlari = [sonuclar[m]["accuracy"] for m in modeller]
    f1_skorlari = [sonuclar[m]["makro_f1"] for m in modeller]

    x = np.arange(len(modeller))
    genislik = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    cubuk1 = ax.bar(x - genislik / 2, accuracy_skorlari, genislik, label="Accuracy", color="#4C72B0")
    cubuk2 = ax.bar(x + genislik / 2, f1_skorlari, genislik, label="Makro F1", color="#DD8452")

    # Hedef çizgisi
    ax.axhline(y=HEDEF_F1, color="red", linestyle="--", linewidth=1.5, label=f"Hedef F1={HEDEF_F1}")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in modeller], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor", fontsize=12)
    ax.set_title("Model Karşılaştırması", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.bar_label(cubuk1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(cubuk2, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(dosya_yolu, dpi=150)
    plt.close(fig)

    print(f"  Karşılaştırma grafiği kaydedildi: {dosya_yolu}")
    return dosya_yolu


# ---------------------------------------------------------------------------
# Ana Değerlendirme Akışı
# ---------------------------------------------------------------------------

def degerlendirme_pipeline(
    X_test,
    y_test,
    model_dizini: str = "models",
    figure_dizini: str = "figures",
) -> dict:
    """
    models/ dizinindeki tüm .pkl modellerini yükler, değerlendirir ve
    görsel çıktıları figures/ dizinine kaydeder.

    Parameters
    ----------
    X_test : array-like
        Test özellikleri (ham metin Series).
    y_test : array-like
        Gerçek test etiketleri.
    model_dizini : str
        Modellerin bulunduğu dizin.
    figure_dizini : str
        Grafiklerin kaydedileceği dizin.

    Returns
    -------
    dict
        Her model için metrik sonuçlarını içeren sözlük.
    """
    print("=" * 55)
    print("MODEL DEĞERLENDİRME BAŞLADI")
    print("=" * 55)

    if not os.path.isdir(model_dizini):
        raise FileNotFoundError(f"Model dizini bulunamadı: {model_dizini}")

    pkl_dosyalari = [f for f in os.listdir(model_dizini) if f.endswith(".pkl")
                     and "vectorizer" not in f]

    if not pkl_dosyalari:
        raise FileNotFoundError(f"'{model_dizini}' dizininde eğitilmiş model bulunamadı.")

    tum_sonuclar = {}

    for dosya in sorted(pkl_dosyalari):
        model_adi = dosya.replace(".pkl", "")
        dosya_yolu = os.path.join(model_dizini, dosya)

        print(f"\n  Yükleniyor: {dosya_yolu}")
        pipeline = model_yukle(dosya_yolu)

        y_tahmin = pipeline.predict(X_test)
        metrikler = metrikleri_hesapla(y_test, y_tahmin)
        metrikleri_yazdir(model_adi, metrikler)
        confusion_matrix_kaydet(y_test, y_tahmin, model_adi, figure_dizini)

        tum_sonuclar[model_adi] = {
            "accuracy": metrikler["accuracy"],
            "makro_f1": metrikler["makro_f1"],
        }

    # Karşılaştırma
    karsilastirma_tablosu(tum_sonuclar)
    karsilastirma_grafigi_kaydet(tum_sonuclar, figure_dizini)

    print("\nDeğerlendirme tamamlandı.")
    return tum_sonuclar


# ---------------------------------------------------------------------------
# Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from preprocess import on_isleme_pipeline

    PROJE_KOKU   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    TRAIN_CSV    = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "train.csv")
    TEST_CSV     = os.path.join(PROJE_KOKU, "data", "yorumlar.csv", "test.csv")
    MODEL_DIR    = os.path.join(PROJE_KOKU, "models")
    FIGURE_DIR   = os.path.join(PROJE_KOKU, "figures")

    print("=== Veri yükleniyor (TF-IDF tekrar uygulanıyor)... ===")
    veri = on_isleme_pipeline(TRAIN_CSV, TEST_CSV, model_dizini=MODEL_DIR)

    print("\n=== Değerlendirme başlıyor... ===")
    degerlendirme_pipeline(
        veri["X_test"],
        veri["y_test"],
        model_dizini=MODEL_DIR,
        figure_dizini=FIGURE_DIR,
    )
