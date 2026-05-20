"""
FastAPI tabanlı duygu analizi ve konu sınıflandırma API'si.
POST /api/analyze — Türkçe yorumu alır, duygu + konu + güven skoru döndürür.
"""

import os
import sys
import joblib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

API_DIZINI = os.path.dirname(os.path.abspath(__file__))
PROJE_KOKU = os.path.abspath(os.path.join(API_DIZINI, ".."))

# src/ dizinini Python yoluna ekle (API src modüllerini import edebilsin)
SRC_DIZINI = os.path.join(PROJE_KOKU, "src")
if SRC_DIZINI not in sys.path:
    sys.path.insert(0, SRC_DIZINI)

try:
    from preprocess import metin_temizle
except ImportError:
    # Modül bulunamadıysa basit bir fallback tanımla
    def metin_temizle(metin: str) -> str:
        return str(metin).lower().strip()

MODEL_YOLU = os.path.join(PROJE_KOKU, "models", "lojistik_regresyon.pkl")
VECTORIZER_YOLU = os.path.join(PROJE_KOKU, "models", "tfidf_vectorizer.pkl")

# ---------------------------------------------------------------------------
# Uygulama Durumu (başlatma / kapatma)
# ---------------------------------------------------------------------------

app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başladığında modeli yükler; kapanışta temizler."""
    print("[API] Model ve vektörizer yükleniyor...")
    try:
        app_state["model"] = joblib.load(MODEL_YOLU)
        app_state["vectorizer"] = joblib.load(VECTORIZER_YOLU)
        print("[API] Model ve vektörizer başarıyla yüklendi.")
    except FileNotFoundError as e:
        print(f"[API UYARI] Dosya bulunamadı: {e}")
        print("[API] Tahminler çalışmayacak. Önce 'python sentiment-analysis/src/train.py' çalıştırın.")
        app_state["model"] = None
        app_state["vectorizer"] = None
    yield
    app_state.clear()


# ---------------------------------------------------------------------------
# FastAPI Uygulaması
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Türkçe Duygu Analizi API",
    description="Türkçe müşteri yorumlarında duygu ve konu sınıflandırması",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — web platformuyla haberleşmek için tüm kaynaklara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Konu Sınıflandırması — Anahtar Kelime Tabanlı
# ---------------------------------------------------------------------------

KONU_ANAHTARLARI = {
    "hizmet": ["servis", "garson", "personel", "çalışan", "görevli", "staff"],
    "fiyat": ["pahalı", "ucuz", "fiyat", "ücret", "para", "maliyet", "değmez", "değer"],
    "urun": ["yemek", "ürün", "kalite", "lezzet", "tat", "malzeme", "içerik"],
    "hijyen": ["temiz", "kirli", "hijyen", "pis", "steril", "bakımlı"],
    "teslimat": ["kargo", "teslimat", "kurye", "geç", "hızlı", "paket", "gönderim"],
}


def konu_siniflandir(metin: str) -> str:
    """
    Anahtar kelime eşleştirmesiyle konuyu belirler.

    Parameters
    ----------
    metin : str
        Temizlenmiş yorum metni.

    Returns
    -------
    str
        Tespit edilen konu etiketi. Hiç eşleşme yoksa 'urun' döner.
    """
    metin_lower = metin.lower()
    puan_tablosu = {konu: 0 for konu in KONU_ANAHTARLARI}

    for konu, kelimeler in KONU_ANAHTARLARI.items():
        for kelime in kelimeler:
            if kelime in metin_lower:
                puan_tablosu[konu] += 1

    en_yuksek = max(puan_tablosu, key=puan_tablosu.get)
    # Hiç eşleşme yoksa varsayılan konu
    return en_yuksek if puan_tablosu[en_yuksek] > 0 else "urun"


# ---------------------------------------------------------------------------
# Özet Üretme
# ---------------------------------------------------------------------------

OZET_SABLONLARI = {
    ("olumlu", "hizmet"): "Müşteri hizmet kalitesinden memnun kalmış.",
    ("olumlu", "fiyat"): "Fiyat-performans dengesi olumlu bulunmuş.",
    ("olumlu", "urun"): "Ürün/yemek kalitesi beğenilmiş.",
    ("olumlu", "hijyen"): "Hijyen koşulları olumlu değerlendirilmiş.",
    ("olumlu", "teslimat"): "Teslimat süreci ve hızı memnuniyet verici.",
    ("olumsuz", "hizmet"): "Hizmet kalitesinden şikayet var.",
    ("olumsuz", "fiyat"): "Fiyat yüksek ya da değer sağlanamamış.",
    ("olumsuz", "urun"): "Ürün kalitesi beklentilerin altında.",
    ("olumsuz", "hijyen"): "Hijyen sorunları tespit edilmiş.",
    ("olumsuz", "teslimat"): "Teslimat süreci sorunlu veya gecikmiş.",
    ("notr", "hizmet"): "Hizmet ortalama düzeyde değerlendirilmiş.",
    ("notr", "fiyat"): "Fiyat makul karşılanmış.",
    ("notr", "urun"): "Ürün/yemek kalitesi orta düzeyde.",
    ("notr", "hijyen"): "Hijyen açısından belirgin bir görüş yok.",
    ("notr", "teslimat"): "Teslimat süreci kabul edilebilir.",
}


def ozet_uret(duygu: str, konu: str) -> str:
    """
    Duygu ve konu etiketine göre kısa açıklama üretir.

    Parameters
    ----------
    duygu : str
        Tahmin edilen duygu etiketi.
    konu : str
        Tahmin edilen konu etiketi.

    Returns
    -------
    str
        İnsan okunabilir özet cümle.
    """
    return OZET_SABLONLARI.get((duygu, konu), "Yorum analiz edildi.")


# ---------------------------------------------------------------------------
# Pydantic Şemaları
# ---------------------------------------------------------------------------

class AnalizIstegi(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, description="Türkçe yorum metni")


class AnalizYaniti(BaseModel):
    duygu: str = Field(..., description="olumlu | olumsuz | notr")
    konu: str = Field(..., description="hizmet | fiyat | urun | hijyen | teslimat")
    guven_skoru: float = Field(..., ge=0.0, le=1.0)
    ozet: str


# ---------------------------------------------------------------------------
# Endpointler
# ---------------------------------------------------------------------------

@app.get("/", summary="API sağlık kontrolü")
async def anasayfa():
    """API'nin çalışıp çalışmadığını kontrol eder."""
    return {
        "durum": "çalışıyor",
        "model_yuklendi": app_state.get("model") is not None,
        "versiyon": "1.0.0",
    }


@app.post("/api/analyze", response_model=AnalizYaniti, summary="Duygu ve konu analizi")
async def analiz_et(istek: AnalizIstegi):
    """
    Türkçe müşteri yorumunu analiz eder.

    - **duygu**: olumlu / olumsuz / notr
    - **konu**: hizmet / fiyat / urun / hijyen / teslimat
    - **guven_skoru**: 0.0 – 1.0 arası model güven değeri
    - **ozet**: kısa açıklama cümlesi
    """
    model      = app_state.get("model")
    vectorizer = app_state.get("vectorizer")
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model henüz yüklenmedi. Önce 'python src/train.py' çalıştırın.",
        )

    try:
        temiz = metin_temizle(istek.text)

        if not temiz.strip():
            raise HTTPException(status_code=422, detail="Temizlendikten sonra metin boş kaldı.")

        # Ham metni TF-IDF matrisine dönüştür, sonra tahmin yap
        X = vectorizer.transform([temiz])
        duygu = str(model.predict(X)[0])

        # Güven skoru
        try:
            olasiliklar = model.predict_proba(X)[0]
            siniflar = list(model.classes_)
            guven = float(olasiliklar[siniflar.index(duygu)])
        except (AttributeError, ValueError):
            guven = 1.0

        # Konu tespiti
        konu = konu_siniflandir(istek.text)

        # Özet
        ozet = ozet_uret(duygu, konu)

        return AnalizYaniti(
            duygu=duygu,
            konu=konu,
            guven_skoru=round(guven, 4),
            ozet=ozet,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analiz sırasında hata oluştu: {str(e)}")


@app.get("/api/health", summary="Model durumu")
async def saglik_kontrolu():
    """Model yükleme durumunu ve sürüm bilgisini döndürür."""
    return {
        "durum": "hazır" if app_state.get("model") else "model_yok",
        "model_yolu": MODEL_YOLU,
        "model_yuklendi": app_state.get("model") is not None,
    }


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=API_DIZINI,
    )
