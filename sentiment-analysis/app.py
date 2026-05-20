"""
Türkçe Duygu & Konu Analizi — FastAPI Sunucusu
===============================================
Çalıştırma : uvicorn app:app --host 0.0.0.0 --port 8000 --reload
             (veya doğrudan: python app.py)

Gereksinim : Önce train.py çalıştırılmış olmalı (models/ klasörü dolu olmalı)

Endpoints:
    POST /api/analyze  → { duygu, konu, guven_skoru, duygu_olasiliklari }
    GET  /api/health   → Model yükleme durumu
    GET  /             → Kılavuz
"""

# ── Standart kütüphaneler ────────────────────────────────────────────────────
import os
import re
import sys
import logging
from contextlib import asynccontextmanager

# ── Üçüncü parti ─────────────────────────────────────────────────────────────
import joblib
import numpy as np
import nltk

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# YAPILANDIRMA
# ════════════════════════════════════════════════════════════════════════════
PROJE_KOKU = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(PROJE_KOKU, "models")

MODEL_DOSYALARI = {
    "duygu_vec":  "sentiment_vectorizer.joblib",
    "duygu_clf":  "sentiment_model.joblib",
    "konu_vec":   "topic_vectorizer.joblib",
    "konu_clf":   "topic_model.joblib",
}

# ════════════════════════════════════════════════════════════════════════════
# NLTK & STEMMER (train.py ile BİREBİR AYNI — tutarlılık kritik!)
# ════════════════════════════════════════════════════════════════════════════
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
    # KALDIRILDI — duygu kuvvetlendiriciler:
    # "çok"  → "çok kötü", "çok güzel" gibi ifadelerde anlam taşır
    # "hiç"  → "hiç iyi değil" gibi güçlü olumsuzlamalarda kritik
    # "daha" → karşılaştırma içerir ("daha kötü", "daha iyi")
    # "en"   → üstünlük ifadesi ("en kötü", "en güzel")
}
# Duygu kuvvetlendiriciler: NLTK listesinde de olsa kesinlikle korunmalı.
# "çok kötü", "hiç beğenmedim", "en berbat", "daha iyi" gibi ifadelerde
# bu kelimeler silinirse model sinyali zayıflar veya tersine döner.
DUYGU_KUVVETLENDIRICILER = {"çok", "hiç", "daha", "en", "ama", "fakat"}

try:
    STOPWORDS = (set(stopwords.words("turkish")) | OZEL_STOPWORD) - DUYGU_KUVVETLENDIRICILER
except OSError:
    STOPWORDS = OZEL_STOPWORD - DUYGU_KUVVETLENDIRICILER

try:
    from TurkishStemmer import TurkishStemmer as _TS
    _stemmer = _TS()
    USE_STEMMER = True
except ImportError:
    USE_STEMMER = False
    log.warning("TurkishStemmer bulunamadı — stemming devre dışı.")


# ════════════════════════════════════════════════════════════════════════════
# METİN TEMİZLEME (train.py ile birebir aynı fonksiyon)
# ════════════════════════════════════════════════════════════════════════════

def metin_temizle(metin: str) -> str:
    """
    Ham Türkçe metni eğitimde kullanılan pipeline ile temizler.
    train.py'deki fonksiyonla AYNI olması zorunludur;
    aksi hâlde vektör uzayı tutarsız olur.
    """
    if not isinstance(metin, str):
        return ""

    metin = metin.lower()
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)
    metin = re.sub(r"\S+@\S+\.\S+", " ", metin)
    metin = re.sub(r"\d+", " ", metin)
    metin = re.sub(r"[^\w\sçğışöüÇĞİŞÖÜ]", " ", metin)
    metin = re.sub(r"\s+", " ", metin).strip()

    kelimeler = [
        k for k in metin.split()
        if k not in STOPWORDS and len(k) > 1
    ]

    if USE_STEMMER:
        try:
            kelimeler = [_stemmer.stem(k) for k in kelimeler]
        except Exception:
            pass

    return " ".join(kelimeler)


# ════════════════════════════════════════════════════════════════════════════
# UYGULAMA DURUMU (modeller burada tutulur)
# ════════════════════════════════════════════════════════════════════════════
_modeller: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama başlarken modelleri RAM'e yükler; kapanışta temizler."""
    log.info("Modeller yükleniyor...")
    eksik = []
    for anahtar, dosya in MODEL_DOSYALARI.items():
        yol = os.path.join(MODEL_DIR, dosya)
        if not os.path.exists(yol):
            eksik.append(yol)
        else:
            _modeller[anahtar] = joblib.load(yol)
            log.info(f"  Yüklendi: {dosya}")

    if eksik:
        for yol in eksik:
            log.error(f"  BULUNAMADI: {yol}")
        log.error("Önce 'python train.py' çalıştırın.")
    else:
        log.info("Tüm modeller başarıyla yüklendi.")

    yield
    _modeller.clear()
    log.info("Modeller bellekten temizlendi.")


# ════════════════════════════════════════════════════════════════════════════
# FASTAPI UYGULAMASI
# ════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Türkçe Duygu & Konu Analizi API",
    description=(
        "Hepsiburada ürün yorumları için Lojistik Regresyon + TF-IDF + SMOTE "
        "tabanlı duygu ve konu sınıflandırma servisi."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC ŞEMAları
# ════════════════════════════════════════════════════════════════════════════

class AnalizIstegi(BaseModel):
    metin: str = Field(
        ...,
        min_length=3,
        max_length=5_000,
        description="Analiz edilecek ham Türkçe yorum metni",
        examples=["Ürün çok kaliteliydi, kargoda hiç sorun yaşamadım."],
    )
    yildiz: int | None = Field(
        default=None,
        ge=1,
        le=5,
        description=(
            "Opsiyonel yıldız puanı (1-5). "
            "1-2 → olumsuz garantisi, 3 → model kararı, 4-5 → olumlu garantisi. "
            "Verilmezse yalnızca metin modeli kullanılır."
        ),
        examples=[2],
    )


class AnalizYaniti(BaseModel):
    duygu: str = Field(..., description="olumlu | olumsuz | nötr")
    konu: str  = Field(..., description="hizmet | fiyat | urun | hijyen | teslimat")
    guven_skoru: float = Field(..., ge=0.0, le=1.0, description="Duygu modeli güven skoru")
    duygu_olasiliklari: dict = Field(..., description="Her sınıf için ham olasılıklar")
    yildiz_override: bool = Field(
        default=False,
        description="True ise yıldız puanı modelin tahminini geçersiz kıldı.",
    )


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINTler
# ════════════════════════════════════════════════════════════════════════════

@app.post(
    "/api/analyze",
    response_model=AnalizYaniti,
    summary="Duygu ve konu analizi",
    description=(
        "Türkçe yorumu alır; duygu (olumlu/olumsuz/nötr), "
        "konu (hizmet/fiyat/urun/hijyen/teslimat) ve güven skoru döndürür."
    ),
)
async def analiz_et(istek: AnalizIstegi):
    # Model yüklü mu?
    gerekli = list(MODEL_DOSYALARI.keys())
    eksik   = [k for k in gerekli if k not in _modeller]
    if eksik:
        raise HTTPException(
            status_code=503,
            detail=(
                "Bir veya daha fazla model yüklenemedi. "
                "Sunucuyu başlatmadan önce 'python train.py' çalıştırın."
            ),
        )

    # ── Metin Ön İşleme ──────────────────────────────────────────────────────
    temiz = metin_temizle(istek.metin)
    if len(temiz.strip()) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                "Metin, ön işlemeden sonra çok kısa kaldı. "
                "Lütfen daha açıklayıcı bir yorum girin."
            ),
        )

    # ── Yıldız Puanı: Kesin Override Kontrolü ────────────────────────────────
    # Rapordan: 1-2 yıldız=olumsuz, 3=nötr, 4-5=olumlu.
    # 1-2 ve 4-5 yıldızda kullanıcının niyeti nettir; model tahminini ezmek
    # daha güvenilir sonuç verir. 3 yıldızda model kararı korunur.
    YILDIZ_OVERRIDE = {
        1: "olumsuz",
        2: "olumsuz",
        # 3 → model kararı (kararsız yorum)
        4: "olumlu",
        5: "olumlu",
    }
    override_aktif = istek.yildiz is not None and istek.yildiz in YILDIZ_OVERRIDE

    if override_aktif:
        duygu_etiketi  = YILDIZ_OVERRIDE[istek.yildiz]
        # Model hâlâ çalıştırılır; olasılıklar log ve şeffaflık için kullanılır.
        try:
            duygu_x        = _modeller["duygu_vec"].transform([temiz])
            duygu_olasilik = _modeller["duygu_clf"].predict_proba(duygu_x)[0]
            duygu_siniflari = _modeller["duygu_clf"].classes_
            olasilik_dict  = {
                str(s): round(float(p), 4)
                for s, p in zip(duygu_siniflari, duygu_olasilik)
            }
            model_tahmini = _modeller["duygu_clf"].predict(duygu_x)[0]
            if model_tahmini != duygu_etiketi:
                log.info(
                    f"Yıldız override: model='{model_tahmini}' → "
                    f"yıldız({istek.yildiz})='{duygu_etiketi}'"
                )
        except Exception:
            olasilik_dict = {duygu_etiketi: 1.0}
        guven = 1.0   # yıldız puanı kesin sinyal, güven tam

    else:
        # ── Duygu Tahmini (model) ────────────────────────────────────────────
        try:
            duygu_x        = _modeller["duygu_vec"].transform([temiz])
            duygu_etiketi  = _modeller["duygu_clf"].predict(duygu_x)[0]
            duygu_olasilik = _modeller["duygu_clf"].predict_proba(duygu_x)[0]
            duygu_siniflari = _modeller["duygu_clf"].classes_

            guven = float(
                duygu_olasilik[list(duygu_siniflari).index(duygu_etiketi)]
            )
            olasilik_dict = {
                str(s): round(float(p), 4)
                for s, p in zip(duygu_siniflari, duygu_olasilik)
            }
        except Exception as exc:
            log.exception("Duygu tahmini sırasında hata:")
            raise HTTPException(status_code=500, detail=f"Duygu tahmini başarısız: {exc}")

    # ── Konu Tahmini ─────────────────────────────────────────────────────────
    try:
        konu_x       = _modeller["konu_vec"].transform([temiz])
        konu_etiketi = _modeller["konu_clf"].predict(konu_x)[0]
    except Exception as exc:
        log.exception("Konu tahmini sırasında hata:")
        raise HTTPException(status_code=500, detail=f"Konu tahmini başarısız: {exc}")

    # ── Yanıt ────────────────────────────────────────────────────────────────
    return AnalizYaniti(
        duygu=str(duygu_etiketi),
        konu=str(konu_etiketi),
        guven_skoru=round(guven, 4),
        duygu_olasiliklari=olasilik_dict,
        yildiz_override=override_aktif,
    )


@app.get("/api/health", summary="Model durumu")
async def saglik():
    """Yüklü model sayısını ve hangi modellerin hazır olduğunu döndürür."""
    durum = {k: k in _modeller for k in MODEL_DOSYALARI}
    return {
        "durum":           "hazır" if all(durum.values()) else "model_eksik",
        "yuklenen_modeller": durum,
        "model_dizini":    MODEL_DIR,
    }


@app.get("/", summary="API Kılavuzu")
async def anasayfa():
    return {
        "servis":    "Türkçe Duygu & Konu Analizi API v2",
        "durum":     "çalışıyor",
        "analiz":    "POST /api/analyze  →  { metin: '...' }",
        "saglik":    "GET  /api/health",
        "belgeler":  "GET  /docs",
    }


# ════════════════════════════════════════════════════════════════════════════
# DOĞRUDAN ÇALIŞTIRMA
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
