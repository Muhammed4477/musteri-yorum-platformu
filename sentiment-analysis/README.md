# Türkçe Müşteri Yorumlarında Duygu Analizi

Müşteri yorumlarını **olumlu / olumsuz / notr** olarak sınıflandıran ve **konu** (hizmet, fiyat, ürün, hijyen, teslimat) tespiti yapan NLP pipeline'ı.

## Proje Yapısı

```
sentiment-analysis/
├── data/               # Ham ve işlenmiş veri
├── figures/            # Confusion matrix ve karşılaştırma grafikleri
├── models/             # Kaydedilen modeller (.pkl)
├── notebooks/          # Jupyter notebooklar
├── src/
│   ├── preprocess.py   # Veri ön işleme pipeline'ı
│   ├── train.py        # Model eğitimi (Naive Bayes + Lojistik Regresyon)
│   ├── evaluate.py     # Model değerlendirme ve görselleştirme
│   └── predict.py      # Tek yorum tahmini
├── api/
│   └── main.py         # FastAPI uygulaması
└── requirements.txt
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### 1. Veri Hazırlama
CSV dosyanızın `yorum_metni` ve `puan` (1–5) sütunları olmalı.

```python
from src.preprocess import on_isleme_pipeline
veri = on_isleme_pipeline("data/yorumlar.csv")
```

### 2. Model Eğitimi

```bash
python src/train.py
```

### 3. Değerlendirme

```bash
python src/evaluate.py
```

Çıktılar:
- Konsol: Accuracy, Makro F1, Classification Report
- `figures/confusion_matrix_*.png`
- `figures/model_karsilastirma.png`

### 4. Tek Yorum Tahmini

```python
from src.predict import yorumu_tahmin_et
sonuc = yorumu_tahmin_et("Ürün çok güzeldi, kesinlikle tavsiye ederim!")
print(sonuc)
# {'duygu': 'olumlu', 'guven_skoru': 0.9231, 'temiz_metin': '...'}
```

### 5. API Başlatma

```bash
cd api
uvicorn main:app --reload --port 8000
```

**Endpoint:** `POST http://localhost:8000/api/analyze`

```json
// İstek
{"text": "Kargo geç geldi ama ürün çok güzeldi."}

// Yanıt
{
  "duygu": "olumlu",
  "konu": "teslimat",
  "guven_skoru": 0.87,
  "ozet": "Teslimat süreci kabul edilebilir."
}
```

## Metrikler

| Model               | Hedef Makro F1 |
|---------------------|---------------|
| Naive Bayes         | ≥ 0.80        |
| Lojistik Regresyon  | ≥ 0.80        |

## Veri Formatı

```csv
yorum_metni,puan
"Ürün harika geldi çok beğendim",5
"Kargo geç geldi hayal kırıklığı",1
"Fiyatına göre idare eder",3
```
