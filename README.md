<div align="center">

# 🧠 MüşteriYorum Yönetim Platformu

### Türkçe Müşteri Yorumlarında Yapay Zeka Destekli Duygu Analizi

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://postgresql.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

**İki dersin mikroservis mimarisiyle birleşimi:**  
İnternet Programcılığı (Node.js + PostgreSQL) × Yapay Zeka (NLP + FastAPI)

---

**Geliştirici:** [Muhammed Çelik](https://www.linkedin.com/in/muhammed-%C3%A7elik-766915286/) &nbsp;|&nbsp; Yapay Zeka & İnternet Programcılığı Dersi Projesi

</div>

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Özellikler](#-özellikler)
- [Sistem Mimarisi](#-sistem-mimarisi)
- [Kullanılan Teknolojiler](#-kullanılan-teknolojiler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Yapay Zeka Modeli](#-yapay-zeka-modeli)
- [API Dokümantasyonu](#-api-dokümantasyonu)
- [Ekran Görüntüleri](#-ekran-görüntüleri)
- [Katkıda Bulunma](#-katkıda-bulunma)
- [Geliştirici](#-geliştirici)

---

## 🎯 Proje Hakkında

**MüşteriYorum Yönetim Platformu**, kafe, restoran, kuaför ve otel gibi KOBİ'lerin müşteri yorumlarını otomatik olarak analiz eden, web tabanlı bir yönetim sistemidir.

### Çözdüğü Problem

Türkiye'deki küçük işletmeler günlük yüzlerce müşteri yorumu alır; ancak:
- Yorumları manuel okumak zaman alıcıdır
- Hangi konularda şikayet aldıkları net görünmez
- Olumlu/olumsuz yorum oranını takip edemezler

Bu platform, **yapay zeka destekli duygu analizi**, **görsel raporlar** ve **kategori bazlı filtreleme** ile bu sorunları otomatik çözer.

---

## ✨ Özellikler

| Özellik | Açıklama |
|---|---|
| 🤖 **Duygu Analizi** | Her yorum otomatik olumlu / olumsuz / nötr olarak etiketlenir |
| 🏷️ **Konu Sınıflandırma** | Hizmet, Fiyat, Ürün, Hijyen, Teslimat kategorileri |
| 📊 **Görsel Dashboard** | Pasta, çubuk ve trend grafikleri (Chart.js) |
| 📁 **Toplu Yükleme** | CSV / Excel dosyasıyla yüzlerce yorum bir anda yüklenir |
| 📄 **Rapor İndirme** | PDF ve Excel formatında rapor çıktısı |
| 👥 **3 Rol Sistemi** | Müşteri, İşletme Sahibi, Admin rolleri |
| 🔐 **JWT Kimlik Doğrulama** | Güvenli, stateless oturum yönetimi |
| 🌐 **Mikroservis Mimarisi** | Node.js backend ↔ FastAPI YZ servisi |

---

## 🏗️ Sistem Mimarisi

```
┌─────────────────────────────────────┐
│          KULLANICI                  │
│  (Müşteri / İşletme Sahibi / Admin) │
└───────────────┬─────────────────────┘
                │ HTTPS
                ▼
┌─────────────────────────────────────┐
│     WEB ARAYÜZ (Frontend)           │
│   HTML5 + Bootstrap 5 + Chart.js    │
└───────────────┬─────────────────────┘
                │ fetch() + JWT Token
                ▼
┌─────────────────────────────────────┐
│       NODE.JS BACKEND               │
│     Express + Sequelize + JWT       │
│           Port: 3000                │
└──┬───────────────────────────┬──────┘
   │                           │
   │ Sequelize ORM             │ HTTP fetch
   ▼                           ▼
┌──────────────┐     ┌──────────────────┐
│  PostgreSQL  │     │   FASTAPI (YZ)   │
│ musteri_yorum│     │    Port: 8000    │
│   6 tablo   │     │ Python + sklearn  │
└──────────────┘     └─────────┬────────┘
                               ▼
                    ┌────────────────────┐
                    │    NLP MODELİ      │
                    │ Logistic Regress.  │
                    │ TF-IDF (440k veri) │
                    │   F1: 0.8965       │
                    └────────────────────┘
```

**Veri Akışı:**
1. Müşteri tarayıcıdan yorum yazar
2. JS → `POST /api/musteri/yorum` isteği atar (JWT token ile)
3. Node.js yorumu PostgreSQL'e kaydeder
4. Node.js, FastAPI'ye `POST /api/analyze` isteği atar
5. FastAPI model üzerinden tahmin yapar
6. Sonuç (`duygu`, `konu_etiketi`, `guven_skoru`) döner
7. Node.js sonucu `analysis_results` tablosuna kaydeder
8. İşletme sahibi dashboard'da yorumu duygu rozeti ile görür

---

## 🛠️ Kullanılan Teknolojiler

### Backend
| Teknoloji | Versiyon | Kullanım Amacı |
|---|---|---|
| Node.js | 18+ | Sunucu runtime |
| Express.js | 4.x | REST API framework |
| Sequelize | 6.x | ORM (PostgreSQL bağlantısı) |
| jsonwebtoken | — | JWT kimlik doğrulama |
| bcryptjs | — | Şifre hashleme |
| multer | — | Dosya yükleme |
| pdfkit | — | PDF rapor oluşturma |
| xlsx | — | Excel okuma/yazma |

### Yapay Zeka Servisi
| Teknoloji | Kullanım Amacı |
|---|---|
| Python 3.10+ | ML ekosistemi |
| FastAPI | REST API (YZ servisi) |
| scikit-learn | Logistic Regression + TF-IDF |
| joblib | Model serileştirme (.pkl) |
| pandas | Veri işleme |

### Frontend
| Teknoloji | Kullanım Amacı |
|---|---|
| HTML5 / CSS3 | Sayfa yapısı ve stiller |
| Bootstrap 5 | Responsive UI bileşenleri |
| Vanilla JavaScript | Dinamik davranış, API çağrıları |
| Chart.js | Grafikler (pasta, çubuk, çizgi) |

### Veritabanı
| Teknoloji | Kullanım Amacı |
|---|---|
| PostgreSQL 14+ | Ana ilişkisel veritabanı |

---

## 🚀 Kurulum

### Gereksinimler

```bash
node --version   # 18+ olmalı
python --version # 3.10+ olmalı
psql --version   # PostgreSQL 14+ olmalı
```



## 🤖 Yapay Zeka Modeli

### Veri Seti

| Özellik | Detay |
|---|---|
| Kaynak | [Hepsiburada Türkçe Sentiment (Kaggle)](https://www.kaggle.com/datasets/savasy/ttc4900) |
| Boyut | ~440.000 yorum |
| Sınıflar | Olumlu / Olumsuz / Nötr |
| Format | CSV (metin + yıldız puanı) |

### Model Karşılaştırması

| Model | Accuracy | F1-Score | Seçim |
|---|---|---|---|
| Naive Bayes | 0.8624 | 0.8589 | ✗ |
| **Logistic Regression** | **0.8973** | **0.8965** | **✓** |

### Ön İşleme Pipeline'ı

```
Ham Türkçe Yorum
      ↓
Türkçe karakter normalizasyonu
      ↓
Küçük harfe çevirme + noktalama temizleme
      ↓
Stop-words çıkarma (ve, bir, için, ama...)
      ↓
TF-IDF Vektorizasyon (n-gram: 1-2)
      ↓
Logistic Regression (class_weight='balanced')
      ↓
Tahmin: { duygu, konu_etiketi, guven_skoru }
```

### Fallback Mekanizması

YZ servisi çalışmazsa (timeout, hata) Node.js kelime tabanlı basit bir mock analiz yapar — demo sırasında YZ servisi olmasa bile platform çalışmaya devam eder.

---

## 📡 API Dokümantasyonu

Tüm istekler `Content-Type: application/json` ile yapılır.  
Korumalı endpoint'ler `Authorization: Bearer <jwt>` başlığı gerektirir.

### Auth Endpoint'leri

| Metod | URL | Açıklama | Yetki |
|---|---|---|---|
| POST | `/api/auth/register` | Yeni kullanıcı kaydı | Public |
| POST | `/api/auth/login` | Giriş yapma | Public |
| GET | `/api/auth/me` | Mevcut kullanıcı bilgisi | Login |
| PUT | `/api/auth/me` | Profil güncelleme | Login |

### Yorum Endpoint'leri

| Metod | URL | Açıklama |
|---|---|---|
| GET | `/api/reviews` | Yorumları listele (filtreli) |
| POST | `/api/reviews` | Manuel yorum ekle |
| PUT | `/api/reviews/:id` | Yorum güncelle |
| DELETE | `/api/reviews/:id` | Yorum sil |
| POST | `/api/reviews/upload` | CSV/Excel ile toplu yükleme |

### Analiz Endpoint'leri

| Metod | URL | Açıklama |
|---|---|---|
| GET | `/api/analysis/summary` | Duygu/konu/trend özeti |
| GET | `/api/analysis/reviews-by?duygu=olumlu` | Duyguya göre yorumlar |
| POST | `/api/analysis/trigger` | Seçili yorumları yeniden analiz et |
| GET | `/api/analysis/negative` | Sadece olumsuz yorumlar |

### YZ Servisi

```bash
POST http://localhost:8000/api/analyze

# İstek:
{"text": "Kahve harikaydı, servis biraz yavaştı."}

# Yanıt:
{
  "duygu": "olumlu",
  "konu": "Hizmet",
  "guven_skoru": 0.87,
  "ozet": "Müşteri kahveden memnun, servisten şikayetçi."
}
```

> Tüm endpoint'ler için: `http://localhost:8000/docs` (Swagger UI)

---

## 📁 Proje Yapısı

```
musteri-yorum-platformu/
├── server.js               # Ana Express uygulaması
├── package.json
├── .env                    # Gizli değişkenler (git'e eklenmez)
│
├── config/
│   └── db.js               # Sequelize bağlantısı
│
├── models/                 # Veritabanı modelleri
│   ├── User.js
│   ├── Business.js
│   ├── Review.js
│   ├── AnalysisResult.js
│   ├── Category.js
│   └── BulkUpload.js
│
├── middleware/
│   ├── auth.js             # JWT doğrulama
│   └── role.js             # Rol kontrolü
│
├── routes/                 # API endpoint'leri
│   ├── auth.js
│   ├── reviews.js
│   ├── analysis.js
│   ├── reports.js
│   ├── admin.js
│   ├── businesses.js
│   └── musteri.js
│
├── public/                 # Frontend (HTML/CSS/JS)
│   ├── index.html
│   ├── login.html
│   ├── dashboard/
│   ├── reviews/
│   ├── analysis/
│   ├── reports/
│   ├── admin/
│   ├── musteri/
│   ├── css/style.css
│   └── js/
│       ├── main.js
│       └── charts.js
│
└── sentiment-analysis/     # Yapay Zeka Mikroservisi
    ├── api/
    │   └── main.py         # FastAPI uygulaması
    ├── models/
    │   └── model.pkl       # Eğitilmiş model
    ├── train.py            # Model eğitimi
    ├── predict.py          # Tahmin fonksiyonu
    └── requirements.txt
```

---

## 👨‍💻 Geliştirici

<div align="center">

### Muhammed Çelik

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/muhammed-%C3%A7elik-766915286/)

*Bilgisayar Mühendisliği öğrencisi — Yapay Zeka & Web Geliştirme*

</div>

---

## 📜 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

<div align="center">

**⭐ Beğendiyseniz yıldız vermeyi unutmayın!**

</div>
