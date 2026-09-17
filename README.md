

# Kredi Risk Analizi ve Tahminleme Projesi 📊

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM-4CAF50?logo=scikit-learn)

Bu proje, makine öğrenmesi algoritmaları kullanarak kredi başvuru sahiplerinin risk durumlarını değerlendirmeyi ve temerrüde düşme (krediyi ödeyememe) olasılıklarını tahmin etmeyi amaçlayan uçtan uca (end-to-end) bir Veri Bilimi ve Yapay Zeka projesidir. Aynı zamanda projeye entegre edilmiş LLM (Büyük Dil Modelleri) destekli akıllı bir asistan ile analiz sonuçları kullanıcıya doğal dilde yorumlanarak sunulmaktadır.

Bu proje, **Miuul Veri Bilimi Bootcamp** bitirme projesi olarak geliştirilmiştir.

---

## 🌟 Projenin Temel Özellikleri

- **Kapsamlı Keşifçi Veri Analizi (EDA):** Veri setindeki eksik değerlerin, aykırı değerlerin tespiti ve değişkenler arası ilişkilerin istatistiksel ve görsel analizi.
- **Gelişmiş Özellik Mühendisliği (Feature Engineering):** Veri setinden yeni, anlamlı değişkenler türetilerek makine öğrenmesi modellerinin performansının artırılması.
- **Tahminleyici Modelleme:** LightGBM gibi güçlü gradyan artırma algoritmaları kullanılarak kredi riskinin yüksek doğrulukla modellenmesi.
- **LLM Entegrasyonu (Gemini & Groq):** Model sonuçlarına dayanarak müşteriye özel ret/onay sebepleri ve tavsiyeler üreten yapay zeka asistanı.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş, kullanıcıların kredi başvuru verilerini girerek anında risk raporu ve LLM yorumu alabildiği modern web uygulaması.

---

## 📁 Proje Yapısı

```bash
Credit_Analysis_Project/
├── 01_eda_and_preprocessing.py      # Keşifçi Veri Analizi ve Veri Ön İşleme adımları
├── 02_feature_engineering.py        # Özellik Mühendisliği (Yeni değişkenlerin üretilmesi)
├── 03_modeling_evaluation.py        # Makine Öğrenmesi modellerinin kurulması ve değerlendirilmesi
└── streamlit_web_app/               # Kullanıcı arayüzü uygulaması
    ├── app.py                       # Ana Streamlit uygulama dosyası
    ├── lgbm_credit_risk_model.pkl   # Eğitilmiş LightGBM Modeli
    ├── requirements.txt             # Proje bağımlılıkları
    ├── .env.example                 # Örnek çevre değişkenleri dosyası
    └── utils/                       # Yardımcı fonksiyonlar (Veri işleme, modelleme, LLM)
```

---

## 🔬 Veri Bilimi Süreci ve Kod Dosyalarının Kullanımı

Projede modelin oluşturulma aşaması modüler olarak 3 ayrı Python dosyasına bölünmüştür. Eğer modelin nasıl eğitildiğini incelemek veya modeli sıfırdan eğitmek isterseniz, bu dosyaları sırasıyla çalıştırabilirsiniz:

### 1. Keşifçi Veri Analizi ve Ön İşleme (`01_eda_and_preprocessing.py`)
- **Amacı:** Ham veriyi alıp, içindeki desenleri keşfetmek ve eksik/hatalı verileri temizlemek.
- **Yaptıkları:**
  - Veri setindeki eksik değerlerin (Missing Values) görselleştirilmesi ve medyan/mod yöntemleriyle doldurulması.
  - Aykırı değerlerin (Outliers) IQR yöntemiyle tespit edilip baskılanması.
  - Korelasyon analizi ve değişkenlerin dağılım grafiklerinin çizilmesi.
- **Kullanımı:** `python 01_eda_and_preprocessing.py`

### 2. Özellik Mühendisliği (`02_feature_engineering.py`)
- **Amacı:** Makine öğrenmesi modelinin örüntüleri daha iyi öğrenmesini sağlayacak yeni, anlamlı değişkenler (features) üretmek.
- **Yaptıkları:**
  - Mevcut verilerden mantıksal yeni değişkenler oluşturulması (Örn: *Kredi/Gelir Oranı, Kredi/Yaş Oranı*).
  - Kategorik değişkenlerin One-Hot Encoding ve Label Encoding ile sayısal formata dönüştürülmesi.
  - Veri setinin ölçeklendirilmesi (Scaling).
- **Kullanımı:** `python 02_feature_engineering.py`

### 3. Modelleme ve Değerlendirme (`03_modeling_evaluation.py`)
- **Amacı:** İşlenmiş veri seti üzerinde farklı algoritmalarla modeller kurmak, performanslarını ölçmek ve en iyi modeli kaydetmek.
- **Yaptıkları:**
  - Verinin Train/Test olarak bölünmesi ve sınıf dengesizliğinin giderilmesi (SMOTE vb.).
  - Çeşitli algoritmaların (LightGBM vb.) eğitilmesi ve hiperparametre optimizasyonu.
  - Accuracy, ROC-AUC, F1-Score gibi metriklerle modellerin kıyaslanması.
  - En başarılı modelin `.pkl` dosyası olarak kaydedilip `streamlit_web_app` klasörüne aktarılması.
- **Kullanımı:** `python 03_modeling_evaluation.py`

---

## 🛠 Kullanılan Teknolojiler

- **Dil:** Python 3.9+
- **Veri İşleme & Analiz:** Pandas, NumPy, Scikit-Learn
- **Makine Öğrenmesi:** LightGBM
- **Web Arayüzü:** Streamlit
- **Yapay Zeka (LLM):** Google Gemini, Groq API

---

## 🚀 Kurulum ve Çalıştırma

Projeyi yerel bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/Credit_Analysis_Project.git
cd Credit_Analysis_Project
```

### 2. Sanal Ortam Oluşturun (Opsiyonel ama Önerilir)
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# MacOS/Linux için:
source venv/bin/activate
```

### 3. Gerekli Kütüphaneleri Yükleyin
```bash
cd streamlit_web_app
pip install -r requirements.txt
```

### 4. Çevre Değişkenlerini (Environment Variables) Ayarlayın
`streamlit_web_app` klasörü içindeki `.env.example` dosyasının adını `.env` olarak değiştirin ve içine kendi API anahtarlarınızı girin:

```env
GEMINI_API_KEY=sizin_gemini_api_anahtariniz
GROQ_API_KEY=sizin_groq_api_anahtariniz
```

### 5. Uygulamayı Başlatın
```bash
streamlit run app.py
```
Uygulama varsayılan olarak tarayıcınızda `http://localhost:8501` adresinde açılacaktır.

---

## 👥 Takım (Geliştiriciler)

Bu proje harika bir ekip çalışmasının ürünüdür:

- **Burak Pabuccu** - *Veri Bilimcisi & Geliştirici* - [GitHub Profili](https://github.com/burakpabuccuu)
- **Banu Serra Batar** - *Veri Bilimcisi & Geliştirici* - [GitHub Profili]() *(Linkleri güncelleyebilirsiniz)*
- **Tunahan Sönmez** - *Veri Bilimcisi & Geliştirici* - [GitHub Profili](https://github.com/tunahansonmez) *(Linkleri güncelleyebilirsiniz)*

Miuul Bootcamp sürecinde elde ettiğimiz bilgi birikimi ve takım içi uyum sayesinde bu başarılı projeyi ortaya koyduk! 🎉

---

## 📜 Lisans

Bu proje eğitim amaçlı geliştirilmiştir ve açık kaynaklıdır. İstediğiniz gibi inceleyebilir ve yararlanabilirsiniz.
