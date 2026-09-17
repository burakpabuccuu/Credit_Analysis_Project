"""
LLM entegrasyon modülü.
Gemini Flash (birincil) + Groq Llama 3 (yedek) ile kredi risk açıklaması üretir.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.data_utils import get_friendly_feature_name

load_dotenv()


# ─────────────────────────────────────────────
# Prompt Şablonu
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Sen deneyimli bir banka kredi risk analistisisin. Görevin, sistemin ürettiği kredi risk skorunu 
ve müşterinin ödeme geçmişi ile finansal faktörlerini analiz ederek, şube personeli / kredi tahsis uzmanı için 
anlaşılır, profesyonel ve son kullanıcı dostu bir Türkçe kredi değerlendirme raporu üretmektir.

Kurallar:
1. Kesinlikle "SHAP", "LightGBM", "Makine Öğrenmesi Modeli", "Pipeline", "Veri Bilimi", "Temerrüt Sınıflandırıcısı" gibi teknik kavramları kullanma.
2. Tamamen bankacılık ve kredi tahsis terminolojisi kullan (örn: "kredi geri ödeme performansı", "kart limit kullanım oranı", "gecikme geçmişi").
3. Aksiyon alınabilir somut bankacılık önerileri sun (örn: "Kefil veya ek teminat istenebilir", "Limit düşürülerek onay verilebilir", "Otomatik ödeme talimatı şartıyla onaylanabilir").
4. Yanıtın 120-180 kelime arasında, net ve profesyonel olmalıdır.
5. Emoji kullanarak görsel zenginlik ve kolay okunabilirlik sağla.
6. Yanıtı şu 3 bölüme ayır:
   - 📊 Risk & Kredi Uygunluk Değerlendirmesi
   - 🔍 Öne Çıkan Finansal Bulgular
   - 💡 Şube / Tahsis Önerileri
"""


def _build_user_prompt(prediction_result: dict, shap_result: dict, customer_summary: dict) -> str:
    """LLM'e gönderilecek kullanıcı prompt'unu oluşturur."""

    risk_features = shap_result.get("top_risk_features", [])
    safe_features = shap_result.get("top_safe_features", [])

    risk_text = "\n".join([
        f"  - {get_friendly_feature_name(f['feature'])}: Değer={f['value']}"
        for f in risk_features
    ]) if risk_features else "  - Belirgin risk faktörü yok"

    safe_text = "\n".join([
        f"  - {get_friendly_feature_name(f['feature'])}: Değer={f['value']}"
        for f in safe_features
    ]) if safe_features else "  - Belirgin olumlu faktör yok"

    return f"""
Aşağıdaki müşterinin kredi başvuru analizini değerlendir:

**Müşteri Profili & Finansal Durumu:**
- Müşteri ID: #{customer_summary['id']}
- Yaş: {customer_summary['age']}
- Cinsiyet: {customer_summary['sex']}
- Eğitim Düzeyi: {customer_summary['education']}
- Medeni Durum: {customer_summary['marriage']}
- Mevcut Kredi Limiti: {customer_summary['limit_bal']}
- Ortalama Aylık Fatura: {customer_summary['avg_bill']}
- Ortalama Aylık Ödeme: {customer_summary['avg_payment']}
- Limit Kullanım Oranı: {customer_summary['utilization']}
- Ödeme Gecikme Geçmişi: {customer_summary['has_delay']} (Gecikmeli Ay Sayısı: {customer_summary['delay_months']}, Maks Gecikme: {customer_summary['max_delay']} ay)

**Sistem Değerlendirme Çıktısı:**
- Kredi Kararı: {prediction_result['label']}
- Risk Skoru: %{prediction_result['risk_score']:.1f}

**Kredi Riskini Artıran Faktörler:**
{risk_text}

**Kredi Güvenilirliğini Destekleyen Olumlu Faktörler:**
{safe_text}
"""



def _get_config(key: str, default: str = "") -> str:
    """Ortam değişkenini hem Streamlit secrets hem de os.getenv üzerinden okur."""
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


# ─────────────────────────────────────────────
# Gemini Model Zinciri — Kota/hata durumunda sıradaki modele geçer
# ─────────────────────────────────────────────

def _get_gemini_models() -> list[str]:
    """GEMINI_MODELS virgülle ayrılmış model listesini döndürür."""
    raw = _get_config("GEMINI_MODELS", "")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    # Tek model tanımlıysa (eski format uyumluluğu)
    single = _get_config("GEMINI_MODEL", "")
    if single:
        return [single]
    return ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemma-4-31b-it", "gemini-2.5-flash-lite"]


def _call_gemini(prompt: str) -> tuple[str, str] | None:
    """
    Gemini API çağrısı — model zinciri desteği.
    İlk model kotaya takılırsa veya hata verirse sıradaki model anında denenir.
    Doğrudan API timeout'u (5 sn) kullanılarak thread kilitlenmesi tamamen önlenir.

    Returns:
        tuple(yanıt, model_adı) veya None
    """
    api_key = _get_config("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    models = _get_gemini_models()

    for model_name in models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [SYSTEM_PROMPT, prompt],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=500,
                ),
                request_options={"timeout": 5.0},
            )

            if response and response.text:
                return response.text, model_name

        except Exception as e:
            # Kota (429), model 404, rate limit veya geçici sunucu hatalarında anında sıradakine geç
            st.toast(f"🔄 {model_name} yanıt vermedi — sıradaki model deneniyor...", icon="⚡")
            continue

    st.warning("⚠️ Tanımlı Gemini modelleri yanıt veremedi. Yedek sistem devreye alınıyor...")
    return None


# ─────────────────────────────────────────────
# Groq API (Yedek / Fallback) — 5sn timeout
# ─────────────────────────────────────────────

def _call_groq(prompt: str) -> str | None:
    """Groq API çağrısı (5 saniye timeout)."""
    api_key = _get_config("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key, timeout=5.0)
        response = client.chat.completions.create(
            model=_get_config("GROQ_MODEL", "qwen/qwen3.8-27b"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        st.warning(f"⚠️ Groq API hatası: {e}")
        return None


# ─────────────────────────────────────────────
# Ana Fonksiyon (Çoklu Fallback Mekanizmalı)
# ─────────────────────────────────────────────

def generate_risk_explanation(
    prediction_result: dict,
    shap_result: dict,
    customer_summary: dict,
) -> tuple[str, str]:
    """
    LLM ile kredi risk açıklaması üretir.
    Gemini model zinciri (her biri max 5sn) → Groq (5sn) → Statik rapor.

    Returns:
        tuple: (açıklama_metni, kullanılan_model_adı)
    """
    prompt = _build_user_prompt(prediction_result, shap_result, customer_summary)

    # 1. Gemini model zincirini dene
    gemini_result = _call_gemini(prompt)
    if gemini_result:
        return gemini_result  # tuple(text, model_name) olarak döner

    # 2. Groq fallback (max 5sn)
    result = _call_groq(prompt)
    if result:
        return result, _get_config("GROQ_MODEL", "qwen/qwen3.8-27b")

    # 3. Statik fallback (anında)
    risk_features = shap_result.get("top_risk_features", [])
    risk_text = ", ".join([get_friendly_feature_name(f['feature']) for f in risk_features]) if risk_features else "Belirgin risk faktörü bulunamadı"

    static_msg = (
        f"📊 **Risk Değerlendirmesi:** Yapılan değerlendirme sonucunda müşterinin "
        f"kredi risk seviyesi %{prediction_result['risk_score']:.1f} olarak hesaplanmıştır. "
        f"Ön Karar: **{prediction_result['label']}**\n\n"
        f"🔍 **Temel Bulgular:** Kararı en çok etkileyen göstergeler: {risk_text}.\n\n"
        f"💡 **Öneriler:** Müşterinin son dönem ödeme hareketleri ve limit kullanım dengesi göz önünde bulundurularak nihai tahsis kararı verilmelidir."
    )
    return static_msg, "Sistem Otomatik Değerlendirmesi"
