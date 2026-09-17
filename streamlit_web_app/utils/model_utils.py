"""
Model ve SHAP yardımcı fonksiyonları.
LightGBM Pipeline modelini yükler, tahmin yapar ve SHAP değerlerini hesaplar.
"""

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Model Yükleme (Cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🔄 Model yükleniyor...")
def load_model(model_path: str = "lgbm_credit_risk_model.pkl"):
    """Modeli diskten yükler ve bellekte tutar."""
    model = joblib.load(model_path)
    return model


# ─────────────────────────────────────────────
# Tahmin Fonksiyonları
# ─────────────────────────────────────────────

def predict_customer(model, customer_data: pd.DataFrame) -> dict:
    """
    Tek bir müşteri için tahmin üretir.
    Karar eşiği .env dosyasındaki RISK_THRESHOLD değerine göre belirlenir.

    Returns:
        dict: {
            "prediction": 0 veya 1,
            "probability_default": float (0-1),
            "probability_no_default": float (0-1),
            "label": "KREDİ ONAY" veya "KREDİ RED",
            "risk_score": float (0-100)
        }
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    feature_cols = list(model.feature_names_in_)
    X = customer_data[feature_cols]

    proba = model.predict_proba(X)[0]
    risk_score = float(proba[1]) * 100  # 0-100 arası risk skoru

    # .env veya st.secrets'tan eşik değerini oku (varsayılan: 35)
    threshold_val = 35.0
    try:
        if hasattr(st, "secrets") and "RISK_THRESHOLD" in st.secrets:
            threshold_val = float(st.secrets["RISK_THRESHOLD"])
        else:
            threshold_val = float(os.getenv("RISK_THRESHOLD", "35"))
    except Exception:
        threshold_val = 35.0

    threshold = threshold_val
    prediction = 1 if risk_score >= threshold else 0

    return {
        "prediction": prediction,
        "probability_default": float(proba[1]),
        "probability_no_default": float(proba[0]),
        "label": "KREDİ RED ❌" if prediction == 1 else "KREDİ ONAY ✅",
        "risk_score": risk_score,
        "threshold": threshold,
    }


# ─────────────────────────────────────────────
# SHAP Hesaplama
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🧮 SHAP açıklayıcı hazırlanıyor...")
def get_shap_explainer(_model):
    """
    Pipeline içindeki LGBMClassifier için SHAP TreeExplainer oluşturur.
    Not: _model prefix'i Streamlit cache hashing'i bypass etmek için.
    """
    lgbm_model = _model.named_steps["model"]
    explainer = shap.TreeExplainer(lgbm_model)
    return explainer


def compute_shap_values(model, explainer, customer_data: pd.DataFrame) -> dict:
    """
    Bir müşteri için SHAP değerlerini hesaplar.

    Returns:
        dict: {
            "shap_values": np.array,
            "base_value": float,
            "feature_names": list,
            "feature_values": list,
            "top_risk_features": list[dict]  — risk artıran ilk 3 feature
            "top_safe_features": list[dict]  — riski azaltan ilk 3 feature
        }
    """
    feature_cols = list(model.feature_names_in_)
    X = customer_data[feature_cols]

    # Pipeline preprocessor'dan geçir (SimpleImputer)
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)

    # SHAP değerlerini hesapla
    shap_vals = explainer.shap_values(X_transformed)

    # Binary classification: class 1 (default) için SHAP değerlerini al
    if isinstance(shap_vals, list):
        shap_values_default = shap_vals[1][0]  # Class 1, first sample
    else:
        shap_values_default = shap_vals[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]  # Class 1

    # Feature importance sıralama
    feature_importance = list(zip(feature_cols, shap_values_default, X.iloc[0].values))
    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

    # Risk artıran (pozitif SHAP) ve azaltan (negatif SHAP) feature'lar
    top_risk = [
        {"feature": f, "shap_value": round(float(sv), 4), "value": v}
        for f, sv, v in feature_importance if sv > 0
    ][:3]

    top_safe = [
        {"feature": f, "shap_value": round(float(sv), 4), "value": v}
        for f, sv, v in feature_importance if sv < 0
    ][:3]

    return {
        "shap_values": shap_values_default,
        "base_value": float(base_value),
        "feature_names": feature_cols,
        "feature_values": X.iloc[0].values.tolist(),
        "top_risk_features": top_risk,
        "top_safe_features": top_safe,
    }
