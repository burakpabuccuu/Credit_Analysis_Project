"""
Veri yükleme ve müşteri sorgulama yardımcı fonksiyonları.
"""

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────
# Veri Yükleme (Cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="📊 Veri seti yükleniyor...")
def load_dataset(csv_path: str = "test_dataset_final.csv") -> pd.DataFrame:
    """Test veri setini diskten yükler ve bellekte tutar."""
    df = pd.read_csv(csv_path)
    return df


# ─────────────────────────────────────────────
# Müşteri Sorgulama
# ─────────────────────────────────────────────

def get_customer_by_id(df: pd.DataFrame, customer_id: int) -> pd.DataFrame | None:
    """
    ID'ye göre müşteri verisini döndürür.

    Returns:
        pd.DataFrame (tek satır) veya None (bulunamadıysa)
    """
    result = df[df["ID"] == customer_id]
    if result.empty:
        return None
    return result


def get_all_customer_ids(df: pd.DataFrame) -> list:
    """Tüm müşteri ID'lerini sıralı liste olarak döndürür."""
    return sorted(df["ID"].unique().tolist())


def get_customer_summary(customer_data: pd.DataFrame) -> dict:
    """
    Müşteri verisinden okunabilir bir özet çıkarır.
    UI'da müşteri bilgi kartında kullanılır.
    """
    row = customer_data.iloc[0]

    sex_map = {1: "Erkek", 2: "Kadın"}
    education_map = {1: "Lisansüstü", 2: "Üniversite", 3: "Lise", 4: "Diğer"}
    marriage_map = {1: "Evli", 2: "Bekar", 3: "Diğer"}

    return {
        "id": int(row["ID"]),
        "limit_bal": f"{row['LIMIT_BAL']:,.0f} ₺",
        "sex": sex_map.get(int(row["SEX"]), "Bilinmiyor"),
        "education": education_map.get(int(row["EDUCATION"]), "Diğer"),
        "marriage": marriage_map.get(int(row["MARRIAGE"]), "Diğer"),
        "age": int(row["AGE"]),
        "avg_bill": f"{row['avg_bill_amt']:,.0f} ₺",
        "avg_payment": f"{row['avg_payment_amt']:,.0f} ₺",
        "utilization": f"{row['avg_utilization_ratio']:.1%}",
        "has_delay": "Evet ⚠️" if int(row["has_delay"]) == 1 else "Hayır ✅",
        "delay_months": int(row["delay_month_count"]),
        "max_delay": int(row["max_positive_delay"]),
    }


# ─────────────────────────────────────────────
# Anlaşılır Özellik (Feature) İsimleri
# ─────────────────────────────────────────────

FEATURE_NAME_MAP = {
    "LIMIT_BAL": "Kredi Limiti",
    "SEX": "Cinsiyet",
    "EDUCATION": "Eğitim Düzeyi",
    "MARRIAGE": "Medeni Durum",
    "AGE": "Yaş",
    "PAY_0": "Son Ay Ödeme Durumu",
    "PAY_2": "2 Ay Önceki Ödeme Durumu",
    "PAY_3": "3 Ay Önceki Ödeme Durumu",
    "PAY_4": "4 Ay Önceki Ödeme Durumu",
    "PAY_5": "5 Ay Önceki Ödeme Durumu",
    "PAY_6": "6 Ay Önceki Ödeme Durumu",
    "BILL_AMT1": "Son Fatura Tutarı",
    "BILL_AMT2": "2 Ay Önceki Fatura Tutarı",
    "BILL_AMT3": "3 Ay Önceki Fatura Tutarı",
    "BILL_AMT4": "4 Ay Önceki Fatura Tutarı",
    "BILL_AMT5": "5 Ay Önceki Fatura Tutarı",
    "BILL_AMT6": "6 Ay Önceki Fatura Tutarı",
    "PAY_AMT1": "Son Ödenen Tutar",
    "PAY_AMT2": "2 Ay Önce Ödenen Tutar",
    "PAY_AMT3": "3 Ay Önce Ödenen Tutar",
    "PAY_AMT4": "4 Ay Önce Ödenen Tutar",
    "PAY_AMT5": "5 Ay Önce Ödenen Tutar",
    "PAY_AMT6": "6 Ay Önce Ödenen Tutar",
    "delay_month_count": "Gecikmeli Ay Sayısı",
    "has_delay": "Gecikme Geçmişi",
    "max_positive_delay": "Maksimum Gecikme Süresi",
    "recent_delay": "Son Dönem Gecikmesi",
    "recent_3m_delay_count": "Son 3 Ay Gecikme Sayısı",
    "older_3m_delay_count": "Önceki 3 Ay Gecikme Sayısı",
    "delay_count_change": "Gecikme Değişim Eğilimi",
    "positive_bill_month_count": "Aktif Borçlu Ay Sayısı",
    "avg_bill_amt": "Ortalama Fatura Tutarı",
    "avg_payment_amt": "Ortalama Ödeme Tutarı",
    "avg_utilization_ratio": "Ort. Limit Kullanım Oranı",
    "total_payment_to_bill_ratio": "Toplam Ödeme / Fatura Oranı",
    "log_total_payment_to_bill_ratio": "Ödeme / Borç Düzeyi",
    "recent_3m_avg_bill": "Son 3 Ay Ort. Fatura",
    "older_3m_avg_bill": "Önceki 3 Ay Ort. Fatura",
    "bill_change_3m": "3 Aylık Fatura Değişimi",
    "bill_trend_direction": "Fatura Eğilim Yönü",
    "recent_3m_avg_payment": "Son 3 Ay Ort. Ödeme",
    "older_3m_avg_payment": "Önceki 3 Ay Ort. Ödeme",
    "payment_change_3m": "3 Aylık Ödeme Değişimi",
    "payment_trend_direction": "Ödeme Eğilim Yönü",
    "recent_3m_utilization": "Son 3 Ay Limit Kullanım Oranı",
    "older_3m_utilization": "Önceki 3 Ay Limit Kullanım Oranı",
    "utilization_change_3m": "Limit Kullanım Değişimi",
    "payment_change_to_limit": "Ödeme Değişim / Limit Oranı",
}


def get_friendly_feature_name(feature_name: str) -> str:
    """Teknik özellik adını kullanıcı dostu Türkçe isme dönüştürür."""
    return FEATURE_NAME_MAP.get(feature_name, feature_name)

