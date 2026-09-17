"""
🏦 Akıllı Kredi Risk Karar Destek Sistemi
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Streamlit Web Uygulaması — Miuul Bootcamp Bitirme Projesi

LightGBM modeli + SHAP açıklanabilirlik + LLM içgörüsü
"""

import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
import shap
import numpy as np

from utils.model_utils import load_model, predict_customer, get_shap_explainer, compute_shap_values
from utils.data_utils import load_dataset, get_customer_by_id, get_all_customer_ids, get_customer_summary, get_friendly_feature_name
from utils.llm_utils import generate_risk_explanation

# ─────────────────────────────────────────────
# Sayfa Konfigürasyonu
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Kredi Risk Karar Destek Sistemi",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — Profesyonel Banka Teması
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --bank-primary: #0A1628;
        --bank-secondary: #1A2942;
        --bank-accent: #2E7BFF;
        --bank-accent-light: #5B9AFF;
        --bank-gold: #F0B90B;
        --bank-success: #00C48C;
        --bank-danger: #FF4757;
        --bank-warning: #FFA502;
        --bank-text: #E8ECF1;
        --bank-muted: #8899AA;
        --bank-card: rgba(26, 41, 66, 0.7);
        --bank-border: rgba(46, 123, 255, 0.15);
        --glass-bg: rgba(10, 22, 40, 0.85);
        --glass-border: rgba(255, 255, 255, 0.08);
    }

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main Header ── */
    .main-header {
        background: linear-gradient(135deg, #0A1628 0%, #1A2942 50%, #0D2137 100%);
        border: 1px solid var(--bank-border);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(46,123,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: var(--bank-muted);
        font-size: 0.95rem;
        margin: 0.3rem 0 0;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, var(--bank-card), rgba(13, 33, 55, 0.6));
        border: 1px solid var(--glass-border);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        backdrop-filter: blur(20px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(46, 123, 255, 0.12);
    }
    .metric-label {
        color: var(--bank-muted);
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* ── Decision Banner ── */
    .decision-approve {
        background: linear-gradient(135deg, rgba(0,196,140,0.12) 0%, rgba(0,196,140,0.04) 100%);
        border: 1px solid rgba(0,196,140,0.3);
        border-left: 4px solid var(--bank-success);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        animation: fadeIn 0.5s ease;
    }
    .decision-reject {
        background: linear-gradient(135deg, rgba(255,71,87,0.12) 0%, rgba(255,71,87,0.04) 100%);
        border: 1px solid rgba(255,71,87,0.3);
        border-left: 4px solid var(--bank-danger);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        animation: fadeIn 0.5s ease;
    }
    .decision-text {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }

    /* ── Info Cards ── */
    .info-card {
        background: var(--bank-card);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1.2rem;
    }
    .info-label {
        color: var(--bank-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .info-value {
        color: var(--bank-text);
        font-size: 1rem;
        font-weight: 600;
    }

    /* ── AI Insight Box ── */
    .ai-insight {
        background: linear-gradient(145deg, rgba(46,123,255,0.06), rgba(240,185,11,0.04));
        border: 1px solid rgba(46,123,255,0.2);
        border-radius: 14px;
        padding: 1.5rem;
        position: relative;
    }
    .ai-insight::before {
        content: '💡';
        position: absolute;
        top: -12px;
        left: 16px;
        font-size: 1.4rem;
        background: var(--bank-primary);
        padding: 0 8px;
    }
    .ai-insight h3 {
        color: var(--bank-accent-light);
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.8rem;
    }
    .ai-insight p, .ai-insight li {
        color: var(--bank-text);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .ai-model-badge {
        display: inline-block;
        background: rgba(46,123,255,0.15);
        color: var(--bank-accent-light);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-top: 0.8rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0A1628 0%, #0D1B2A 100%);
        border-right: 1px solid var(--bank-border);
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: var(--bank-text) !important;
        font-weight: 500;
    }

    /* ── Section Headers ── */
    .section-header {
        color: var(--bank-text);
        font-size: 1.1rem;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--bank-accent);
        margin-bottom: 1rem;
        display: inline-block;
    }

    /* ── Animations ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* ── Factor Box Container ── */
    .factor-item {
        background: var(--bank-card);
        border: 1px solid var(--glass-border);
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 8px;
    }

    /* ── Streamlit overrides ── */
    .stMetric {
        background: var(--bank-card);
        border: 1px solid var(--glass-border);
        border-radius: 12px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Veri & Model Yükleme
# ─────────────────────────────────────────────
model = load_model()
df = load_dataset()
explainer = get_shap_explainer(model)
all_ids = get_all_customer_ids(df)

# ─────────────────────────────────────────────
# Sidebar — Müşteri Arama
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <span style="font-size: 2.5rem;">🏦</span>
        <h2 style="color: #E8ECF1; font-size: 1.15rem; font-weight: 700; margin: 0.5rem 0 0.2rem;">
            Kredi Risk Analizi
        </h2>
        <p style="color: #8899AA; font-size: 0.8rem; margin: 0;">
            Karar Destek Sistemi v1.0
        </p>
    </div>
    <hr style="border-color: rgba(46,123,255,0.15); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    st.markdown("##### 🔍 Müşteri Sorgula")

    search_method = st.radio(
        "Arama yöntemi:",
        ["ID Listesinden Seç", "ID Numarası Gir"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if search_method == "ID Listesinden Seç":
        selected_id = st.selectbox(
            "Müşteri ID",
            options=all_ids,
            index=0,
            help="Değerlendirilecek müşteriyi ID listesinden seçin.",
        )
    else:
        selected_id = st.number_input(
            "Müşteri ID",
            min_value=min(all_ids),
            max_value=max(all_ids),
            value=all_ids[0],
            step=1,
            help="Müşteri ID numarasını girin.",
        )

    analyze_btn = st.button(
        "🚀 Analiz Et",
        use_container_width=True,
        type="primary",
    )

    st.markdown("<hr style='border-color: rgba(46,123,255,0.15); margin: 1.5rem 0 1rem;'>", unsafe_allow_html=True)

    # Portfolio stats
    st.markdown("##### 📈 Portföy Özeti")
    st.caption(f"Toplam Müşteri Portföyü: **{len(df):,}**")
    default_rate = df["default.payment.next.month"].mean()
    st.caption(f"Portföy Tarihsel Risk Oranı: **{default_rate:.1%}**")
    st.caption(f"Risk Analiz Motoru: **Aktif (Yapay Zeka Destekli)**")

# ─────────────────────────────────────────────
# Ana Başlık
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Akıllı Kredi Risk Karar Destek Sistemi</h1>
    <p>Yapay Zeka Destekli Otomatik Kredi Değerlendirme ve Risk Analiz Portalı</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Ana İçerik
# ─────────────────────────────────────────────
if analyze_btn:
    # Müşteri verisi al
    customer_data = get_customer_by_id(df, selected_id)

    if customer_data is None:
        st.error(f"⚠️ Müşteri ID **{selected_id}** bulunamadı. Lütfen geçerli bir ID girin.")
        st.stop()

    # Müşteri özeti
    summary = get_customer_summary(customer_data)

    # Model tahmini
    prediction = predict_customer(model, customer_data)

    # SHAP hesaplama
    shap_result = compute_shap_values(model, explainer, customer_data)

    # ── Karar Banneri ──
    if prediction["prediction"] == 0:
        st.markdown(f"""
        <div class="decision-approve">
            <p class="decision-text" style="color: #00C48C;">
                ✅ KREDİ TALEBİ UYGUNDUR (ONAY) — Müşteri #{summary['id']}
            </p>
            <p style="color: #8899AA; margin: 0.3rem 0 0; font-size: 0.9rem;">
                Sistem değerlendirmesi: Müşterinin finansal göstergeleri ve geçmiş ödeme performansı kredi tahsisi için uygun seviyededir.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="decision-reject">
            <p class="decision-text" style="color: #FF4757;">
                ❌ KREDİ TALEBİ YÜKSEK RİSKLİ (RED) — Müşteri #{summary['id']}
            </p>
            <p style="color: #8899AA; margin: 0.3rem 0 0; font-size: 0.9rem;">
                Sistem değerlendirmesi: Müşterinin geçmiş ödeme gecikmeleri veya finansal dengeleri sebebiyle yüksek risk tespit edilmiştir.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrikler ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        risk_color = "#FF4757" if prediction["risk_score"] > 50 else "#FFA502" if prediction["risk_score"] > 30 else "#00C48C"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Risk Skoru</div>
            <div class="metric-value" style="color: {risk_color};">%{prediction['risk_score']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Güven Skoru</div>
            <div class="metric-value" style="color: #00C48C;">%{prediction['probability_no_default']*100:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Kredi Limiti</div>
            <div class="metric-value" style="color: #2E7BFF;">{summary['limit_bal']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Limit Kullanım Oranı</div>
            <div class="metric-value" style="color: #F0B90B;">{summary['utilization']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk Gauge Chart ──
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown('<div class="section-header">📊 Risk Seviyesi Göstergesi</div>', unsafe_allow_html=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction["risk_score"],
            number={"suffix": "%", "font": {"size": 48, "color": "#E8ECF1", "family": "Inter"}},
            title={"text": "Kredi Risk Oranı", "font": {"size": 16, "color": "#8899AA", "family": "Inter"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#2A3F5F",
                         "tickfont": {"color": "#8899AA"}},
                "bar": {"color": risk_color, "thickness": 0.75},
                "bgcolor": "rgba(26,41,66,0.5)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,196,140,0.12)"},
                    {"range": [30, 50], "color": "rgba(255,165,2,0.12)"},
                    {"range": [50, 100], "color": "rgba(255,71,87,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#F0B90B", "width": 3},
                    "thickness": 0.8,
                    "value": prediction["risk_score"],
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=60, b=20, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-header">👤 Müşteri Profili</div>', unsafe_allow_html=True)

        profile_items = [
            ("Müşteri ID", f"#{summary['id']}"),
            ("Yaş", f"{summary['age']}"),
            ("Cinsiyet", summary['sex']),
            ("Eğitim", summary['education']),
            ("Medeni Durum", summary['marriage']),
            ("Gecikme Geçmişi", summary['has_delay']),
            ("Gecikme Ay Sayısı", f"{summary['delay_months']}"),
            ("Maks Gecikme", f"{summary['max_delay']} ay"),
            ("Ort. Fatura", summary['avg_bill']),
            ("Ort. Ödeme", summary['avg_payment']),
        ]

        for i in range(0, len(profile_items), 2):
            c1, c2 = st.columns(2)
            with c1:
                label, value = profile_items[i]
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-label">{label}</div>
                    <div class="info-value">{value}</div>
                </div>
                """, unsafe_allow_html=True)
            if i + 1 < len(profile_items):
                with c2:
                    label, value = profile_items[i + 1]
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-label">{label}</div>
                        <div class="info-value">{value}</div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Karar Gerekçeleri & Etki Analizi ──
    shap_col, feature_col = st.columns([1.3, 1])

    with shap_col:
        st.markdown('<div class="section-header">📊 Karar Gerekçeleri ve Etki Analizi</div>', unsafe_allow_html=True)

        matplotlib.use("Agg")
        fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
        fig_shap.patch.set_facecolor("#0A1628")
        ax_shap.set_facecolor("#0A1628")

        # Top 10 feature
        feature_names = shap_result["feature_names"]
        shap_vals = shap_result["shap_values"]
        abs_shap = np.abs(shap_vals)
        top_indices = np.argsort(abs_shap)[-10:][::-1]

        top_features_raw = [feature_names[i] for i in top_indices]
        top_features_friendly = [get_friendly_feature_name(f) for f in top_features_raw]
        top_shap_vals = [shap_vals[i] for i in top_indices]

        colors = ["#FF4757" if v > 0 else "#00C48C" for v in top_shap_vals]

        bars = ax_shap.barh(
            range(len(top_features_friendly)),
            top_shap_vals,
            color=colors,
            height=0.6,
            edgecolor="none",
        )
        ax_shap.set_yticks(range(len(top_features_friendly)))
        ax_shap.set_yticklabels(top_features_friendly, fontsize=9, color="#E8ECF1", fontfamily="sans-serif")
        ax_shap.invert_yaxis()
        ax_shap.set_xlabel("Karara Etki Gücü", fontsize=10, color="#8899AA")
        ax_shap.axvline(x=0, color="#2A3F5F", linewidth=0.8, linestyle="--")
        ax_shap.tick_params(colors="#8899AA")
        for spine in ax_shap.spines.values():
            spine.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_shap)
        plt.close(fig_shap)

        st.caption("🔴 Kırmızı: Kredi riskini artıran etkenler | 🟢 Yeşil: Güven artıran olumlu etkenler")

    with feature_col:
        st.markdown('<div class="section-header">⚡ En Belirleyici Faktörler</div>', unsafe_allow_html=True)

        st.markdown("**🔴 Riski Artıran Faktörler:**")
        if shap_result["top_risk_features"]:
            for f in shap_result["top_risk_features"]:
                friendly_name = get_friendly_feature_name(f['feature'])
                st.markdown(f"""
                <div class="factor-item" style="border-left: 3px solid #FF4757;">
                    <div class="info-label">{friendly_name}</div>
                    <div class="info-value" style="font-size: 0.95rem;">
                        Değer: {f['value']:,.1f} 
                        <span style="color: #FF4757; font-size: 0.8rem; margin-left: 6px;">⚠️ Riski Yükseltiyor</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Belirgin bir risk artırıcı faktör bulunmamaktadır.")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("**🟢 Olumlu / Güven Artıran Faktörler:**")
        if shap_result["top_safe_features"]:
            for f in shap_result["top_safe_features"]:
                friendly_name = get_friendly_feature_name(f['feature'])
                st.markdown(f"""
                <div class="factor-item" style="border-left: 3px solid #00C48C;">
                    <div class="info-label">{friendly_name}</div>
                    <div class="info-value" style="font-size: 0.95rem;">
                        Değer: {f['value']:,.1f} 
                        <span style="color: #00C48C; font-size: 0.8rem; margin-left: 6px;">✅ Güven Sağlıyor</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("Belirgin bir olumlu faktör bulunmamaktadır.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Yapay Zeka Risk & Kredi Değerlendirme Raporu ──
    st.markdown('<div class="section-header">💡 Yapay Zeka Kredi Değerlendirme Raporu</div>', unsafe_allow_html=True)

    cache_key = f"ai_report_{summary['id']}_{prediction['risk_score']:.1f}"
    if cache_key not in st.session_state:
        with st.spinner("🧠 Kapsamlı kredi değerlendirme raporu hazırlanıyor..."):
            st.session_state[cache_key] = generate_risk_explanation(prediction, shap_result, summary)

    explanation, model_name = st.session_state[cache_key]

    st.markdown(f"""
    <div class="ai-insight">
        <h3>Uzman Risk Analizi & Kredi Tavsiyesi</h3>
        {explanation}
        <div class="ai-model-badge">🤖 Akıllı Karar Destek Asistanı</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Hoş Geldin Ekranı ──
    st.markdown("<br>", unsafe_allow_html=True)

    col_welcome_l, col_welcome_c, col_welcome_r = st.columns([1, 2, 1])
    with col_welcome_c:
        st.markdown("""
        <div style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏦</div>
            <h2 style="color: #E8ECF1; font-weight: 700; margin-bottom: 0.5rem;">
                Hoş Geldiniz
            </h2>
            <p style="color: #8899AA; font-size: 1rem; max-width: 480px; margin: 0 auto 2rem;">
                Kredi başvurusu değerlendirmek için sol menüden bir 
                <strong style="color: #2E7BFF;">Müşteri ID</strong> seçin ve 
                <strong style="color: #F0B90B;">Analiz Et</strong> butonuna tıklayın.
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">⚡</div>
                    <div style="color: #8899AA; font-size: 0.85rem; font-weight: 500;">Anlık Risk Skoru</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">📊</div>
                    <div style="color: #8899AA; font-size: 0.85rem; font-weight: 500;">Karar Gerekçeleri</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5rem; margin-bottom: 0.3rem;">💡</div>
                    <div style="color: #8899AA; font-size: 0.85rem; font-weight: 500;">Uzman Risk Raporu</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; padding: 1rem; border-top: 1px solid rgba(46,123,255,0.1);">
    <p style="color: #556677; font-size: 0.75rem; margin: 0;">
        🏦 Kredi Risk Karar Destek Sistemi — Akıllı Bankacılık Çözümleri
    </p>
</div>
""", unsafe_allow_html=True)
