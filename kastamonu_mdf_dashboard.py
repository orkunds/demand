"""
MDF — Satış Talep Tahmin Dashboard
=====================================================
Modeller: XGBoost | ARIMA | SARIMA | Prophet | LSTM
Kurulum : pip install streamlit pandas numpy matplotlib plotly scikit-learn
           xgboost statsmodels neuralprophet torch openpyxl
Çalıştır: streamlit run kastamonu_mdf_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MDF Talep Tahmin Paneli",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Renk paleti ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#1B5E20",
    "secondary": "#2E7D32",
    "accent": "#66BB6A",
    "bg": "#F1F8E9",
    "text": "#1B5E20",
    "xgb": "#E65100",
    "arima": "#1565C0",
    "sarima": "#6A1B9A",
    "prophet": "#AD1457",
    "lstm": "#00695C",
    "actual": "#1B5E20",
}

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar Arka Planı */
/* 1. Arka planı daha açık ve canlı yapıyoruz */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), 
                    url("https://www.izu.edu.tr/images/default-source/aday/izu-de-yasam/sosyal-alanlar.jpg?sfvrsn=e908e763_4");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

/* Sidebar içindeki TÜM metinleri (label, p, span) beyaza zorla */
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .stMarkdown p, 
[data-testid="stSidebar"] span { 
    color: black !important; 
}

/* Dosya yükleme kutusunun içindeki metinleri SİYAH yapar */
[data-testid="stSidebar"] .stFileUploader section p, 
[data-testid="stSidebar"] .stFileUploader section span {
    color: black !important;
}

/* "Browse files" butonunun içindeki metni siyah yapar */
[data-testid="stSidebar"] .stFileUploader button p {
    color: black !important;
}
/* Dosya yükleyici metinleri için özel düzeltme */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: white !important;
}

/* Sol taraftaki sidebar'ın arka planını değiştirir */
[data-testid="stSidebar"] {
    background-color: #C8E6C9 !important; /* Koyu yeşil yapar */
    /* background-color: #121212 !important; -> Siyah yapmak istersen bunu aç */
/* Sayfa başlığını beyaz yap */
h1 { color: white !important; }

/* Kart başlıkları yeşil kalsın */
.metric-card h3 { color: #1B5E20 !important; }
.metric-card h1 { color: #2E7D32 !important; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data(path):
    df = pd.read_excel(path, sheet_name="Talep_Analizi", header=1)
    df["Tarih"] = pd.to_datetime(df["Tarih_Pazartesi"])
    return df

@st.cache_data
def load_uretim(path):
    df = pd.read_excel(path, sheet_name="Uretim_Takibi", header=1)
    df["Tarih"] = pd.to_datetime(df["Tarih_Pazartesi"])
    return df

@st.cache_data
def load_stok(path):
    df = pd.read_excel(path, sheet_name="Bitirmis_Urun_Stok", header=1)
    df["Tarih"] = pd.to_datetime(df["Tarih_Pazartesi"])
    return df

# ── Üst başlık ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#1B5E20,#388E3C);
     padding:20px 30px;border-radius:10px;margin-bottom:20px;'>
  <h1 style='color:white;margin:0;font-size:28px;'>
    🌲 MDF Satış Talep Tahmin Paneli
  </h1>
  <p style='color:#C8E6C9;margin:4px 0 0 0;font-size:14px;'>
    2023–2024 Entegre Veri Seti | XGBoost · ARIMA · SARIMA · Prophet · LSTM
  </p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR — KONTROLLER
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    
    uploaded = st.file_uploader("📂 Excel Dosyası Yükle", type=["xlsx"])
    
    st.markdown("---")
    st.markdown("### 🔍 Filtreler")
    
    # Yükleme işlemi
    DATA_PATH = uploaded if uploaded else "KE_MDF_Entegre_Veri_Seti_2023_2024.xlsx"
    
    try:
        df_raw = load_data(DATA_PATH)
        uretim_df = load_uretim(DATA_PATH)
        stok_df   = load_stok(DATA_PATH)
        data_ok = True
    except Exception as e:
        st.error(f"Veri yüklenemedi: {e}")
        st.info("Soldaki alandan Excel dosyasını yükleyiniz.")
        st.stop()
    
    urun_tipleri = ["Tümü"] + sorted(df_raw["Urun_Tipi"].unique().tolist())
    seç_urun     = st.selectbox("Ürün Tipi", urun_tipleri)
    
    kanallar     = ["Tümü"] + sorted(df_raw["Kanal"].unique().tolist())
    seç_kanal    = st.selectbox("Satış Kanalı", kanallar)
    
    segmentler   = ["Tümü"] + sorted(df_raw["Musteri_Segmenti"].unique().tolist())
    seç_segment  = st.selectbox("Müşteri Segmenti", segmentler)
    
    st.markdown("---")
    st.markdown("### 🤖 Model Seçimi")
    seç_modeller = st.multiselect(
        "Çalıştırılacak Modeller",
        ["XGBoost", "ARIMA", "SARIMA", "Prophet", "LSTM"],
        default=["XGBoost", "ARIMA", "SARIMA"],
    )
    
    tahmin_hafta = st.slider("Tahmin Haftası (gelecek)", 4, 26, 12)
    
    st.markdown("---")
    st.markdown("### 📊 Görünüm")
    sayfa = st.radio("Sayfa", ["📈 Özet KPI", "📉 Zaman Serisi", "🤖 Modeller", "🏭 Üretim & Stok"])

# ════════════════════════════════════════════════════════════════════════════
# VERİ FİLTRELEME
# ════════════════════════════════════════════════════════════════════════════
df = df_raw.copy()
if seç_urun    != "Tümü": df = df[df["Urun_Tipi"]        == seç_urun]
if seç_kanal   != "Tümü": df = df[df["Kanal"]            == seç_kanal]
if seç_segment != "Tümü": df = df[df["Musteri_Segmenti"] == seç_segment]

# Haftalık agregasyon
weekly = (
    df.groupby("Hafta")
      .agg(
          Satis=("Gerceklesen_Satis_m3", "sum"),
          Siparis=("Siparis_Miktari_m3", "sum"),
          Tahmin_Sistem=("Tahmin_Talep_m3", "sum"),
          Gelir=("Satis_Geliri_TRY", "sum"),
          Tarih=("Tarih", "first"),
      )
      .reset_index()
      .sort_values("Tarih")
)
weekly["Tarih"] = pd.to_datetime(weekly["Tarih"])

# ════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════════════════════════
def mae(a, p):   return np.mean(np.abs(a - p))
def rmse(a, p):  return np.sqrt(np.mean((a - p)**2))
def mape(a, p):  return np.mean(np.abs((a - p) / (a + 1e-9))) * 100

def future_dates(last_date, n):
    return pd.date_range(last_date + pd.Timedelta(weeks=1), periods=n, freq="W-MON")

def plot_forecast(series_actual, forecasts_dict, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series_actual.index, y=series_actual.values,
        name="Gerçekleşen", line=dict(color=COLORS["actual"], width=2.5),
        mode="lines"
    ))
    color_map = {"XGBoost": COLORS["xgb"], "ARIMA": COLORS["arima"],
                 "SARIMA": COLORS["sarima"], "Prophet": COLORS["prophet"],
                 "LSTM": COLORS["lstm"]}
    for name, (dates, vals) in forecasts_dict.items():
        fig.add_trace(go.Scatter(
            x=dates, y=vals, name=name,
            line=dict(color=color_map.get(name, "#999"), width=2, dash="dot"),
            mode="lines+markers", marker=dict(size=4)
        ))
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis_title="Tarih", yaxis_title="Satış (m³)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=450
    )
    return fig

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 1 — ÖZET KPI
# ════════════════════════════════════════════════════════════════════════════
if sayfa == "📈 Özet KPI":
    total_satis   = weekly["Satis"].sum()
    total_gelir   = weekly["Gelir"].sum()
    ort_hafta     = weekly["Satis"].mean()
    yoy_buro      = ((weekly.iloc[-1]["Satis"] - weekly.iloc[0]["Satis"])
                     / (weekly.iloc[0]["Satis"] + 1e-9) * 100)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <h3>📦 Toplam Satış (m³)</h3>
            <h1>{total_satis:,.0f}</h1>
            <p>Tüm dönem</p></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <h3>💰 Toplam Gelir (TRY)</h3>
            <h1>{total_gelir/1e9:.2f}B</h1>
            <p>Milyar TRY</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <h3>📊 Ortalama Haftalık (m³)</h3>
            <h1>{ort_hafta:,.0f}</h1>
            <p>Hafta başına</p></div>""", unsafe_allow_html=True)
    with c4:
        renk = "🟢" if yoy_buro > 0 else "🔴"
        st.markdown(f"""<div class='metric-card'>
            <h3>{renk} Trend Değişim</h3>
            <h1>{yoy_buro:+.1f}%</h1>
            <p>İlk → Son Hafta</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    
    # Aylık satış grafiği
    monthly = df.copy()
    monthly["AyYil"] = monthly["Tarih"].dt.to_period("M").astype(str)
    agg_monthly = monthly.groupby(["AyYil", "Urun_Tipi"])["Gerceklesen_Satis_m3"].sum().reset_index()
    fig_bar = px.bar(agg_monthly, x="AyYil", y="Gerceklesen_Satis_m3",
                     color="Urun_Tipi", title="Aylık Satış (m³) — Ürün Bazında",
                     labels={"Gerceklesen_Satis_m3": "Satış (m³)", "AyYil": "Ay"},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_bar.update_layout(template="plotly_white", height=380, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        # Kanal dağılımı
        kanal_agg = df.groupby("Kanal")["Gerceklesen_Satis_m3"].sum()
        fig_pie = px.pie(values=kanal_agg.values, names=kanal_agg.index,
                         title="Satış Kanalı Dağılımı",
                         color_discrete_sequence=["#1B5E20", "#66BB6A"])
        fig_pie.update_layout(height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_r:
        # Müşteri segmenti
        seg_agg = df.groupby("Musteri_Segmenti")["Gerceklesen_Satis_m3"].sum().sort_values()
        fig_h = px.bar(x=seg_agg.values, y=seg_agg.index, orientation="h",
                       title="Müşteri Segmenti Bazında Satış",
                       labels={"x": "Satış (m³)", "y": "Segment"},
                       color=seg_agg.values,
                       color_continuous_scale="Greens")
        fig_h.update_layout(height=320, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)

    # Korelasyon ısı haritası
    st.markdown("#### 🔗 Değişken Korelasyonları")
    num_cols = ["Tahmin_Talep_m3","Siparis_Miktari_m3","Gerceklesen_Satis_m3",
                "Satis_Fiyati_USD_m3","Doviz_Kuru_USDTRY","Mevsim_Katsayi",
                "Talep_Trend_Katsayi","Iade_Orani_Pct"]
    corr = df[num_cols].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdYlGn",
                         title="Korelasyon Matrisi", zmin=-1, zmax=1)
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 2 — ZAMAN SERİSİ
# ════════════════════════════════════════════════════════════════════════════
elif sayfa == "📉 Zaman Serisi":
    st.markdown("<div class='section-header'>📉 Zaman Serisi Analizi</div>", unsafe_allow_html=True)

    # Haftalık satış + sipariş + tahmin
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["Satis"],
                                name="Gerçekleşen Satış", line=dict(color=COLORS["actual"], width=2.5)))
    fig_ts.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["Siparis"],
                                name="Sipariş Miktarı", line=dict(color="#FB8C00", width=1.5, dash="dot")))
    fig_ts.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["Tahmin_Sistem"],
                                name="Sistem Tahmini", line=dict(color="#5C6BC0", width=1.5, dash="dash")))
    fig_ts.update_layout(title="Haftalık Satış — Sipariş — Sistem Tahmini (m³)",
                         xaxis_title="Tarih", yaxis_title="m³",
                         template="plotly_white", height=420,
                         legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_ts, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        # Hareketli ortalama
        weekly["MA4"]  = weekly["Satis"].rolling(4).mean()
        weekly["MA12"] = weekly["Satis"].rolling(12).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["Satis"],
                                    name="Gerçek", line=dict(color="#B0BEC5", width=1)))
        fig_ma.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["MA4"],
                                    name="4h Ort.", line=dict(color="#EF5350", width=2)))
        fig_ma.add_trace(go.Scatter(x=weekly["Tarih"], y=weekly["MA12"],
                                    name="12h Ort.", line=dict(color="#1B5E20", width=2)))
        fig_ma.update_layout(title="Hareketli Ortalama (4h / 12h)",
                              template="plotly_white", height=320,
                              xaxis_title="Tarih", yaxis_title="m³")
        st.plotly_chart(fig_ma, use_container_width=True)

    with col2:
        # Yıl bazında karşılaştırma (haftaya göre)
        df["Hafta_No"] = df["Tarih"].dt.isocalendar().week.astype(int)
        yil_cmp = df.groupby(["Yil", "Hafta_No"])["Gerceklesen_Satis_m3"].sum().reset_index()
        fig_yil = px.line(yil_cmp, x="Hafta_No", y="Gerceklesen_Satis_m3",
                          color="Yil", title="Yıl Karşılaştırması (Hafta Bazında)",
                          labels={"Hafta_No": "Hafta No", "Gerceklesen_Satis_m3": "Satış (m³)"},
                          color_discrete_map={2023: "#1B5E20", 2024: "#F57F17"})
        fig_yil.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_yil, use_container_width=True)

    # Mevsimsellik
    df["Ay_Ad"] = df["Tarih"].dt.month_name()
    mevsim = df.groupby(df["Tarih"].dt.month)["Gerceklesen_Satis_m3"].mean()
    ay_adlari = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
    fig_mev = go.Figure(go.Bar(x=ay_adlari, y=mevsim.values,
                                marker_color=[COLORS["accent"] if v >= mevsim.mean()
                                              else "#EF9A9A" for v in mevsim.values]))
    fig_mev.add_hline(y=mevsim.mean(), line_dash="dash",
                      annotation_text="Ortalama", line_color="grey")
    fig_mev.update_layout(title="Aylık Mevsimsel Ortalama Satış (m³)",
                          template="plotly_white", height=320,
                          xaxis_title="Ay", yaxis_title="Ort. Satış (m³)")
    st.plotly_chart(fig_mev, use_container_width=True)

    # Gelir box-plot
    df["Ceyrek"] = df["Tarih"].dt.to_period("Q").astype(str)
    fig_box = px.box(df, x="Ceyrek", y="Gerceklesen_Satis_m3",
                     color="Ceyrek", title="Çeyreklik Satış Dağılımı",
                     color_discrete_sequence=px.colors.sequential.Greens_r)
    fig_box.update_layout(template="plotly_white", height=360, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 3 — MODELLER
# ════════════════════════════════════════════════════════════════════════════
elif sayfa == "🤖 Modeller":
    st.markdown("<div class='section-header'>🤖 Tahmin Modelleri</div>", unsafe_allow_html=True)

    if not seç_modeller:
        st.warning("Lütfen sol panelden en az bir model seçin.")
        st.stop()

    # Zaman serisini hazırla
    ts = weekly.set_index("Tarih")["Satis"]
    ts.index = pd.DatetimeIndex(ts.index).to_period("W").to_timestamp()
    ts = ts.asfreq("W-MON")
    ts = ts.interpolate()

    train_size = int(len(ts) * 0.8)
    train = ts.iloc[:train_size]
    test  = ts.iloc[train_size:]
    future_idx = future_dates(ts.index[-1], tahmin_hafta)

    results      = {}   # model -> (test_pred, future_pred)
    metrics_dict = {}

    progress = st.progress(0, "Modeller çalıştırılıyor…")

    # ── XGBoost ─────────────────────────────────────────────────────────────
    if "XGBoost" in seç_modeller:
        try:
            from xgboost import XGBRegressor

            def make_features(series):
                df_f = pd.DataFrame({"y": series})
                df_f["lag1"]  = df_f["y"].shift(1)
                df_f["lag2"]  = df_f["y"].shift(2)
                df_f["lag4"]  = df_f["y"].shift(4)
                df_f["lag12"] = df_f["y"].shift(12)
                df_f["roll4"]  = df_f["y"].rolling(4).mean().shift(1)
                df_f["roll12"] = df_f["y"].rolling(12).mean().shift(1)
                df_f["week"]   = series.index.isocalendar().week.astype(int)
                df_f["month"]  = series.index.month
                df_f["quarter"] = series.index.quarter
                return df_f.dropna()

            df_feat = make_features(ts)
            X = df_feat.drop("y", axis=1)
            y = df_feat["y"]

            X_tr = X.iloc[:train_size - 12]
            y_tr = y.iloc[:train_size - 12]
            X_te = X.iloc[train_size - 12: train_size + len(test) - 12]
            y_te = y.iloc[train_size - 12: train_size + len(test) - 12]

            xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
            xgb.fit(X_tr, y_tr)

            test_pred_xgb  = xgb.predict(X_te)
            actual_aligned = y_te.values

            # Recursive future forecast
            history = list(ts.values)
            fut_xgb = []
            for i in range(tahmin_hafta):
                s_tmp = pd.Series(history, index=pd.date_range(ts.index[0],
                                  periods=len(history), freq="W-MON"))
                feat   = make_features(s_tmp).iloc[-1:].drop("y", axis=1)
                pred   = float(xgb.predict(feat)[0])
                fut_xgb.append(pred)
                history.append(pred)

            results["XGBoost"] = (
                pd.Series(test_pred_xgb, index=test.index[:len(test_pred_xgb)]),
                pd.Series(fut_xgb, index=future_idx)
            )
            metrics_dict["XGBoost"] = {
                "MAE":  mae(actual_aligned, test_pred_xgb),
                "RMSE": rmse(actual_aligned, test_pred_xgb),
                "MAPE": mape(actual_aligned, test_pred_xgb),
            }
        except ImportError:
            st.warning("xgboost kurulu değil: `pip install xgboost`")
        progress.progress(20)

    # ── ARIMA ───────────────────────────────────────────────────────────────
    if "ARIMA" in seç_modeller:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arima_model = ARIMA(train, order=(2, 1, 2)).fit()
            test_fc  = arima_model.forecast(steps=len(test))
            futr_fc  = arima_model.forecast(steps=len(test) + tahmin_hafta)[-tahmin_hafta:]
            results["ARIMA"] = (
                pd.Series(test_fc.values,   index=test.index),
                pd.Series(futr_fc.values,   index=future_idx)
            )
            metrics_dict["ARIMA"] = {
                "MAE":  mae(test.values, test_fc.values),
                "RMSE": rmse(test.values, test_fc.values),
                "MAPE": mape(test.values, test_fc.values),
            }
        except ImportError:
            st.warning("statsmodels kurulu değil: `pip install statsmodels`")
        progress.progress(40)

    # ── SARIMA ──────────────────────────────────────────────────────────────
    if "SARIMA" in seç_modeller:
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            sarima_model = SARIMAX(train, order=(1,1,1),
                                   seasonal_order=(1,1,0,52)).fit(disp=False)
            test_fc_s  = sarima_model.forecast(steps=len(test))
            futr_fc_s  = sarima_model.forecast(steps=len(test) + tahmin_hafta)[-tahmin_hafta:]
            results["SARIMA"] = (
                pd.Series(test_fc_s.values, index=test.index),
                pd.Series(futr_fc_s.values, index=future_idx)
            )
            metrics_dict["SARIMA"] = {
                "MAE":  mae(test.values, test_fc_s.values),
                "RMSE": rmse(test.values, test_fc_s.values),
                "MAPE": mape(test.values, test_fc_s.values),
            }
        except ImportError:
            st.warning("statsmodels kurulu değil: `pip install statsmodels`")
        progress.progress(60)

    # ── Prophet ─────────────────────────────────────────────────────────────
    if "Prophet" in seç_modeller:
        try:
            from prophet import Prophet
            prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                        changepoint_prior_scale=0.05)
            m.fit(prophet_df)
            future_pr = m.make_future_dataframe(periods=len(test) + tahmin_hafta, freq="W")
            forecast_pr = m.predict(future_pr)
            test_index_dt = (
    test.index.to_timestamp() 
    if hasattr(test.index, "to_timestamp") 
    else pd.to_datetime(test.index)
)
            test_yhat = forecast_pr[forecast_pr["ds"].isin(test_index_dt)]["yhat"].values
            fut_yhat  = forecast_pr.tail(tahmin_hafta)["yhat"].values
            test_len  = min(len(test), len(test_yhat))
            results["Prophet"] = (
                pd.Series(test_yhat[:test_len], index=test.index[:test_len]),
                pd.Series(fut_yhat,             index=future_idx)
            )
            if test_len > 0:
                       metrics_dict["Prophet"] = {
        "MAE":  mae(test.values[:test_len], test_yhat[:test_len]),
        "RMSE": rmse(test.values[:test_len], test_yhat[:test_len]),
        "MAPE": mape(test.values[:test_len], test_yhat[:test_len]),
    }
    
            else:
                       
                           st.warning("Prophet: Test tarihleri eşleşmedi, metrik hesaplanamadı.")

                       
        except ImportError:
            st.warning("prophet kurulu değil: `pip install prophet`")
        progress.progress(80)

    # ── LSTM ────────────────────────────────────────────────────────────────
    if "LSTM" in seç_modeller:
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import MinMaxScaler

            SEQ_LEN = 12

            scaler = MinMaxScaler()
            ts_scaled = scaler.fit_transform(ts.values.reshape(-1, 1)).flatten()

            def make_sequences(data, seq_len):
                X_out, y_out = [], []
                for i in range(len(data) - seq_len):
                    X_out.append(data[i:i+seq_len])
                    y_out.append(data[i+seq_len])
                return np.array(X_out), np.array(y_out)

            X_seq, y_seq = make_sequences(ts_scaled, SEQ_LEN)
            split = int(len(X_seq) * 0.8)
            X_tr_t = torch.FloatTensor(X_seq[:split]).unsqueeze(-1)
            y_tr_t = torch.FloatTensor(y_seq[:split]).unsqueeze(-1)
            X_te_t = torch.FloatTensor(X_seq[split:]).unsqueeze(-1)

            class LSTMModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True, dropout=0.2)
                    self.fc   = nn.Linear(64, 1)
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])

            model_lstm = LSTMModel()
            opt        = torch.optim.Adam(model_lstm.parameters(), lr=0.001)
            loss_fn    = nn.MSELoss()

            for _ in range(80):
                model_lstm.train()
                opt.zero_grad()
                pred_t = model_lstm(X_tr_t)
                loss   = loss_fn(pred_t, y_tr_t)
                loss.backward()
                opt.step()

            model_lstm.eval()
            with torch.no_grad():
                test_pred_lstm = scaler.inverse_transform(
                    model_lstm(X_te_t).numpy()
                ).flatten()

            test_idx_lstm = ts.index[split + SEQ_LEN: split + SEQ_LEN + len(test_pred_lstm)]
            actual_lstm   = ts.values[split + SEQ_LEN: split + SEQ_LEN + len(test_pred_lstm)]

            # Recursive future
            last_seq = ts_scaled[-SEQ_LEN:].tolist()
            fut_lstm = []
            for _ in range(tahmin_hafta):
                inp  = torch.FloatTensor(last_seq[-SEQ_LEN:]).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    p = float(model_lstm(inp).item())
                fut_lstm.append(p)
                last_seq.append(p)
            fut_lstm_inv = scaler.inverse_transform(
                np.array(fut_lstm).reshape(-1,1)).flatten()

            t_len = min(len(test), len(test_pred_lstm))
            results["LSTM"] = (
                pd.Series(test_pred_lstm[:t_len], index=test.index[:t_len]),
                pd.Series(fut_lstm_inv, index=future_idx)
            )
            metrics_dict["LSTM"] = {
                "MAE":  mae(actual_lstm[:t_len], test_pred_lstm[:t_len]),
                "RMSE": rmse(actual_lstm[:t_len], test_pred_lstm[:t_len]),
                "MAPE": mape(actual_lstm[:t_len], test_pred_lstm[:t_len]),
            }
        except ImportError:
            st.warning("PyTorch kurulu değil: `pip install torch`")
        progress.progress(100)

    progress.empty()

    if not results:
        st.error("Hiçbir model çalıştırılamadı. Kütüphaneleri kontrol edin.")
        st.stop()

    # ── Metrik tablosu ──────────────────────────────────────────────────────
    st.markdown("#### 📊 Model Performans Karşılaştırması (Test Seti)")
    met_df = pd.DataFrame(metrics_dict).T.round(2)
    met_df.index.name = "Model"

    if "MAPE" in met_df.columns and met_df["MAPE"].notna().any():
        best_model = met_df["MAPE"].dropna().idxmin()
    else:
        best_model = None

    met_df["⭐"] = met_df.index.map(lambda m: "✅ En İyi" if m == best_model else "")
    met_df = met_df.reset_index()

    st.dataframe(
        met_df.style.highlight_min(
        subset=[c for c in ["MAPE", "RMSE", "MAE"] if c in met_df.columns],
        color="#C8E6C9"
    ),
        use_container_width=True
)
    

    # ── Test tahmin grafiği ──────────────────────────────────────────────────
    st.markdown("#### 📈 Test Seti Tahminleri (Gerçek vs Tahmin)")
    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test.index, y=test.values,
                                  name="Gerçekleşen", line=dict(color=COLORS["actual"], width=3)))
    color_map = {"XGBoost": COLORS["xgb"], "ARIMA": COLORS["arima"],
                 "SARIMA": COLORS["sarima"], "Prophet": COLORS["prophet"],
                 "LSTM": COLORS["lstm"]}
    for name, (test_p, _) in results.items():
        fig_test.add_trace(go.Scatter(x=test_p.index, y=test_p.values,
                                      name=name, line=dict(color=color_map.get(name, "#999"),
                                                           width=2, dash="dot")))
    fig_test.update_layout(title="Test Seti — Model Tahminleri (m³)",
                           template="plotly_white", height=420,
                           xaxis_title="Tarih", yaxis_title="m³",
                           legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_test, use_container_width=True)

    # ── Gelecek tahmin grafiği ───────────────────────────────────────────────
    st.markdown(f"#### 🔮 Gelecek {tahmin_hafta} Haftalık Tahminler")
    fig_fut = go.Figure()
    fig_fut.add_trace(go.Scatter(x=ts.index[-26:], y=ts.values[-26:],
                                 name="Tarihsel (son 26h)",
                                 line=dict(color=COLORS["actual"], width=2.5)))
    for name, (_, fut_p) in results.items():
        fig_fut.add_trace(go.Scatter(x=fut_p.index, y=fut_p.values,
                                     name=f"{name} Tahmin",
                                     line=dict(color=color_map.get(name, "#999"),
                                               width=2, dash="dot"),
                                     mode="lines+markers", marker=dict(size=5)))
    fig_fut.add_vrect(x0=ts.index[-1], x1=future_idx[-1],
                      fillcolor="rgba(0,128,0,0.05)", line_width=0,
                      annotation_text="Tahmin Bölgesi", annotation_position="top left")
    fig_fut.update_layout(title=f"Gelecek {tahmin_hafta} Hafta Tahminleri (m³)",
                          template="plotly_white", height=450,
                          xaxis_title="Tarih", yaxis_title="m³",
                          legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_fut, use_container_width=True)

    # ── Tahmin tablosu ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Gelecek Tahmin Tablosu")
    fut_df = pd.DataFrame({"Tarih": future_idx})
    for name, (_, fut_p) in results.items():
        fut_df[name] = fut_p.values.round(1)
    fut_df["Ortalama"] = fut_df[[n for n in results]].mean(axis=1).round(1)
    st.dataframe(fut_df.style.format({c: "{:,.0f}" for c in fut_df.columns if c != "Tarih"}),
                 use_container_width=True)

    # ── Hata dağılımı ───────────────────────────────────────────────────────
    st.markdown("#### 📉 Model Hata Karşılaştırması (Bar)")
    fig_met = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "MAPE (%)"])
    names = list(metrics_dict.keys())
    colors_list = [color_map.get(n, "#999") for n in names]
    for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        vals = [metrics_dict[n][metric] for n in names]
        fig_met.add_trace(go.Bar(x=names, y=vals, marker_color=colors_list,
                                  showlegend=False), row=1, col=i+1)
    fig_met.update_layout(height=320, template="plotly_white",
                          title="Model Hata Metrikleri Karşılaştırması")
    st.plotly_chart(fig_met, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# SAYFA 4 — ÜRETİM & STOK
# ════════════════════════════════════════════════════════════════════════════
elif sayfa == "🏭 Üretim & Stok":
    st.markdown("<div class='section-header'>🏭 Üretim & Stok Analizi</div>", unsafe_allow_html=True)

    # Üretim tablosu
    uret_cols = [c for c in uretim_df.columns if c not in ["Tarih_Pazartesi"]]
    uret_weekly = uretim_df.groupby("Hafta").agg(
        Planlanan=("Planlanan_Kapasite_m3", "sum"),
        Gerceklesen=("Net_Uretim_m3", "sum"),
        OEE=("OEE_Pct", "mean"),
        Tarih=("Tarih", "first")
    ).reset_index().sort_values("Tarih")

    c1, c2, c3 = st.columns(3)
    with c1:
        ort_oee = uret_weekly["OEE"].mean()
        st.markdown(f"""<div class='metric-card'>
            <h3>⚙️ Ortalama OEE</h3>
            <h1>{ort_oee:.1f}%</h1>
            <p>Overall Equipment Effectiveness</p></div>""", unsafe_allow_html=True)
    with c2:
        toplam_plan = uret_weekly["Planlanan"].sum()
        toplam_grc  = uret_weekly["Gerceklesen"].sum()
        eff = toplam_grc / toplam_plan * 100
        st.markdown(f"""<div class='metric-card'>
            <h3>🎯 Üretim Verimliliği</h3>
            <h1>{eff:.1f}%</h1>
            <p>Gerçekleşen / Planlanan</p></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <h3>📦 Toplam Üretim (m³)</h3>
            <h1>{toplam_grc/1000:.1f}K</h1>
            <p>Tüm dönem</p></div>""", unsafe_allow_html=True)

    # Üretim zaman serisi
    fig_uret = go.Figure()
    fig_uret.add_trace(go.Scatter(x=uret_weekly["Tarih"], y=uret_weekly["Planlanan"],
                                   name="Planlanan", line=dict(color="#1565C0", width=2, dash="dash")))
    fig_uret.add_trace(go.Scatter(x=uret_weekly["Tarih"], y=uret_weekly["Gerceklesen"],
                                   name="Gerçekleşen", line=dict(color=COLORS["actual"], width=2.5)))
    fig_uret.update_layout(title="Haftalık Üretim: Planlanan vs Gerçekleşen (m³)",
                            template="plotly_white", height=380,
                            xaxis_title="Tarih", yaxis_title="m³")
    st.plotly_chart(fig_uret, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig_oee = px.line(uret_weekly, x="Tarih", y="OEE",
                          title="Haftalık OEE (%)",
                          color_discrete_sequence=["#E65100"])
        fig_oee.add_hline(y=85, line_dash="dash", annotation_text="Hedef %85",
                          line_color="green")
        fig_oee.update_layout(template="plotly_white", height=310)
        st.plotly_chart(fig_oee, use_container_width=True)

    with col_r2:
        stok_weekly = stok_df.groupby("Hafta").agg(
            Stok=("Stok_Kapanis_m3", "sum"),
            Tarih=("Tarih", "first")
        ).reset_index().sort_values("Tarih")
        fig_stok = px.area(stok_weekly, x="Tarih", y="Stok",
                           title="Haftalık Bitmiş Ürün Stoku (m³)",
                           color_discrete_sequence=["#66BB6A"])
        fig_stok.update_layout(template="plotly_white", height=310)
        st.plotly_chart(fig_stok, use_container_width=True)

    # Satış - Üretim karşılaştırma
    sales_w = weekly[["Tarih","Satis"]].set_index("Tarih")
    prod_w  = uret_weekly[["Tarih","Gerceklesen"]].set_index("Tarih")
    comb    = sales_w.join(prod_w, how="inner").reset_index()
    comb.columns = ["Tarih","Satış (m³)","Üretim (m³)"]
    comb_m  = comb.melt("Tarih", var_name="Tip", value_name="m³")

    fig_sp = px.line(comb_m, x="Tarih", y="m³", color="Tip",
                     title="Satış vs Üretim Karşılaştırması (m³)",
                     color_discrete_map={"Satış (m³)": "#1B5E20", "Üretim (m³)": "#1565C0"})
    fig_sp.update_layout(template="plotly_white", height=370)
    st.plotly_chart(fig_sp, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:12px;'>"
    "Kastamonu Entegre MDF | Satış Talep Tahmin Paneli "
    "</p>",
    unsafe_allow_html=True
)
