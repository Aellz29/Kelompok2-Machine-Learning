import streamlit as st

# ==========================================
# 1. KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(
    page_title="Social Media Engagement Predictor",
    page_icon="📈",
    layout="wide"
)

# ==========================================
# 2. INJEKSI CUSTOM CSS (BIAR MAKIN MEWAH & ESTETIK)
# ==========================================
st.markdown("""
    <style>
        /* Efek kartu modern untuk metrik */
        div[data-testid="stMetricBlock"] {
            background-color: #1e293b;
            border: 1px solid #334155;
            padding: 20px 25px !important;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s;
        }
        div[data-testid="stMetricBlock"]:hover {
            transform: translateY(-2px);
            border-color: #38bdf8;
        }
        /* Styling teks metrik */
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        div[data-testid="stMetricValue"] {
            color: #38bdf8 !important;
            font-size: 32px !important;
            font-weight: 700 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR - NAVIGASI & INFORMASI TIM
# ==========================================
with st.sidebar:
    st.title("🚀 Navigasi Proyek")
    st.write("Gunakan menu di atas untuk berpindah halaman dan melihat fitur interaktif.")
    
    st.markdown("---")
    st.markdown("### 👥 Struktur Organisasi Tim:")
    st.markdown("- **Ailum M. L.** (Project Manager / Core Dev)")
    st.markdown("- **Mendy** (UI/UX & Visualisasi EDA)")
    st.markdown("- **Muhammad** (Analisis Data & PPT)")
    st.markdown("- **Ihsan Kamil** (Lead Machine Learning)")
    st.markdown("- **Fatih** (Model Evaluation & Testing)")
    st.markdown("- **Indah B.** (Data Cleaning Specialist)")
    st.markdown("- **Widya** (Feature Engineering Specialist)")
    
    st.markdown("---")
    st.info("💡 **Status Proyek:** Phase 2 (Integration & Multi-Page Migration)")

# ==========================================
# 4. KONTEN UTAMA (LANDING PAGE)
# ==========================================
# Menambahkan dekorasi badge html agar terlihat seperti dashboard enterprise
st.markdown("""
    <div style="display: flex; gap: 8px; margin-bottom: -10px;">
        <span style="background-color: #0369a1; color: #e0f2fe; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold;">PRODUCTION READY</span>
        <span style="background-color: #15803d; color: #dcfce7; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: bold;">ML ENGINE V2.4</span>
    </div>
""", unsafe_allow_html=True)

st.title("📱 Social Media Engagement Predictor Dashboard")
st.subheader("Optimasi Performa Konten Berbasis Machine Learning")

st.markdown("""
Selamat datang di aplikasi **Social Media Engagement Predictor**. Aplikasi ini dirancang untuk membantu para content creator, 
digital marketer, dan brand dalam memprediksi serta mengoptimalkan tingkat interaksi (*engagement rate*) dari postingan mereka 
sebelum konten tersebut diunggah ke media sosial.
""")

st.markdown("---")

# ==========================================
# 5. RINGKASAN PROGRESS & METRIK (DENGAN STYLE GLOW HOVER)
# ==========================================
st.subheader("📊 Ringkasan Eksekutif & Status Pengembangan")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Sistem Arsitektur", 
        value="Multi-Page v2", 
        delta="Migrasi Sukses"
    )

with col2:
    st.metric(
        label="Model Utama", 
        value="Random Forest", 
        delta="Optimized", 
        delta_color="normal"
    )

with col3:
    st.metric(
        label="Akurasi Model Terbaik (R²)", 
        value="47.04%", 
        delta="Naik dari 31.25% Baseline", 
        delta_color="normal"
    )

st.markdown("---")

# ==========================================
# 6. EXPANDER DENGAN STATUS TERLIPAT (expanded=False) ⚙️
# ==========================================
st.subheader("⚙️ Detail Informasi & Spesifikasi Model")

with st.expander("🔍 Klik untuk Meninjau Detail Jeroan Model Machine Learning", expanded=False):
    col_mod1, col_mod2 = st.columns(2)
    
    with col_mod1:
        st.markdown("### 📊 Parameter Hasil Tuning:")
        st.write("• **Algoritma Utama:** Random Forest Regressor")
        st.write("• **Akurasi Akhir (R² Score):** `47.04%` (Setelah Hyperparameter Tuning)")
        st.write("• **Akurasi Baseline:** `31.25%` (Sebelum Optimasi)")
        st.write("• **Fitur Input Utama (X):** `Follower_Count`, `Content_Length`, `Hour_of_Day`, `Platform`, `Content_Type`")
        st.write("• **Target Prediksi (y):** `Engagement_Rate` (Skala Persentase)")

    with col_mod2:
        st.markdown("### 🧠 Mengapa Random Forest?")
        st.write("""
        Algoritma **Random Forest** dipilih karena bekerja dengan metode *ensemble* (menggabungkan puluhan 
        pohon keputusan/Decision Trees) untuk membuat satu prediksi akhir yang kokoh. 
        
        Kelebihan utama model ini dalam kasus kelompok kami adalah kemampuannya menangani hubungan non-linear 
        antara jumlah pengikut (*followers*) dan panjang konten terhadap tingkat interaksi (*engagement*) 
        tanpa mengalami *overfitting* yang parah jika dibandingkan dengan Decision Tree tunggal.
        """)

st.markdown("---")

# ==========================================
# 7. PANDUAN PENGGUNAAN FITUR (MULTI-PAGE)
# ==========================================
st.subheader("💡 Panduan Navigasi Fitur Aplikasi")
st.write("Silakan lihat menu di sebelah kiri atau pilih halaman yang ingin diakses:")

col_page1, col_page2, col_page3 = st.columns(3)

with col_page1:
    st.markdown("### 1. 📊 Dashboard EDA")
    st.write("Berisi data statistik interaktif dan **diagram garis** tren *engagement* harian untuk melihat *prime time* upload konten.")
    st.caption("Dikerjakan oleh: Mendy & Muhammad")

with col_page2:
    st.markdown("### 2. 🤖 Komparasi Model")
    st.write("Menampilkan pembuktian ilmiah berupa tabel metrik dan **diagram batang** perbandingan antara Random Forest vs Decision Tree.")
    st.caption("Dikerjakan oleh: Ihsan & Fatih")

with col_page3:
    st.markdown("### 3. 🚀 Prediksi Engagement")
    st.write("Fitur utama berupa kalkulator prediksi interaktif untuk menghitung estimasi *engagement rate* secara *live*.")
    st.caption("Dikerjakan oleh: Ailum (Integrasi Core)")

# Footer
st.markdown("<br><br><hr><center><small>Informatics Engineering | Digitech University 2026</small></center>", unsafe_allow_html=True)