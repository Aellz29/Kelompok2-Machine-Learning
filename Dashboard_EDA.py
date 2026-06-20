import streamlit as st
import pandas as pd

# Judul Dashboard
st.title("📊 Dashboard Exploratory Data Analysis (EDA)")

# Membaca Data
df = pd.read_csv("data_cleaned.csv")

# Preview Data
st.subheader("Preview Dataset")
st.dataframe(df.head())

# ====================================
# Grafik 1 - Tren Engagement per Jam
# ====================================

st.subheader("📈 Tren Rata-rata Engagement Rate Berdasarkan Jam")

hourly_engagement = (
    df.groupby("Hour_of_Day")["Engagement_Rate"]
      .mean()
)

st.line_chart(hourly_engagement)

# ====================================
# Grafik 2 - Engagement per Content Type
# ====================================

st.subheader("📊 Rata-rata Engagement Rate Berdasarkan Jenis Konten")

content_engagement = (
    df.groupby("Content_Type")["Engagement_Rate"]
      .mean()
      .sort_values(ascending=False)
)

st.bar_chart(content_engagement)

# ====================================
# Grafik 3 - Engagement per Platform
# ====================================

st.subheader("📱 Rata-rata Engagement Rate Berdasarkan Platform")

platform_engagement = (
    df.groupby("Platform")["Engagement_Rate"]
      .mean()
      .sort_values(ascending=False)
)

st.bar_chart(platform_engagement)

# ====================================
# Kesimpulan
# ====================================

st.subheader("📝 Insight Singkat")

st.write("""
Dashboard ini menampilkan pola engagement pengguna media sosial berdasarkan waktu posting,
jenis konten, dan platform yang digunakan. Grafik membantu mengidentifikasi waktu terbaik
untuk posting serta jenis konten yang menghasilkan engagement tertinggi.
""")