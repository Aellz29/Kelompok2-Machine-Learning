import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. SETUP HALAMAN & JUDUL APP
# ==========================================
st.set_page_config(page_title="Prediksi Engagement Sosmed", layout="centered")
st.title("📊 Social Media Engagement Predictor")
st.subheader("Aplikasi Prediksi Waktu Terbaik & Optimasi Posting Konten")

# ==========================================
# 2. LOAD & PREPROCESS DATA (SINKRON DENGAN IHSAN)
# ==========================================
@st.cache_data
def load_and_prep_data():
    # Menggunakan dataset asli seperti yang dipakai Ihsan pas training
    data_path = "data/social_media_engagement_dataset.csv"
    if not os.path.exists(data_path):
        st.error(f"File {data_path} tidak ditemukan! Pastikan dataset ada di folder data/.")
        return None, None
        
    df = pd.read_csv(data_path)
    df_clean = df.copy()
    
    # Buang kolom teks unik seperti yang dilakukan di skrip Ihsan
    kolom_dibuang = ['Post_ID', 'Timestamp', 'User_ID']
    for col in kolom_dibuang:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            
    # One-Hot Encoding otomatis (Harus sama persis dengan skrip train Ihsan)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    
    # Pisahkan fitur (X) dan target (y)
    X = df_encoded.drop(columns=['Engagement_Rate'])
    y = df_encoded['Engagement_Rate']
    
    return X, y

X, y = load_and_prep_data()

# ==========================================
# 3. LOAD TUNED MODEL IHSAN KAMIL
# ==========================================
@st.cache_resource
def train_model(X, y):
    model_pkl_path = "models/best_random_forest.pkl"
    
    if os.path.exists(model_pkl_path):
        try:
            # Load model cerdas berakurasi 47.04% milik Ihsan
            model = joblib.load(model_pkl_path)
            return model
        except Exception as e:
            st.warning(f"Gagal me-load model .pkl milik Ihsan ({e}). Menggunakan baseline model.")
    
    # Backup jika file model pkl ga ketemu
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# ==========================================
# 4. SIDEBAR INPUT UNTUK USER
# ==========================================
st.sidebar.header("🎯 Input Parameter Konten")

input_platform = st.sidebar.selectbox("Platform Media Sosial", ['Instagram', 'Twitter', 'Facebook', 'LinkedIn', 'TikTok'])
input_type = st.sidebar.selectbox("Tipe Konten", ['Image', 'Video', 'Text', 'Carousel'])
input_category = st.sidebar.selectbox("Kategori Konten", ['Entertainment', 'Education', 'Tech', 'Fashion', 'Food', 'Lifestyle'])

# Tambahan fitur numerik yang ada di dataset latihan Ihsan
input_followers = st.sidebar.number_input("Jumlah Followers Akun", min_value=0, value=1000, step=100)
input_hashtags = st.sidebar.slider("Jumlah Hashtag yang Dipakai", min_value=0, max_value=20, value=5)
input_length = st.sidebar.slider("Panjang Karakter Caption", min_value=1, max_value=500, value=150)

# Input Waktu Posting
input_hari = st.sidebar.selectbox("Hari Posting", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
input_jam = st.sidebar.slider("Jam Posting (Format 24 Jam)", min_value=0, max_value=23, value=19)

# Tambahan input metrik engagement pendukung jika terdeteksi di model latihan
input_likes = st.sidebar.number_input("Target Likes (Estimasi awal)", min_value=0, value=50, step=5)
input_saves = st.sidebar.number_input("Target Saves (Estimasi awal)", min_value=0, value=10, step=5)
input_comments = st.sidebar.number_input("Target Comments (Estimasi awal)", min_value=0, value=5, step=1)
input_shares = st.sidebar.number_input("Target Shares (Estimasi awal)", min_value=0, value=5, step=1)

# ==========================================
# 5. PEMROSESAN PREDIKSI & OUTPUT UTAMA
# ==========================================
if st.sidebar.button("🚀 Prediksi Engagement Rate"):
    # Buat baris data baru dengan kolom yang diwajibkan oleh model latihan X
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0 # isi awal semua kolom dengan angka 0
    
    # Isi kolom numerik standar bawaan dataset
    if 'Hour_of_Day' in input_data.columns: input_data['Hour_of_Day'] = input_jam
    if 'Follower_Count' in input_data.columns: input_data['Follower_Count'] = input_followers
    if 'Hashtag_Count' in input_data.columns: input_data['Hashtag_Count'] = input_hashtags
    if 'Content_Length' in input_data.columns: input_data['Content_Length'] = input_length
    
    # Mengisi metrik engagement pendukung jika terdeteksi dilihat saat fit time
    if 'Likes' in input_data.columns: input_data['Likes'] = input_likes
    if 'Saves' in input_data.columns: input_data['Saves'] = input_saves
    if 'Comments' in input_data.columns: input_data['Comments'] = input_comments
    if 'Shares' in input_data.columns: input_data['Shares'] = input_shares
    
    # Isi kolom kategori hasil One-Hot Encoding (Set nilai ke 1 jika dipilih user)
    col_day = f"Day_of_Week_{input_hari}"
    col_platform = f"Platform_{input_platform}"
    col_type = f"Content_Type_{input_type}"
    col_category = f"Category_{input_category}"
    
    if col_day in input_data.columns: input_data[col_day] = 1
    if col_platform in input_data.columns: input_data[col_platform] = 1
    if col_type in input_data.columns: input_data[col_type] = 1
    if col_category in input_data.columns: input_data[col_category] = 1
    
    try:
        # Eksekusi Prediksi menggunakan model Ihsan Kamil yang presisi!
        prediksi = model.predict(input_data)[0]
        
        # Tampilkan Hasil Sukses
        st.balloons()
        st.success(f"### 🔥 Estimasi Engagement Rate Konten Lu: **{prediksi:.2f}%**")
        
        # Rekomendasi tambahan
        st.markdown("---")
        st.subheader("💡 Tips Optimasi Konten Tambahan:")
        if input_jam < 12:
            st.info("📌 Konten lu diposting pagi/siang hari. Coba tes posting di jam prima malam hari (18:00 - 21:00) untuk membandingkan peningkatan engagement secara organik.")
        else:
            st.info("📌 Waktu posting yang lu pilih sudah berada di jam prime time! Pertahankan konsistensi jadwal posting ini.")
            
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.text("Detail kolom yang dikirim ke model:")
        st.write(input_data)
else:
    st.info("👈 Silakan isi parameter konten di sidebar sebelah kiri, lalu klik tombol **Prediksi Engagement Rate** untuk melihat hasilnya!")