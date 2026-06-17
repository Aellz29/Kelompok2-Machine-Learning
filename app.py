import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. SETUP HALAMAN & JUDUL APP
# ==========================================
st.set_page_config(page_title="Prediksi Engagement Sosmed", layout="centered")
st.title("📊 Social Media Engagement Predictor")
st.subheader("Aplikasi Prediksi Waktu Terbaik & Optimasi Posting Konten")

# ==========================================
# 2. LOAD & PREPROCESS DATA (Fungsi Cepat)
# ==========================================
@st.cache_data # Biar aplikasi cepet, data di-cache
def load_and_prep_data():
    # Load dataset
    df = pd.read_csv('data/social_media_engagement_dataset.csv')
    
    # Mapping Hari
    hari_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Day_of_Week_Num'] = df['Day_of_Week'].map(hari_map)
    
    # Fitur-fitur
    fitur_numerik = ['Hour_of_Day', 'Day_of_Week_Num', 'Hashtag_Count', 'Content_Length', 'Follower_Count']
    fitur_kategori = ['Platform', 'Content_Type', 'Category', 'Sentiment']
    
    X_kat = pd.get_dummies(df[fitur_kategori], drop_first=True)
    X_num = df[fitur_numerik]
    X = pd.concat([X_num, X_kat], axis=1)
    y = df['Engagement_Rate']
    
    return X, y, fitur_kategori

X, y, fitur_kategori = load_and_prep_data()

# ==========================================
# 3. TRAIN MODEL (Baseline Model Lu Tadi)
# ==========================================
@st.cache_resource # Biar model ga dilatih ulang setiap user ngeklik tombol
def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model(X, y)

# ==========================================
# 4. SIDEBAR INPUT UNTUK USER
# ==========================================
st.sidebar.header("📥 Input Parameter Postingan")

# Input Waktu (Krusial buat judul projek lu!)
input_jam = st.sidebar.slider("Jam Posting (Hour of Day)", 0, 23, 12)
input_hari = st.sidebar.selectbox("Hari Posting", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
hari_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# Input Metadata Konten
input_followers = st.sidebar.number_input("Jumlah Followers", min_value=0, value=10000, step=1000)
input_hashtags = st.sidebar.slider("Jumlah Hashtag", 0, 30, 5)
input_length = st.sidebar.slider("Panjang Konten (Karakter)", 10, 500, 150)

# Input Kategori (Dropdown)
input_platform = st.sidebar.selectbox("Platform", ['Instagram', 'TikTok', 'YouTube', 'Twitter', 'Facebook']) # Sesuaikan dengan dataset lu
input_type = st.sidebar.selectbox("Tipe Konten", ['Video', 'Image', 'Text', 'Carousel'])
input_category = st.sidebar.selectbox("Kategori Konten", ['Entertainment', 'Education', 'Lifestyle', 'Tech', 'Fashion'])
input_sentiment = st.sidebar.selectbox("Sentimen Konten", ['Positive', 'Neutral', 'Negative'])

# ==========================================
# 5. PROSES PREDIKSI
# ==========================================
if st.button("🔮 Prediksi Engagement Rate"):
    
    # Bikin dataframe kosong dengan struktur kolom yang sama persis dengan X latihan
    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0 # isi awal dengan angka 0
    
    # Masukkan nilai input numerik
    input_data['Hour_of_Day'] = input_jam
    input_data['Day_of_Week_Num'] = hari_map[input_hari]
    input_data['Follower_Count'] = input_followers
    input_data['Hashtag_Count'] = input_hashtags
    input_data['Content_Length'] = input_length
    
    # Masukkan nilai One-Hot Encoding untuk fitur kategori (jika kolomnya ada di X)
    col_platform = f"Platform_{input_platform}"
    col_type = f"Content_Type_{input_type}"
    col_category = f"Category_{input_category}"
    col_sentiment = f"Sentiment_{input_sentiment}"
    
    if col_platform in input_data.columns: input_data[col_platform] = 1
    if col_type in input_data.columns: input_data[col_type] = 1
    if col_category in input_data.columns: input_data[col_category] = 1
    if col_sentiment in input_data.columns: input_data[col_sentiment] = 1
    
    # Eksekusi Prediksi pake model Random Forest Lu
    prediksi = model.predict(input_data)[0]
    
    # Tampilkan Hasil di Layar Utama Streamlit
    st.balloons()
    st.success(f"### 🎉 Hasil Prediksi!")
    st.metric(label="Estimasi Engagement Rate yang akan didapat:", value=f"{prediksi:.2f}%")
    st.info("💡 *Tips untuk Tim:* Coba ubah parameter 'Jam Posting' dan 'Hari Posting' di sidebar untuk mencari kombinasi waktu terbaik dengan nilai prediksi tertinggi!")