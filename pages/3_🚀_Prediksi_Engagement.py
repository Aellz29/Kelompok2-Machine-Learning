import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Prediksi Engagement", page_icon="🚀", layout="wide")

st.title("🚀 Kalkulator Prediksi Engagement Rate")
st.write("Silahkan input detail data konten di bawah ini untuk memprediksi estimasi performa engagement.")
st.markdown("---")

# 1. MEMBUAT FORM INPUT (Layout 2 Kolom Kebanggaan Lu)
col1, col2 = st.columns(2)
with col1:
    platform = st.selectbox("Platform Media Sosial", ["Instagram", "TikTok", "YouTube"])
    followers = st.number_input("Jumlah Followers / Subscribers", min_value=0, value=1000, step=100)
    content_type = st.selectbox("Tipe Konten", ["Video/Reels", "Foto/Feeds", "Carousel"])
with col2:
    content_length = st.number_input("Panjang Konten (Karakter Caption / Detik Video)", min_value=0, value=150)
    hour_of_day = st.slider("Jam Posting (Format 24 Jam)", 0, 23, 19)

st.markdown("<br>", unsafe_allow_html=True)

# 2. PROSES EKSEKUSI PREDIKSI LANGSUNG
# Tombol Eksekusi
if st.button("🔥 Hitung Estimasi Engagement", use_container_width=True):
    import os
    import joblib
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    possible_paths = [
        os.path.join(root_dir, "models", "best_random_forest.pkl"),
        os.path.join(root_dir, "Kelompok2-Machine-Learning", "models", "best_random_forest.pkl"),
        os.path.join("models", "best_random_forest.pkl"),
        "best_random_forest.pkl"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
            
    try:
        if model_path is None:
            raise FileNotFoundError("File 'best_random_forest.pkl' tidak ditemukan. Pastikan posisi file ada di folder models/ ya!")

        # 1. Load model asli lu
        model = joblib.load(model_path)
        
        # 2. PROSES CONVERT STRING TO NUMERIC (Label Encoding Mapping)
        # Sesuai standar alfabetis coding biasanya:
        platform_map = {"Instagram": 0, "Tiktok": 1, "Youtube": 2}
        content_map = {"Carousel": 0, "Foto/Feeds": 1, "Video/Reels": 2}
        
        platform_numeric = platform_map.get(platform, 0)
        content_numeric = content_map.get(content_type, 0)
        
        # 3. KONDISIKAN STRUKTUR INPUT DATAFRAME
        if hasattr(model, "feature_names_in_"):
            fitur_model = model.feature_names_in_
            data_dict = {}
            for nama_kolom in fitur_model:
                if 'platform' in nama_kolom.lower():
                    data_dict[nama_kolom] = platform_numeric
                elif 'follower' in nama_kolom.lower() or 'sub' in nama_kolom.lower():
                    data_dict[nama_kolom] = followers
                elif 'type' in nama_kolom.lower() or 'tipe' in nama_kolom.lower():
                    data_dict[nama_kolom] = content_numeric
                elif 'length' in nama_kolom.lower() or 'panjang' in nama_kolom.lower():
                    data_dict[nama_kolom] = content_length
                elif 'hour' in nama_kolom.lower() or 'jam' in nama_kolom.lower():
                    data_dict[nama_kolom] = hour_of_day
                else:
                    data_dict[nama_kolom] = 0 
            
            input_data = pd.DataFrame([data_dict])[fitur_model]
        else:
            # Jika model tidak menyimpan nama kolom, buat dataframe dengan input angka langsung
            input_data = pd.DataFrame([{
                'platform': platform_numeric,
                'followers': followers,
                'content_type': content_numeric,
                'content_length': content_length,
                'hour_of_day': hour_of_day
            }])
        
        # 4. Eksekusi Prediksi menggunakan model asli
        prediction = model.predict(input_data)
        hasil_prediksi = f"{prediction[0]:.2f}%"
        
        # Tampilkan hasil ke UI
        st.success("🎯 Hasil Prediksi Berhasil Dihitung Berdasarkan Model Asli!")
        st.metric(
            label="Estimasi Engagement Rate", 
            value=hasil_prediksi, 
            delta="Berdasarkan Random Forest Final"
        )
        
    except Exception as e:
        st.error(f"Gagal melakukan prediksi otomatis.")
        st.code(f"Pesan Error Asli: {e}", language="bash")