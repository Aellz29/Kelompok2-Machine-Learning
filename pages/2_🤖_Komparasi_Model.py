import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Model Performance Comparison")
st.write("Halaman ini menyajikan perbandingan performa antara model **Random Forest** dan **Decision Tree**.")

st.markdown("---")

# 1. Menyiapkan Data Performa Final (Hasil Gabungan)
data = {
    'Model': ['Random Forest Regressor', 'Decision Tree Regressor'],
    'R² Score (Akurasi)': [0.4704, 0.3520],  # Angka akurasi final
    'MAE (Rata-rata Eror)': [4.26, 5.15]     # Angka MAE final
}
df_performa = pd.DataFrame(data)

# 2. Menampilkan Tabel Ringkasan Evaluasi
st.subheader("📋 Tabel Metrik Evaluasi")
st.write("Metrik di bawah ini menunjukkan seberapa baik model dalam memprediksi Engagement Rate:")

# Menampilkan dataframe dengan format persentase dan desimal yang rapi
st.dataframe(df_performa.style.format({
    'R² Score (Akurasi)': '{:.2%}',
    'MAE (Rata-rata Eror)': '{:.2f}'
}), use_container_width=True)

st.markdown("---")

# 3. Menampilkan Grafik Batang Perbandingan Akurasi
st.subheader("📈 Visualisasi Perbandingan Akurasi")

# Membuat grafik menggunakan Matplotlib & Seaborn
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(x='Model', y='R² Score (Akurasi)', data=df_performa, hue='Model', palette='Set2', legend=False, ax=ax)

# Menambahkan label angka di atas setiap batang grafik
for p in ax.patches:
    ax.annotate(f"{p.get_height()*100:.2f}%", 
                (p.get_x() + p.get_width() / 2., p.get_height() + 0.02), 
                ha='center', va='center', xytext=(0, 5), 
                textcoords='offset points', fontweight='bold')
                
ax.set_ylabel("Skor $R^2$ (Akurasi)")
ax.set_ylim(0, 1.0) # Rentang grafik dari 0 sampai 100%

# Render grafik ke dalam aplikasi Streamlit
st.pyplot(fig)

st.markdown("---")

# 4. Kesimpulan Analisis
st.subheader("💡 Kesimpulan Eksperimen")
st.success(
    "Berdasarkan hasil pengujian, model **Random Forest Regressor** unggul secara signifikan dengan "
    "akurasi mencapai **47.04%** dan tingkat eror paling rendah (MAE 4.26). "
    "Hal ini membuktikan bahwa pendekatan *ensemble learning* (menggabungkan banyak pohon keputusan) "
    "jauh lebih stabil dan kuat dalam memprediksi algoritma media sosial dibanding satu pohon tunggal (Decision Tree)."
)