import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
# Menggunakan backend 'Agg' agar grafik langsung disimpan menjadi file gambar tanpa eror pop-up Windows
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=== 1. Memuat Dataset dan Model ===")

# ------------------------------------------------------------------------
# TUGAS FATIH: Ganti teks di bawah ini dengan lokasi ASLI di laptop kamu!
# Contoh: "C:/pml/Kelompok2-Machine-Learning/data/social_media_engagement_dataset.csv"
# ------------------------------------------------------------------------
data_path = "C:/MASUKKAN_ALAMAT_FOLDER_KAMU/data/social_media_engagement_dataset.csv"
model_path = "C:/MASUKKAN_ALAMAT_FOLDER_KAMU/models/best_random_forest.pkl"
# ------------------------------------------------------------------------

# Validasi manual agar kita tahu jalurnya sudah benar atau belum
if not os.path.exists(data_path):
    print(f"❌ SALAH ALAMAT: File CSV tidak ada di: {data_path}")
    print("Silakan cek kembali alamat folder kamu di baris kode nomor 18!")
    exit()

if not os.path.exists(model_path):
    print(f"❌ SALAH ALAMAT: File Model .pkl tidak ada di: {model_path}")
    print("Silakan cek kembali alamat folder kamu di baris kode nomor 19!")
    exit()
    
df = pd.read_csv(data_path)
best_model = joblib.load(model_path)
print("✅ Model dan Dataset BERHASIL dimuat!")

print("\n=== 2. Sinkronisasi Pemrosesan Data Awal ===")
df_clean = df.copy()

kolom_dibuang = ['Post_ID', 'Timestamp', 'User_ID']
for col in kolom_dibuang:
    if col in df_clean.columns:
        df_clean = df_clean.drop(columns=[col])
        
df_encoded = pd.get_dummies(df_clean, drop_first=True)

X = df_encoded.drop(columns=['Engagement_Rate'])
y = df_encoded['Engagement_Rate']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Ukuran Data Testing untuk Evaluasi: {X_test.shape}")

print("\n=== 3. Menjalankan Prediksi & Evaluasi Metrik ===")
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)
akurasi_persen = r2 * 100

print(f"Hasil Evaluasi Model:")
print(f"-> Mean Absolute Error (MAE)     : {mae:.2f}")
print(f"-> Mean Squared Error (MSE)       : {mse:.2f}")
print(f"-> Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"-> R² Score / Akurasi             : {akurasi_persen:.2f}%")

print("\n=== 4. Menyimpan Visualisasi Grafik ke File Gambar ===")

# Grafik 1: Perbandingan Aktual vs Hasil Prediksi
plt.figure(figsize=(10, 5))
plt.scatter(y_test, predictions, color='green', alpha=0.5, label='Prediksi vs Asli')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Garis Ideal')
plt.xlabel('Engagement Rate Sebenarnya')
plt.ylabel('Engagement Rate Hasil Prediksi')
plt.title('Evaluasi Model: Aktual vs Prediksi Model')
plt.legend()
plt.grid(True)
plt.savefig('grafik_aktual_vs_prediksi.png', dpi=300)
plt.close()
print("-> Sukses membuat gambar: grafik_aktual_vs_prediksi.png")

# Grafik 2: Fitur Paling Berpengaruh (Feature Importance)
plt.figure(figsize=(10, 5))
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices], y=X.columns[indices], palette='mako')
plt.title('Fitur Paling Berpengaruh terhadap Engagement Rate')
plt.xlabel('Skor Kepentingan')
plt.ylabel('Nama Fitur/Kolom')
plt.tight_layout()
plt.savefig('grafik_feature_importance.png', dpi=300)
plt.close()
print("-> Sukses membuat gambar: grafik_feature_importance.png")

print("\n[BERES] Jalankan perintah selesai! Cek folder kamu sekarang.")