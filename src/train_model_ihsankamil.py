import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_and_tune_model():
    print("=== 1. Memuat Dataset ===")
    # Menyesuaikan dengan letak file data di folder kamu
    data_path = "data/social_media_engagement_dataset.csv" # <-- Sesuaikan nama file aslinya jika berbeda
    
    if not os.path.exists(data_path):
        print(f"Eror: File {data_path} tidak ditemukan! Periksa kembali folder data kamu.")
        return
        
    df = pd.read_csv(data_path)
    print(head_data := df.head())
    
    print("\n=== 2. Pemrosesan Data Awal (Pre-processing) ===")
    
    # 1. Buat salinan data agar data asli aman
    df_clean = df.copy()
    
    # 2. Buang kolom unik dan waktu yang tipenya teks agar tidak bikin eror
    # 'Post_ID' wajib dibuang karena isinya teks unik yang tidak bisa diangka-kan
    kolom_dibuang = ['Post_ID', 'Timestamp', 'User_ID']
    for col in kolom_dibuang:
        if col in df_clean.columns:
            df_clean = df_clean.drop(columns=[col])
            
    # 3. Mengubah semua kolom kategori tersisa (Platform, Content_Type, Category) menjadi angka
    # pd.get_dummies tanpa mengetik nama kolom akan otomatis mengubah semua kolom teks menjadi angka (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_clean, drop_first=True)
    
    # 4. Pisahkan fitur (X) dan target (y)
    X = df_encoded.drop(columns=['Engagement_Rate'])
    y = df_encoded['Engagement_Rate']
    
    # Memisahkan data menjadi Train set (80%) dan Test set (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Ukuran Data Training: {X_train.shape}")
    print(f"Ukuran Data Testing : {X_test.shape}")
    
    print("\n=== 3. Proses Hyperparameter Tuning (Menaikkan Akurasi) ===")
    print("Mohon tunggu, sedang mencari kombinasi parameter terbaik otomatis...")
    
    # Menentukan variasi parameter Random Forest yang akan diuji
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_base = RandomForestRegressor(random_state=42)
    
    # Menggunakan GridSearchCV untuk mencari parameter terbaik berdasarkan skor R2 (Akurasi)
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, 
                               cv=3, n_jobs=-1, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Kombinasi Parameter Terbaik: {grid_search.best_params_}")
    
    print("\n=== 4. Evaluasi Model Terbaik ===")
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    # Hitung metrik evaluasi
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    akurasi_persen = r2 * 100
    
    print(f"Hasil Akhir Model Setelah Tuning:")
    print(f"-> Mean Absolute Error (MAE): {mae:.2f}")
    print(f"-> R² Score / Akurasi Baru   : {akurasi_persen:.2f}% (Sebelumnya: 31.25%)")
    
    print("\n=== 5. Menyimpan Model Hasil Tuning ===")
    # Membuat folder khusus untuk menyimpan model jika belum ada
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/best_random_forest.pkl"
    joblib.dump(best_model, model_save_path)
    print(f"Model cerdas berhasil disimpan di: {model_save_path}")

if __name__ == "__main__":
    train_and_tune_model()