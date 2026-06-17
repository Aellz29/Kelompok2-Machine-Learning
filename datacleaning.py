import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load data
file_path = 'social_media_engagement_dataset.csv'
df = pd.read_csv(file_path)

# 2. Pembersihan
# Hapus duplikat
df = df.drop_duplicates()

# Konversi kolom tanggal
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# 3. Menampilkan hasil ke terminal
print("--- Info Dataset ---")
print(df.info())

print("\n--- Cek Missing Values ---")
print(df.isnull().sum())

print("\n--- 5 Data Pertama ---")
print(df.head())

# 4. Contoh Analisis Singkat
rata_rata_likes = df.groupby('Platform')['Likes'].mean()
print("\n--- Rata-rata Likes per Platform ---")
print(rata_rata_likes)

print(df[['Likes', 'Engagement_Rate']].corr())




# Contoh: Membuat grafik batang perbandingan Likes antar Platform
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='Platform', y='Likes')
plt.title('Rata-rata Likes per Platform')
plt.show()

df.to_csv('data_sosmed_bersih.csv', index=False)
print("Data bersih berhasil disimpan!")