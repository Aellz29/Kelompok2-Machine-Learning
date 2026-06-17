import pandas as pd

df = pd.read_csv('data_sosmed_bersih.csv')

# Cek tipe data Timestamp
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

print("Cek tipe data:")
print(df.dtypes)

print("\nCek apakah ada nilai kosong lagi:")
print(df.isnull().sum().sum())  # Harus 0

print("\nCek apakah masih ada duplikat:")
print(df.duplicated().sum())     # Harus 0

print("Informasi DataFrame:")
print(df.info())