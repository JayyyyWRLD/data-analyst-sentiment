import pandas as pd
from collections import Counter


# Membaca dataset
data = pd.read_csv("dataset_mentah.csv")

# Menggabungkan semua komentar
semua_teks = " ".join(
    data["komentar"]
    .dropna()
    .astype(str)
    .str.lower()
)

# Memisahkan teks menjadi kata
daftar_kata = semua_teks.split()

# Menghitung frekuensi kata
frekuensi_kata = Counter(daftar_kata)

# Menampilkan 100 kata yang paling sering muncul
print("100 kata yang paling sering muncul:\n")

for kata, jumlah in frekuensi_kata.most_common(100):
    print(f"{kata:<20} {jumlah}")