import json
from pathlib import Path

import pandas as pd


# Nama file input dan output
input_file = Path("simple.json")
output_file = Path("dataset_mentah.csv")

# Jumlah data yang ingin digunakan
# Ubah menjadi None jika ingin memakai seluruh data
jumlah_data = 300


# Memastikan file JSON tersedia
if not input_file.exists():
    raise FileNotFoundError(
        f"File {input_file} tidak ditemukan. "
        "Pastikan simple.json berada dalam folder yang sama dengan program."
    )


# Membaca file JSON
try:
    with open(input_file, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    # Jika struktur JSON berupa list
    if isinstance(json_data, list):
        data = pd.DataFrame(json_data)

    # Jika struktur JSON berupa dictionary
    elif isinstance(json_data, dict):
        # Beberapa dataset menyimpan data dalam key bernama data
        if "data" in json_data and isinstance(json_data["data"], list):
            data = pd.DataFrame(json_data["data"])
        else:
            data = pd.DataFrame(json_data)

    else:
        raise ValueError("Struktur JSON tidak dikenali.")

except json.JSONDecodeError:
    # Alternatif jika file menggunakan format JSON Lines
    data = pd.read_json(input_file, lines=True)


# Menampilkan informasi awal
print("Kolom yang tersedia:")
print(data.columns.tolist())

print("\nLima data pertama:")
print(data.head())

print("\nJumlah data awal:", len(data))


# Mencari kolom yang berisi teks ulasan
calon_kolom_ulasan = [
    "comment",
    "komentar",
    "review",
    "reviews",
    "text",
    "ulasan"
]

kolom_ulasan = None

for nama_kolom in calon_kolom_ulasan:
    if nama_kolom in data.columns:
        kolom_ulasan = nama_kolom
        break


if kolom_ulasan is None:
    raise ValueError(
        "Kolom ulasan tidak ditemukan. "
        f"Kolom yang tersedia: {data.columns.tolist()}"
    )


# Mengambil kolom ulasan dan mengubah namanya menjadi komentar
dataset_mentah = data[[kolom_ulasan]].copy()

dataset_mentah = dataset_mentah.rename(
    columns={kolom_ulasan: "komentar"}
)


# Menghapus data kosong
dataset_mentah = dataset_mentah.dropna(
    subset=["komentar"]
)

# Mengubah data menjadi string
dataset_mentah["komentar"] = (
    dataset_mentah["komentar"]
    .astype(str)
    .str.strip()
)

# Menghapus komentar kosong
dataset_mentah = dataset_mentah[
    dataset_mentah["komentar"] != ""
]

# Menghapus komentar duplikat
dataset_mentah = dataset_mentah.drop_duplicates(
    subset=["komentar"]
)


# Mengambil sampel data secara acak
if jumlah_data is not None:
    if len(dataset_mentah) >= jumlah_data:
        dataset_mentah = dataset_mentah.sample(
            n=jumlah_data,
            random_state=42
        )
    else:
        print(
            f"\nPeringatan: jumlah data bersih hanya "
            f"{len(dataset_mentah)}, kurang dari {jumlah_data}."
        )


# Mengatur ulang index
dataset_mentah = dataset_mentah.reset_index(drop=True)

# Menambahkan kolom ID
dataset_mentah.insert(
    0,
    "id",
    range(1, len(dataset_mentah) + 1)
)


# Menyimpan ke CSV
dataset_mentah.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)


# Menampilkan hasil
print("\nKonversi berhasil.")

print("Nama file output:", output_file)

print("Jumlah data akhir:", len(dataset_mentah))

print("\nSepuluh data pertama:")
print(dataset_mentah.head(10))