import re
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ============================================================
# KONFIGURASI NAMA FILE
# ============================================================

DATASET_FILE = Path("dataset_mentah.csv")
KAMUS_FILE = Path("kamus.csv")
OUTPUT_FILE = Path("hasil_preprocessing.csv")


# ============================================================
# PERSIAPAN NLTK DAN SASTRAWI
# ============================================================

# Memastikan stopword Bahasa Indonesia sudah tersedia
try:
    stopwords.words("indonesian")
except LookupError:
    print("Mengunduh stopwords Bahasa Indonesia...")
    nltk.download("stopwords")


# Membuat objek stemmer Sastrawi
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# Mengambil stopword Bahasa Indonesia dari NLTK
stop_words = set(stopwords.words("indonesian"))


# Kata negasi tidak boleh dihapus karena memengaruhi sentimen
kata_negasi = {
    "tidak",
    "bukan",
    "jangan",
    "belum",
    "kurang",
    "tak"
}

stop_words.difference_update(kata_negasi)


# Stopword tambahan berupa kata percakapan
stopword_tambahan = {
    "nih",
    "sih",
    "dong",
    "lah",
    "deh",
    "ya",
    "kok"
}

stop_words.update(stopword_tambahan)


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(file_path):
    """
    Membaca dataset mentah dari file CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {file_path} tidak ditemukan."
        )

    data = pd.read_csv(file_path)

    if "komentar" not in data.columns:
        raise ValueError(
            "Kolom 'komentar' tidak ditemukan dalam dataset."
        )

    # Menghapus komentar kosong
    data = data.dropna(subset=["komentar"]).copy()

    # Memastikan komentar bertipe string
    data["komentar"] = (
        data["komentar"]
        .astype(str)
        .str.strip()
    )

    # Menghapus komentar yang benar-benar kosong
    data = data[data["komentar"] != ""]

    # Menghapus komentar duplikat
    data = data.drop_duplicates(
        subset=["komentar"]
    )

    # Mengatur ulang nomor index
    data = data.reset_index(drop=True)

    return data


# ============================================================
# MEMBACA KAMUS NORMALISASI
# ============================================================

def load_dictionary(file_path):
    """
    Membaca pasangan kata tidak baku dan kata baku.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {file_path} tidak ditemukan."
        )

    kamus_data = pd.read_csv(file_path)

    kolom_wajib = {
        "kata_tidak_baku",
        "kata_baku"
    }

    if not kolom_wajib.issubset(kamus_data.columns):
        raise ValueError(
            "File kamus harus memiliki kolom "
            "'kata_tidak_baku' dan 'kata_baku'."
        )

    # Mengubah seluruh isi kamus menjadi huruf kecil
    kamus_data["kata_tidak_baku"] = (
        kamus_data["kata_tidak_baku"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    kamus_data["kata_baku"] = (
        kamus_data["kata_baku"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    kamus = dict(
        zip(
            kamus_data["kata_tidak_baku"],
            kamus_data["kata_baku"]
        )
    )

    return kamus


# ============================================================
# FUNGSI PEMBERSIH SPASI
# ============================================================

def clean_whitespace(text):
    """
    Menghapus spasi yang berlebihan.
    """

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# TAHAP 1: CASE FOLDING
# ============================================================

def case_folding(text):
    """
    Mengubah seluruh huruf menjadi huruf kecil.
    """

    text = str(text).lower()

    return clean_whitespace(text)


# ============================================================
# TAHAP 2: MENGHAPUS URL
# ============================================================

def remove_url(text):
    """
    Menghapus alamat website atau URL dari teks.
    """

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    return clean_whitespace(text)


# ============================================================
# TAHAP 3: MENGHAPUS ANGKA
# ============================================================

def remove_number(text):
    """
    Menghapus seluruh angka dari teks.
    """

    text = re.sub(
        r"\d+",
        " ",
        text
    )

    return clean_whitespace(text)


# ============================================================
# TAHAP 4: MENGHAPUS TANDA BACA
# ============================================================

def remove_punctuation(text):
    """
    Menghapus tanda baca, simbol, dan emoji dari teks.
    """

    # Menyisakan huruf, angka, underscore, dan spasi
    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    # Menghapus underscore
    text = text.replace("_", " ")

    return clean_whitespace(text)


# ============================================================
# TAHAP 5: NORMALISASI KATA
# ============================================================

def normalize_words(text, kamus):
    """
    Mengubah kata tidak baku menjadi kata baku
    menggunakan file kamus.csv.
    """

    daftar_kata = text.split()

    hasil_normalisasi = [
        kamus.get(kata, kata)
        for kata in daftar_kata
    ]

    text = " ".join(hasil_normalisasi)

    return clean_whitespace(text)


# ============================================================
# TAHAP 6: STOPWORD REMOVAL
# ============================================================

def remove_stopwords(text):
    """
    Menghapus kata yang dianggap tidak memberikan
    informasi penting bagi analisis sentimen.
    """

    daftar_kata = text.split()

    hasil_stopword = [
        kata
        for kata in daftar_kata
        if kata not in stop_words
    ]

    text = " ".join(hasil_stopword)

    return clean_whitespace(text)


# ============================================================
# TAHAP 7: STEMMING
# ============================================================

def stemming_text(text):
    """
    Mengubah kata berimbuhan menjadi kata dasar
    menggunakan library Sastrawi.
    """

    text = stemmer.stem(text)

    return clean_whitespace(text)


# ============================================================
# MENAMPILKAN 10 DATA PERTAMA
# ============================================================

def display_results(data):
    """
    Menampilkan hasil setiap tahap preprocessing
    untuk 10 data pertama.
    """

    data_tampil = data.head(10)

    print("\n" + "=" * 70)
    print("HASIL TEXT PREPROCESSING 10 DATA PERTAMA")
    print("=" * 70)

    for nomor, row in data_tampil.iterrows():
        print(f"\nData ke-{nomor + 1}")

        print(
            "Komentar asli       :",
            row["komentar"]
        )

        print(
            "Case folding        :",
            row["case_folding"]
        )

        print(
            "Hapus URL           :",
            row["remove_url"]
        )

        print(
            "Hapus angka         :",
            row["remove_number"]
        )

        print(
            "Hapus tanda baca    :",
            row["remove_punctuation"]
        )

        print(
            "Normalisasi         :",
            row["normalisasi"]
        )

        print(
            "Stopword removal    :",
            row["stopword_removal"]
        )

        print(
            "Stemming            :",
            row["stemming"]
        )

        print(
            "Cleaned komentar    :",
            row["cleaned_komentar"]
        )

        print("-" * 70)


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("Membaca dataset...")

    data = load_dataset(DATASET_FILE)

    print("Membaca kamus kata baku...")

    kamus = load_dictionary(KAMUS_FILE)

    print("Jumlah data awal:", len(data))
    print("Jumlah kata dalam kamus:", len(kamus))

    # Tahap 1: Case folding
    data["case_folding"] = (
        data["komentar"]
        .apply(case_folding)
    )

    # Tahap 2: Menghapus URL
    data["remove_url"] = (
        data["case_folding"]
        .apply(remove_url)
    )

    # Tahap 3: Menghapus angka
    data["remove_number"] = (
        data["remove_url"]
        .apply(remove_number)
    )

    # Tahap 4: Menghapus tanda baca
    data["remove_punctuation"] = (
        data["remove_number"]
        .apply(remove_punctuation)
    )

    # Tahap 5: Normalisasi kata
    data["normalisasi"] = (
        data["remove_punctuation"]
        .apply(
            lambda text: normalize_words(
                text,
                kamus
            )
        )
    )

    # Tahap 6: Stopword removal
    data["stopword_removal"] = (
        data["normalisasi"]
        .apply(remove_stopwords)
    )

    # Tahap 7: Stemming
    data["stemming"] = (
        data["stopword_removal"]
        .apply(stemming_text)
    )

    # Hasil akhir preprocessing
    data["cleaned_komentar"] = data["stemming"]

    # Menghapus hasil preprocessing yang kosong
    jumlah_kosong = (
        data["cleaned_komentar"] == ""
    ).sum()

    print(
        "Jumlah hasil preprocessing kosong:",
        jumlah_kosong
    )

    # Menampilkan 10 data pertama
    display_results(data)

    # Menyimpan seluruh tahapan preprocessing
    data.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nPreprocessing selesai.")
    print("File hasil:", OUTPUT_FILE)
    print("Jumlah data akhir:", len(data))


if __name__ == "__main__":
    main()