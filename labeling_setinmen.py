import asyncio
from pathlib import Path

import pandas as pd
from googletrans import Translator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# ============================================================
# KONFIGURASI FILE
# ============================================================

INPUT_FILE = Path("hasil_preprocessing.csv")
OUTPUT_FILE = Path("dataset_cleaned.csv")
CHECKPOINT_FILE = Path("checkpoint_labeling.csv")

# Jumlah percobaan ulang jika penerjemahan gagal
MAX_RETRY = 3

# Menyimpan sementara setiap sejumlah data
SAVE_INTERVAL = 20


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(file_path):
    """
    Membaca hasil preprocessing dari file CSV.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {file_path} tidak ditemukan. "
            "Jalankan program preprocessing terlebih dahulu."
        )

    data = pd.read_csv(file_path)

    if "komentar" not in data.columns:
        raise ValueError(
            "Kolom 'komentar' tidak ditemukan dalam dataset."
        )

    if "cleaned_komentar" not in data.columns:
        raise ValueError(
            "Kolom 'cleaned_komentar' tidak ditemukan. "
            "Pastikan Soal 6 sudah dijalankan."
        )

    return data


# ============================================================
# MEMILIH KOLOM UNTUK LABELING
# ============================================================

def choose_text_column(data):
    """
    Memilih kolom teks yang digunakan untuk penerjemahan.

    Kolom normalisasi diprioritaskan karena susunan kalimatnya
    masih lebih lengkap dibandingkan hasil stemming.
    """

    if "normalisasi" in data.columns:
        return "normalisasi"

    return "cleaned_komentar"


# ============================================================
# MENYIAPKAN KOLOM HASIL TERJEMAHAN
# ============================================================

def prepare_translation_column(data):
    """
    Membuat atau mengambil kembali hasil penerjemahan sementara.
    """

    # Menggunakan checkpoint jika proses sebelumnya terhenti
    if CHECKPOINT_FILE.exists():
        checkpoint = pd.read_csv(CHECKPOINT_FILE)

        if (
            len(checkpoint) == len(data)
            and "hasil_translate" in checkpoint.columns
        ):
            print(
                "Checkpoint ditemukan. "
                "Penerjemahan dilanjutkan dari proses sebelumnya."
            )

            return checkpoint

    # Membuat kolom kosong jika belum pernah diterjemahkan
    data["hasil_translate"] = ""

    return data


# ============================================================
# MENERJEMAHKAN SATU TEKS
# ============================================================

async def translate_text(translator, text):
    """
    Menerjemahkan teks Bahasa Indonesia ke Bahasa Inggris.
    """

    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text).strip()

    for attempt in range(1, MAX_RETRY + 1):
        try:
            result = await translator.translate(
                text,
                src="id",
                dest="en"
            )

            return result.text

        except Exception as error:
            print(
                f"Percobaan terjemahan ke-{attempt} gagal:",
                error
            )

            # Memberi jeda sebelum mencoba kembali
            await asyncio.sleep(attempt * 2)

    return ""


# ============================================================
# MENERJEMAHKAN SELURUH DATASET
# ============================================================

async def translate_dataset(data, text_column):
    """
    Menerjemahkan seluruh data secara bertahap.
    """

    jumlah_data = len(data)

    async with Translator() as translator:
        for index in data.index:
            hasil_lama = data.at[index, "hasil_translate"]

            # Melewati data yang sudah diterjemahkan
            if (
                pd.notna(hasil_lama)
                and str(hasil_lama).strip() != ""
            ):
                continue

            text = data.at[index, text_column]

            hasil_translate = await translate_text(
                translator,
                text
            )

            data.at[index, "hasil_translate"] = hasil_translate

            print(
                f"Terjemahan {index + 1}/{jumlah_data}: "
                f"{hasil_translate}"
            )

            # Jeda kecil agar permintaan tidak terlalu cepat
            await asyncio.sleep(0.2)

            # Menyimpan proses sementara
            if (index + 1) % SAVE_INTERVAL == 0:
                data.to_csv(
                    CHECKPOINT_FILE,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    f"Checkpoint disimpan pada data ke-{index + 1}."
                )

    # Menyimpan checkpoint setelah seluruh proses
    data.to_csv(
        CHECKPOINT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    return data


# ============================================================
# MEMBERIKAN LABEL SENTIMEN
# ============================================================

sentiment_analyzer = SentimentIntensityAnalyzer()


def labeling_sentiment(text):
    """
    Menghitung skor VADER dan menentukan label sentimen.
    """

    if pd.isna(text) or str(text).strip() == "":
        return pd.Series({
            "score_negative": 0.0,
            "score_neutral": 1.0,
            "score_positive": 0.0,
            "compound": 0.0,
            "label": "Netral"
        })

    scores = sentiment_analyzer.polarity_scores(
        str(text)
    )

    compound = scores["compound"]

    if compound >= 0.05:
        label = "Positif"

    elif compound <= -0.05:
        label = "Negatif"

    else:
        label = "Netral"

    return pd.Series({
        "score_negative": scores["neg"],
        "score_neutral": scores["neu"],
        "score_positive": scores["pos"],
        "compound": compound,
        "label": label
    })


# ============================================================
# MENAMPILKAN HASIL LABELING
# ============================================================

def display_results(data):
    """
    Menampilkan 10 hasil terjemahan dan labeling pertama.
    """

    print("\n" + "=" * 80)
    print("HASIL LABELING SENTIMEN 10 DATA PERTAMA")
    print("=" * 80)

    for nomor, row in data.head(10).iterrows():
        print(f"\nData ke-{nomor + 1}")

        print(
            "Komentar asli      :",
            row["komentar"]
        )

        print(
            "Cleaned komentar   :",
            row["cleaned_komentar"]
        )

        print(
            "Hasil terjemahan   :",
            row["hasil_translate"]
        )

        print(
            "Score negative     :",
            row["score_negative"]
        )

        print(
            "Score neutral      :",
            row["score_neutral"]
        )

        print(
            "Score positive     :",
            row["score_positive"]
        )

        print(
            "Compound           :",
            row["compound"]
        )

        print(
            "Label              :",
            row["label"]
        )

        print("-" * 80)


# ============================================================
# MENAMPILKAN DISTRIBUSI LABEL
# ============================================================

def display_label_distribution(data):
    """
    Menampilkan jumlah masing-masing kelas sentimen.
    """

    distribusi = data["label"].value_counts()

    print("\n" + "=" * 50)
    print("DISTRIBUSI LABEL SENTIMEN")
    print("=" * 50)

    print(distribusi)

    print("\nPersentase label:")

    persentase = (
        data["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(persentase)


# ============================================================
# PROGRAM UTAMA
# ============================================================

async def main():
    print("Membaca hasil preprocessing...")

    data = load_dataset(INPUT_FILE)

    text_column = choose_text_column(data)

    print("Jumlah data:", len(data))
    print("Kolom untuk labeling:", text_column)

    data = prepare_translation_column(data)

    print("\nMemulai proses Google Translate...")

    data = await translate_dataset(
        data,
        text_column
    )

    # Memeriksa hasil terjemahan yang kosong
    jumlah_gagal = (
        data["hasil_translate"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print(
        "\nJumlah data yang gagal diterjemahkan:",
        jumlah_gagal
    )

    if jumlah_gagal > 0:
        print(
            "Beberapa data belum berhasil diterjemahkan."
        )

        print(
            "Jalankan kembali program agar proses dilanjutkan "
            "dari checkpoint."
        )

        return

    print("\nMelakukan labeling menggunakan VADER...")

    hasil_sentimen = data[
        "hasil_translate"
    ].apply(labeling_sentiment)

    data = pd.concat(
        [
            data,
            hasil_sentimen
        ],
        axis=1
    )

    # Menampilkan 10 hasil pertama
    display_results(data)

    # Menampilkan distribusi label
    display_label_distribution(data)

    # Menentukan kolom yang disimpan
    kolom_output = [
        "id",
        "komentar",
        "cleaned_komentar",
        "hasil_translate",
        "score_negative",
        "score_neutral",
        "score_positive",
        "compound",
        "label"
    ]

    # Menghapus kolom ID dari daftar jika tidak tersedia
    kolom_output = [
        kolom
        for kolom in kolom_output
        if kolom in data.columns
    ]

    # Menyimpan hasil akhir
    data[kolom_output].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nLabeling sentimen selesai.")
    print("File hasil:", OUTPUT_FILE)
    print("Jumlah data:", len(data))

    # Menghapus checkpoint setelah proses berhasil
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

        print(
            "File checkpoint telah dihapus "
            "karena proses selesai."
        )


if __name__ == "__main__":
    asyncio.run(main())