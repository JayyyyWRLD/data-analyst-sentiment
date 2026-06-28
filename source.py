# Muhammad Faiz
# 24101152630023
# IF-5
# Pemrograman Data Analyst

import asyncio
import inspect
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from googletrans import Translator
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# KONFIGURASI FILE
# ============================================================

SIMPLE_JSON_FILE = Path("simple.json")
DATASET_MENTAH_FILE = Path("dataset_mentah.csv")
KAMUS_FILE = Path("kamus.csv")
HASIL_PREPROCESSING_FILE = Path("hasil_preprocessing.csv")
DATASET_CLEANED_FILE = Path("dataset_cleaned.csv")
CHECKPOINT_LABELING_FILE = Path("checkpoint_labeling.csv")
HASIL_PREDIKSI_FILE = Path("hasil_prediksi_naive_bayes.csv")
CONFUSION_MATRIX_FILE = Path("confusion_matrix.png")

JUMLAH_DATA = 300
MAX_RETRY_TRANSLATE = 3
SAVE_INTERVAL = 20


# ============================================================
# KAMUS NORMALISASI DEFAULT
# ============================================================

DEFAULT_KAMUS = [
    ("ga", "tidak"),
    ("gini", "begini"),
    ("gimana", "bagaimana"),
    ("banget", "sangat"),
    ("recommended", "direkomendasikan"),
    ("seller", "penjual"),
    ("gw", "saya"),
    ("emang", "memang"),
    ("aja", "saja"),
    ("gitu", "begitu"),
    ("nyesel", "menyesal"),
    ("zonk", "mengecewakan"),
    ("best", "terbaik"),
    ("top", "terbaik"),
    ("worth", "sepadan"),
    ("oke", "baik"),
    ("barangny", "barang"),
    ("produknya", "produk"),
    ("tokonya", "toko"),
    ("pelayanannya", "pelayanan"),
    ("pembeliannya", "pembelian"),
    ("kualitasnya", "kualitas"),
    ("aslinya", "asli"),
    ("barangnya", "barang"),
    ("pesanannya", "pesanan"),
]


# ============================================================
# BAGIAN 1: KONVERSI DATASET JSON MENJADI CSV
# ============================================================

def convert_json_to_csv():
    """Mengubah simple.json menjadi dataset_mentah.csv."""

    if DATASET_MENTAH_FILE.exists():
        print("dataset_mentah.csv sudah tersedia. Tahap konversi dilewati.")
        data = pd.read_csv(DATASET_MENTAH_FILE)
        return data

    if not SIMPLE_JSON_FILE.exists():
        raise FileNotFoundError(
            "File simple.json tidak ditemukan. Letakkan simple.json pada folder yang sama "
            "dengan source.py atau sediakan dataset_mentah.csv."
        )

    try:
        with open(SIMPLE_JSON_FILE, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        if isinstance(json_data, list):
            data = pd.DataFrame(json_data)
        elif isinstance(json_data, dict):
            if "data" in json_data and isinstance(json_data["data"], list):
                data = pd.DataFrame(json_data["data"])
            else:
                data = pd.DataFrame(json_data)
        else:
            raise ValueError("Struktur JSON tidak dikenali.")

    except json.JSONDecodeError:
        data = pd.read_json(SIMPLE_JSON_FILE, lines=True)

    print("\nKolom yang tersedia pada dataset:")
    print(data.columns.tolist())
    print("Jumlah data awal:", len(data))

    calon_kolom_ulasan = ["comment", "komentar", "review", "reviews", "text", "ulasan"]
    kolom_ulasan = None

    for nama_kolom in calon_kolom_ulasan:
        if nama_kolom in data.columns:
            kolom_ulasan = nama_kolom
            break

    if kolom_ulasan is None:
        raise ValueError(
            "Kolom ulasan tidak ditemukan. Kolom yang tersedia: "
            f"{data.columns.tolist()}"
        )

    dataset_mentah = data[[kolom_ulasan]].copy()
    dataset_mentah = dataset_mentah.rename(columns={kolom_ulasan: "komentar"})
    dataset_mentah = dataset_mentah.dropna(subset=["komentar"])
    dataset_mentah["komentar"] = dataset_mentah["komentar"].astype(str).str.strip()
    dataset_mentah = dataset_mentah[dataset_mentah["komentar"] != ""]
    dataset_mentah = dataset_mentah.drop_duplicates(subset=["komentar"])

    if JUMLAH_DATA is not None:
        if len(dataset_mentah) >= JUMLAH_DATA:
            dataset_mentah = dataset_mentah.sample(n=JUMLAH_DATA, random_state=42)
        else:
            print(
                f"Peringatan: jumlah data bersih hanya {len(dataset_mentah)}, "
                f"kurang dari target {JUMLAH_DATA}."
            )

    dataset_mentah = dataset_mentah.reset_index(drop=True)
    dataset_mentah.insert(0, "id", range(1, len(dataset_mentah) + 1))
    dataset_mentah.to_csv(DATASET_MENTAH_FILE, index=False, encoding="utf-8-sig")

    print("\nKonversi dataset berhasil.")
    print("File output:", DATASET_MENTAH_FILE)
    print("Jumlah data akhir:", len(dataset_mentah))
    print("\nSepuluh data pertama:")
    print(dataset_mentah.head(10))

    return dataset_mentah


# ============================================================
# BAGIAN 2: KAMUS NORMALISASI
# ============================================================

def load_or_create_dictionary():
    """Membaca kamus.csv atau membuat kamus default jika file belum tersedia."""

    if not KAMUS_FILE.exists():
        kamus_default = pd.DataFrame(
            DEFAULT_KAMUS,
            columns=["kata_tidak_baku", "kata_baku"],
        )
        kamus_default.to_csv(KAMUS_FILE, index=False, encoding="utf-8-sig")
        print("\nkamus.csv belum tersedia. Program membuat kamus.csv default.")

    kamus_data = pd.read_csv(KAMUS_FILE)

    kolom_wajib = {"kata_tidak_baku", "kata_baku"}
    if not kolom_wajib.issubset(kamus_data.columns):
        raise ValueError(
            "File kamus.csv harus memiliki kolom 'kata_tidak_baku' dan 'kata_baku'."
        )

    kamus_data["kata_tidak_baku"] = (
        kamus_data["kata_tidak_baku"].astype(str).str.lower().str.strip()
    )
    kamus_data["kata_baku"] = (
        kamus_data["kata_baku"].astype(str).str.lower().str.strip()
    )

    kamus = dict(zip(kamus_data["kata_tidak_baku"], kamus_data["kata_baku"]))

    print("Jumlah kata dalam kamus:", len(kamus))
    return kamus


# ============================================================
# BAGIAN 3: PREPROCESSING TEKS
# ============================================================

def prepare_stopwords():
    """Menyiapkan daftar stopword Bahasa Indonesia."""

    try:
        stopwords.words("indonesian")
    except LookupError:
        print("Mengunduh stopwords Bahasa Indonesia dari NLTK...")
        nltk.download("stopwords")

    stop_words = set(stopwords.words("indonesian"))

    # Kata negasi dipertahankan karena penting untuk sentimen.
    kata_negasi = {"tidak", "bukan", "jangan", "belum", "kurang", "tak"}
    stop_words.difference_update(kata_negasi)

    # Stopword tambahan berupa kata percakapan.
    stopword_tambahan = {"nih", "sih", "dong", "lah", "deh", "ya", "kok"}
    stop_words.update(stopword_tambahan)

    return stop_words


def clean_whitespace(text):
    """Menghapus spasi berlebihan."""
    return re.sub(r"\s+", " ", str(text)).strip()


def case_folding(text):
    """Tahap 1: mengubah seluruh huruf menjadi huruf kecil."""
    return clean_whitespace(str(text).lower())


def remove_url(text):
    """Tahap 2: menghapus URL."""
    text = re.sub(r"https?://\S+|www\.\S+", " ", str(text))
    return clean_whitespace(text)


def remove_number(text):
    """Tahap 3: menghapus angka."""
    text = re.sub(r"\d+", " ", str(text))
    return clean_whitespace(text)


def remove_punctuation(text):
    """Tahap 4: menghapus tanda baca, simbol, dan emoji."""
    text = re.sub(r"[^\w\s]", " ", str(text))
    text = text.replace("_", " ")
    return clean_whitespace(text)


def normalize_words(text, kamus):
    """Tahap 5: mengganti kata tidak baku menjadi kata baku."""
    daftar_kata = str(text).split()
    hasil_normalisasi = [kamus.get(kata, kata) for kata in daftar_kata]
    return clean_whitespace(" ".join(hasil_normalisasi))


def remove_stopwords(text, stop_words):
    """Tahap 6: menghapus stopword Bahasa Indonesia."""
    daftar_kata = str(text).split()
    hasil = [kata for kata in daftar_kata if kata not in stop_words]
    return clean_whitespace(" ".join(hasil))


def stemming_text(text, stemmer):
    """Tahap 7: mengubah kata berimbuhan menjadi kata dasar."""
    return clean_whitespace(stemmer.stem(str(text)))


def run_preprocessing():
    """Menjalankan seluruh tahapan preprocessing teks."""

    if not DATASET_MENTAH_FILE.exists():
        raise FileNotFoundError("dataset_mentah.csv tidak ditemukan. Jalankan konversi dataset terlebih dahulu.")

    data = pd.read_csv(DATASET_MENTAH_FILE)
    if "komentar" not in data.columns:
        raise ValueError("Dataset harus memiliki kolom 'komentar'.")

    data = data.dropna(subset=["komentar"]).copy()
    data["komentar"] = data["komentar"].astype(str).str.strip()
    data = data[data["komentar"] != ""]
    data = data.drop_duplicates(subset=["komentar"]).reset_index(drop=True)

    if "id" not in data.columns:
        data.insert(0, "id", range(1, len(data) + 1))

    kamus = load_or_create_dictionary()
    stop_words = prepare_stopwords()
    stemmer = StemmerFactory().create_stemmer()

    print("\nMemulai preprocessing teks...")
    print("Jumlah data awal:", len(data))

    data["case_folding"] = data["komentar"].apply(case_folding)
    data["remove_url"] = data["case_folding"].apply(remove_url)
    data["remove_number"] = data["remove_url"].apply(remove_number)
    data["remove_punctuation"] = data["remove_number"].apply(remove_punctuation)
    data["normalisasi"] = data["remove_punctuation"].apply(lambda text: normalize_words(text, kamus))
    data["stopword_removal"] = data["normalisasi"].apply(lambda text: remove_stopwords(text, stop_words))
    data["stemming"] = data["stopword_removal"].apply(lambda text: stemming_text(text, stemmer))
    data["cleaned_komentar"] = data["stemming"]

    jumlah_kosong = data["cleaned_komentar"].fillna("").str.strip().eq("").sum()
    print("Jumlah hasil preprocessing kosong:", jumlah_kosong)

    print("\nHasil preprocessing 10 data pertama:")
    kolom_tampil = ["id", "komentar", "case_folding", "normalisasi", "stopword_removal", "stemming", "cleaned_komentar"]
    print(data[kolom_tampil].head(10))

    data.to_csv(HASIL_PREPROCESSING_FILE, index=False, encoding="utf-8-sig")

    print("\nPreprocessing selesai.")
    print("File output:", HASIL_PREPROCESSING_FILE)
    print("Jumlah data akhir:", len(data))

    return data


# ============================================================
# BAGIAN 4: LABELING SENTIMEN GOOGLE TRANSLATE + VADER
# ============================================================

def choose_text_column_for_labeling(data):
    """Memilih kolom teks untuk diterjemahkan."""
    if "normalisasi" in data.columns:
        return "normalisasi"
    return "cleaned_komentar"


def prepare_translation_column(data):
    """Membuat atau melanjutkan checkpoint penerjemahan."""

    if CHECKPOINT_LABELING_FILE.exists():
        checkpoint = pd.read_csv(CHECKPOINT_LABELING_FILE)
        if len(checkpoint) == len(data) and "hasil_translate" in checkpoint.columns:
            print("Checkpoint labeling ditemukan. Proses terjemahan dilanjutkan.")
            return checkpoint

    data = data.copy()
    data["hasil_translate"] = ""
    return data


async def translate_text(translator, text):
    """Menerjemahkan satu teks dari Bahasa Indonesia ke Bahasa Inggris."""

    if pd.isna(text) or str(text).strip() == "":
        return ""

    text = str(text).strip()

    for attempt in range(1, MAX_RETRY_TRANSLATE + 1):
        try:
            result = translator.translate(text, src="id", dest="en")

            # Mendukung versi googletrans yang synchronous maupun asynchronous.
            if inspect.isawaitable(result):
                result = await result

            return getattr(result, "text", str(result))

        except Exception as error:
            print(f"Percobaan terjemahan ke-{attempt} gagal: {error}")
            await asyncio.sleep(attempt * 2)

    return ""


async def translate_loop(translator, data, text_column):
    """Melakukan penerjemahan untuk seluruh baris dataset."""

    jumlah_data = len(data)

    for index in data.index:
        hasil_lama = data.at[index, "hasil_translate"]
        if pd.notna(hasil_lama) and str(hasil_lama).strip() != "":
            continue

        text = data.at[index, text_column]
        hasil_translate = await translate_text(translator, text)
        data.at[index, "hasil_translate"] = hasil_translate

        print(f"Terjemahan {index + 1}/{jumlah_data}: {hasil_translate}")
        await asyncio.sleep(0.2)

        if (index + 1) % SAVE_INTERVAL == 0:
            data.to_csv(CHECKPOINT_LABELING_FILE, index=False, encoding="utf-8-sig")
            print(f"Checkpoint disimpan pada data ke-{index + 1}.")

    data.to_csv(CHECKPOINT_LABELING_FILE, index=False, encoding="utf-8-sig")
    return data


async def translate_dataset(data, text_column):
    """Menerjemahkan dataset dengan Google Translate."""

    translator_object = Translator()

    if hasattr(translator_object, "__aenter__"):
        async with translator_object as translator:
            data = await translate_loop(translator, data, text_column)
    else:
        data = await translate_loop(translator_object, data, text_column)

    return data


sentiment_analyzer = SentimentIntensityAnalyzer()


def labeling_sentiment(text):
    """Memberi label Positif, Negatif, atau Netral menggunakan skor VADER."""

    if pd.isna(text) or str(text).strip() == "":
        return pd.Series({
            "score_negative": 0.0,
            "score_neutral": 1.0,
            "score_positive": 0.0,
            "compound": 0.0,
            "label": "Netral",
        })

    scores = sentiment_analyzer.polarity_scores(str(text))
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
        "label": label,
    })


async def run_labeling():
    """Menjalankan Google Translate dan VADER Sentiment."""

    if not HASIL_PREPROCESSING_FILE.exists():
        raise FileNotFoundError("hasil_preprocessing.csv tidak ditemukan. Jalankan preprocessing terlebih dahulu.")

    data = pd.read_csv(HASIL_PREPROCESSING_FILE)
    if "komentar" not in data.columns or "cleaned_komentar" not in data.columns:
        raise ValueError("Data preprocessing harus memiliki kolom 'komentar' dan 'cleaned_komentar'.")

    text_column = choose_text_column_for_labeling(data)
    print("\nMemulai labeling sentimen...")
    print("Jumlah data:", len(data))
    print("Kolom untuk Google Translate:", text_column)

    data = prepare_translation_column(data)
    data = await translate_dataset(data, text_column)

    jumlah_gagal = data["hasil_translate"].fillna("").str.strip().eq("").sum()
    print("\nJumlah data yang gagal diterjemahkan:", jumlah_gagal)

    if jumlah_gagal > 0:
        print("Beberapa data belum berhasil diterjemahkan.")
        print("Jalankan kembali source.py agar proses dilanjutkan dari checkpoint.")
        return None

    hasil_sentimen = data["hasil_translate"].apply(labeling_sentiment)
    data = pd.concat([data, hasil_sentimen], axis=1)

    print("\nHasil labeling sentimen 10 data pertama:")
    kolom_tampil = [
        "id", "komentar", "cleaned_komentar", "hasil_translate",
        "compound", "label"
    ]
    kolom_tampil = [kolom for kolom in kolom_tampil if kolom in data.columns]
    print(data[kolom_tampil].head(10))

    print("\nDistribusi label sentimen:")
    print(data["label"].value_counts())
    print("\nPersentase label sentimen:")
    print(data["label"].value_counts(normalize=True).mul(100).round(2))

    kolom_output = [
        "id", "komentar", "cleaned_komentar", "hasil_translate",
        "score_negative", "score_neutral", "score_positive", "compound", "label"
    ]
    kolom_output = [kolom for kolom in kolom_output if kolom in data.columns]

    data[kolom_output].to_csv(DATASET_CLEANED_FILE, index=False, encoding="utf-8-sig")

    print("\nLabeling sentimen selesai.")
    print("File output:", DATASET_CLEANED_FILE)
    print("Jumlah data:", len(data))

    if CHECKPOINT_LABELING_FILE.exists():
        CHECKPOINT_LABELING_FILE.unlink()
        print("File checkpoint_labeling.csv dihapus karena proses selesai.")

    return data


# ============================================================
# BAGIAN 5: KLASIFIKASI MULTINOMIAL NAIVE BAYES
# ============================================================

def run_classification():
    """Membangun model klasifikasi sentimen dengan Multinomial Naive Bayes."""

    if not DATASET_CLEANED_FILE.exists():
        raise FileNotFoundError("dataset_cleaned.csv tidak ditemukan. Jalankan labeling terlebih dahulu.")

    data = pd.read_csv(DATASET_CLEANED_FILE)
    kolom_wajib = {"cleaned_komentar", "label"}
    if not kolom_wajib.issubset(data.columns):
        raise ValueError("dataset_cleaned.csv harus memiliki kolom 'cleaned_komentar' dan 'label'.")

    data = data.dropna(subset=["cleaned_komentar", "label"]).copy()
    data["cleaned_komentar"] = data["cleaned_komentar"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["cleaned_komentar"] != ""].reset_index(drop=True)

    print("\nMemulai klasifikasi Multinomial Naive Bayes...")
    print("Jumlah data yang digunakan:", len(data))

    print("\nDistribusi dataset:")
    print(data["label"].value_counts())
    print("\nPersentase dataset:")
    print(data["label"].value_counts(normalize=True).mul(100).round(2))

    X = data["cleaned_komentar"]
    y = data["label"]

    stratify_option = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_option,
    )

    print("\nPembagian data 80:20")
    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing :", len(X_test))

    vectorizer = CountVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    print("\nHasil CountVectorizer")
    print("Jumlah fitur/kosakata:", len(vectorizer.get_feature_names_out()))
    print("Ukuran data training:", X_train_vectorized.shape)
    print("Ukuran data testing :", X_test_vectorized.shape)

    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\nEvaluasi Model")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    class_names = ["Negatif", "Netral", "Positif"]
    class_names = [label for label in class_names if label in sorted(data["label"].unique())]

    matrix = confusion_matrix(y_test, y_pred, labels=class_names)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual {label}" for label in class_names],
        columns=[f"Predicted {label}" for label in class_names],
    )

    print("\nConfusion Matrix:")
    print(matrix_df)

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(values_format="d")
    plt.title("Confusion Matrix Multinomial Naive Bayes")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    hasil_prediksi = pd.DataFrame({
        "cleaned_komentar": X_test.values,
        "label_aktual": y_test.values,
        "label_prediksi": y_pred,
    })
    hasil_prediksi["status_prediksi"] = (
        hasil_prediksi["label_aktual"] == hasil_prediksi["label_prediksi"]
    ).map({True: "Benar", False: "Salah"})

    hasil_prediksi.to_csv(HASIL_PREDIKSI_FILE, index=False, encoding="utf-8-sig")

    print("\nKlasifikasi selesai.")
    print("File hasil prediksi:", HASIL_PREDIKSI_FILE)
    print("File confusion matrix:", CONFUSION_MATRIX_FILE)


# ============================================================
# PROGRAM UTAMA
# ============================================================

async def main():
    print("=" * 70)
    print("PROGRAM SENTIMENT ANALYSIS")
    print("=" * 70)

    print("\n[1] Konversi dataset")
    convert_json_to_csv()

    print("\n[2] Preprocessing teks")
    run_preprocessing()

    print("\n[3] Labeling sentimen otomatis")
    hasil_labeling = await run_labeling()

    if hasil_labeling is None:
        print("\nProgram dihentikan sementara karena masih ada data yang gagal diterjemahkan.")
        print("Jalankan kembali source.py setelah koneksi internet stabil.")
        return

    print("\n[4] Klasifikasi sentimen")
    run_classification()

    print("\nSeluruh proses selesai.")


if __name__ == "__main__":
    asyncio.run(main())


# Muhammad Faiz
# 24101152630023
# IF-5
# Pemrograman Data Analyst