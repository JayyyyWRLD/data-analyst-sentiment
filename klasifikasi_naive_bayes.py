from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================
# KONFIGURASI FILE
# ============================================================

INPUT_FILE = Path("dataset_cleaned.csv")
OUTPUT_PREDICTION = Path("hasil_prediksi_naive_bayes.csv")
OUTPUT_CONFUSION_MATRIX = Path("confusion_matrix.png")


# ============================================================
# MEMBACA DATASET
# ============================================================

def load_dataset(file_path):
    """
    Membaca dataset hasil labeling sentimen.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"File {file_path} tidak ditemukan."
        )

    data = pd.read_csv(file_path)

    kolom_wajib = {
        "cleaned_komentar",
        "label"
    }

    if not kolom_wajib.issubset(data.columns):
        raise ValueError(
            "Dataset harus memiliki kolom "
            "'cleaned_komentar' dan 'label'."
        )

    # Menghapus data kosong
    data = data.dropna(
        subset=[
            "cleaned_komentar",
            "label"
        ]
    ).copy()

    # Memastikan teks bertipe string
    data["cleaned_komentar"] = (
        data["cleaned_komentar"]
        .astype(str)
        .str.strip()
    )

    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
    )

    # Menghapus teks kosong
    data = data[
        data["cleaned_komentar"] != ""
    ]

    # Mengatur ulang index
    data = data.reset_index(drop=True)

    return data


# ============================================================
# MENAMPILKAN DISTRIBUSI DATA
# ============================================================

def display_distribution(data):
    """
    Menampilkan jumlah dan persentase setiap kelas.
    """

    print("\n" + "=" * 55)
    print("DISTRIBUSI DATASET")
    print("=" * 55)

    print(data["label"].value_counts())

    print("\nPersentase:")

    persentase = (
        data["label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print(persentase)


# ============================================================
# MEMBUAT CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(
    y_test,
    y_pred,
    class_names
):
    """
    Membuat dan menyimpan grafik confusion matrix.
    """

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=class_names
    )

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title(
        "Confusion Matrix Multinomial Naive Bayes"
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_CONFUSION_MATRIX,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    return matrix


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main():
    print("Membaca dataset_cleaned.csv...")

    data = load_dataset(INPUT_FILE)

    print("Jumlah data:", len(data))

    display_distribution(data)

    # --------------------------------------------------------
    # MEMISAHKAN FITUR DAN TARGET
    # --------------------------------------------------------

    X = data["cleaned_komentar"]
    y = data["label"]

    # --------------------------------------------------------
    # SPLIT DATA 80:20
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n" + "=" * 55)
    print("PEMBAGIAN DATA")
    print("=" * 55)

    print("Jumlah data training:", len(X_train))
    print("Jumlah data testing :", len(X_test))

    print("\nDistribusi data training:")
    print(y_train.value_counts())

    print("\nDistribusi data testing:")
    print(y_test.value_counts())

    # --------------------------------------------------------
    # COUNT VECTORIZATION
    # --------------------------------------------------------

    vectorizer = CountVectorizer()

    X_train_vectorized = vectorizer.fit_transform(
        X_train
    )

    X_test_vectorized = vectorizer.transform(
        X_test
    )

    print("\n" + "=" * 55)
    print("HASIL COUNT VECTORIZER")
    print("=" * 55)

    print(
        "Jumlah fitur atau kosakata:",
        len(vectorizer.get_feature_names_out())
    )

    print(
        "Ukuran data training:",
        X_train_vectorized.shape
    )

    print(
        "Ukuran data testing:",
        X_test_vectorized.shape
    )

    # --------------------------------------------------------
    # MEMBANGUN MODEL MULTINOMIAL NAIVE BAYES
    # --------------------------------------------------------

    model = MultinomialNB()

    model.fit(
        X_train_vectorized,
        y_train
    )

    # --------------------------------------------------------
    # MELAKUKAN PREDIKSI
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test_vectorized
    )

    # --------------------------------------------------------
    # MENGHITUNG METRIK EVALUASI
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 55)
    print("HASIL EVALUASI MODEL")
    print("=" * 55)

    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Accuracy Persen   : {accuracy * 100:.2f}%")
    print(f"Precision Weighted: {precision:.4f}")
    print(f"Recall Weighted   : {recall:.4f}")
    print(f"F1-Score Weighted : {f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    class_names = [
        "Negatif",
        "Netral",
        "Positif"
    ]

    matrix = create_confusion_matrix(
        y_test,
        y_pred,
        class_names
    )

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            matrix,
            index=[
                f"Actual {label}"
                for label in class_names
            ],
            columns=[
                f"Predicted {label}"
                for label in class_names
            ]
        )
    )

    # --------------------------------------------------------
    # MENYIMPAN HASIL PREDIKSI
    # --------------------------------------------------------

    hasil_prediksi = pd.DataFrame({
        "cleaned_komentar": X_test.values,
        "label_aktual": y_test.values,
        "label_prediksi": y_pred
    })

    hasil_prediksi["status_prediksi"] = (
        hasil_prediksi["label_aktual"]
        == hasil_prediksi["label_prediksi"]
    ).map({
        True: "Benar",
        False: "Salah"
    })

    hasil_prediksi.to_csv(
        OUTPUT_PREDICTION,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nProses klasifikasi selesai.")

    print(
        "File hasil prediksi:",
        OUTPUT_PREDICTION
    )

    print(
        "File confusion matrix:",
        OUTPUT_CONFUSION_MATRIX
    )


if __name__ == "__main__":
    main()