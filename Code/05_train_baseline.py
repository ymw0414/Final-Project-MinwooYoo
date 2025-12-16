"""
05_train_baseline.py

Train a TF-IDF + Logistic Regression baseline model
using the cleaned 1980s paragraph-filtered dataset.

Outputs:
    evaluation/baseline_metrics.txt
    evaluation/confusion_matrix.png
    models/tfidf_vectorizer.pkl
    models/logreg_model.pkl
"""

import pandas as pd
import pickle
from pathlib import Path
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt


def train_baseline(df: pd.DataFrame):
    X = df["speech"].tolist()
    y = df["party"].tolist()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        lowercase=True,
        min_df=5,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        solver="saga",
        n_jobs=-1,
    )

    clf.fit(X_train_vec, y_train)

    return (
        clf,
        vectorizer,
        (X_val_vec, y_val),
        (X_test_vec, y_test),
    )


def evaluate(model, X, y):
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, pos_label="R")
    report = classification_report(y, pred)
    cm = confusion_matrix(y, pred)
    return acc, f1, report, cm


def main(args):
    data_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    eval_dir = Path(args.eval_dir)

    model_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(data_path)
    print("Loaded:", df.shape)

    clf, vectorizer, val_set, test_set = train_baseline(df)

    val_acc, val_f1, val_report, _ = evaluate(clf, *val_set)
    test_acc, test_f1, test_report, cm = evaluate(clf, *test_set)

    metrics_path = eval_dir / "baseline_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Validation Metrics\n")
        f.write(f"Accuracy: {val_acc}\n")
        f.write(f"F1: {val_f1}\n")
        f.write(val_report)
        f.write("\n\nTest Metrics\n")
        f.write(f"Accuracy: {test_acc}\n")
        f.write(f"F1: {test_f1}\n")
        f.write(test_report)

    print("Saved metrics:", metrics_path)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig(eval_dir / "confusion_matrix.png")
    plt.close()

    with open(model_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(model_dir / "logreg_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Baseline training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/speeches_clean_1980s_paragraph.parquet",
        help="Input dataset",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="evaluation",
        help="Directory to save evaluation outputs",
    )
    args = parser.parse_args()
    main(args)
