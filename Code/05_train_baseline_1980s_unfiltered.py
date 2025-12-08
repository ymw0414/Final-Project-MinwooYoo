"""
05_train_baseline_1980s_unfiltered.py

This script trains a TF-IDF + Logistic Regression baseline model using all
speeches from the 1980s (file_id ∈ {97, 98, 99, 100}) WITHOUT any paragraph-
level filtering.

Dataset:
    - Includes all Democrat/Republican speeches from 97th–100th Congress
    - No sentence-count or length filtering applied
    - Text is cleaned but otherwise unfiltered

Pipeline:
    1. Load cleaned dataset (speeches_clean_1980s_unfiltered.parquet)
    2. Train/validation/test split
    3. TF-IDF vectorization (lowercase applied only at this step)
    4. Logistic Regression classifier (balanced classes)
    5. Evaluate on validation and test sets
    6. Save metrics, confusion matrix, and model artifacts

Outputs:
    evaluation/baseline_metrics_1980s_unfiltered.txt
    evaluation/confusion_matrix_1980s_unfiltered.png
    models/tfidf_vectorizer_1980s_unfiltered.pkl
    models/logreg_model_1980s_unfiltered.pkl
"""

import pandas as pd
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================
# Paths
# ======================================================================
DATA_PATH = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/speeches_clean_1980s_unfiltered.parquet")

MODEL_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/models")
EVAL_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/evaluation")

MODEL_DIR.mkdir(exist_ok=True)
EVAL_DIR.mkdir(exist_ok=True)

# ======================================================================
# 1. Load Data
# ======================================================================
df = pd.read_parquet(DATA_PATH)
print("Loaded:", df.shape)

X = df["speech"].tolist()
y = df["party"].tolist()

# ======================================================================
# 2. Train/Val/Test split
# ======================================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Train size:", len(X_train))
print("Val size:", len(X_val))
print("Test size:", len(X_test))

# ======================================================================
# 3. TF-IDF Vectorizer
# ======================================================================
vectorizer = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    lowercase=True,
    min_df=5
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# ======================================================================
# 4. Logistic Regression classifier
# ======================================================================
clf = LogisticRegression(
    max_iter=500,
    n_jobs=-1,
    class_weight="balanced",
    solver="saga"
)

clf.fit(X_train_vec, y_train)

# ======================================================================
# 5. Validation Evaluation
# ======================================================================
val_pred = clf.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_pred)
val_f1 = f1_score(y_val, val_pred, average="binary", pos_label="R")

print("\nValidation Accuracy:", val_acc)
print("Validation F1 (R):", val_f1)
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred))

# ======================================================================
# 6. Test Evaluation
# ======================================================================
test_pred = clf.predict(X_test_vec)

test_acc = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average="binary", pos_label="R")

print("\nTest Accuracy:", test_acc)
print("Test F1 (R):", test_f1)
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred))

# ======================================================================
# Save metrics
# ======================================================================
metrics_path = EVAL_DIR / "baseline_metrics_1980s_unfiltered.txt"
with open(metrics_path, "w") as f:
    f.write("Validation Metrics:\n")
    f.write(f"Accuracy: {val_acc}\n")
    f.write(f"F1 Score: {val_f1}\n")
    f.write(classification_report(y_val, val_pred))
    f.write("\nTest Metrics:\n")
    f.write(f"Accuracy: {test_acc}\n")
    f.write(f"F1 Score: {test_f1}\n")
    f.write(classification_report(y_test, test_pred))

print("Saved metrics to:", metrics_path)

# ======================================================================
# Save confusion matrix
# ======================================================================
cm = confusion_matrix(y_test, test_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["D", "R"],
    yticklabels=["D", "R"]
)
plt.title("Confusion Matrix (Test Set)")
plt.xlabel("Predicted")
plt.ylabel("True")

cm_path = EVAL_DIR / "confusion_matrix_1980s_unfiltered.png"
plt.savefig(cm_path)
plt.close()

print("Saved confusion matrix:", cm_path)

# ======================================================================
# Save model + vectorizer
# ======================================================================
with open(MODEL_DIR / "tfidf_vectorizer_1980s_unfiltered.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(MODEL_DIR / "logreg_model_1980s_unfiltered.pkl", "wb") as f:
    pickle.dump(clf, f)

print("Baseline training complete.")
