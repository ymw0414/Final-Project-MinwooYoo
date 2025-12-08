"""
04_preprocess_text_1981plus_unfiltered.py

This script preprocesses the merged speech dataset and keeps all speeches from
1981 onward (file_id >= 97), WITHOUT any paragraph-level filtering.

The text is cleaned minimally:
    - Remove empty/missing speeches
    - Normalize unicode
    - Normalize whitespace
    - Keep only Democrat/Republican speeches

No minimum sentence count or length filtering is applied beyond a minimal threshold.

Output:
    data/processed/speeches_clean_1981plus_unfiltered.parquet
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path

# -------------------------------
# 1. Paths (LOCAL Windows version)
# -------------------------------
INPUT_PATH = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_clean.parquet"
)
OUTPUT_PATH = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_clean_1981plus_unfiltered.parquet"
)

# -------------------------------
# 2. Load full dataset
# -------------------------------
df = pd.read_parquet(INPUT_PATH)
print("Loaded:", df.shape)

# -------------------------------
# 3. Convert file_id to integer
# -------------------------------
df["file_id"] = df["file_id"].astype(int)
print("file_id dtype:", df["file_id"].dtype)

# -------------------------------
# 4. Keep only 1981 onward
#    (file_id >= 97)
# -------------------------------
df = df[df["file_id"] >= 97]
print("After filtering for file_id >= 97:", df.shape)

# -------------------------------
# 5. Keep only valid party labels
# -------------------------------
df = df[df["party"].isin(["D", "R"])]
print("After party filter:", df.shape)

# -------------------------------
# 6. Drop missing / empty text
# -------------------------------
df = df.dropna(subset=["speech"])
df = df[df["speech"].str.strip().str.len() > 0]
print("After removing empty speeches:", df.shape)

# -------------------------------
# 7. Remove extremely short lines
# -------------------------------
df = df[df["speech"].str.len() >= 30]
print("After removing short lines:", df.shape)

# -------------------------------
# 8. Unicode normalization
# -------------------------------
def fix_unicode(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKC", text)

df["speech"] = df["speech"].apply(fix_unicode)

# -------------------------------
# 9. Whitespace normalization
# -------------------------------
def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

df["speech"] = df["speech"].apply(clean_whitespace)

# -------------------------------
# 10. Save cleaned dataset
# -------------------------------
df.to_parquet(OUTPUT_PATH)
print("Saved cleaned dataset:", df.shape)
print("Saved to:", OUTPUT_PATH)

print(df.head())
