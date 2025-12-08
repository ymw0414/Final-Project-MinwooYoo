"""
04_preprocess_text_1980s_unfiltered.py

Preprocess speeches and keep ONLY 1980s decade (file_id 97–100),
WITHOUT paragraph-level filtering.

Output:
    data/processed/speeches_clean_1980s_unfiltered.parquet
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path

INPUT_PATH = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/speeches_with_party.parquet")
OUTPUT_PATH = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/speeches_clean_1980s_unfiltered.parquet")

df = pd.read_parquet(INPUT_PATH)
print("Loaded:", df.shape)

df["file_id"] = df["file_id"].astype(int)

df = df[df["file_id"].isin([97, 98, 99, 100])]
print("After filtering 1980s:", df.shape)

df = df[df["party"].isin(["D", "R"])]
df = df.dropna(subset=["speech"])
df = df[df["speech"].str.strip().str.len() > 0]

print("After removing empty speeches:", df.shape)

def fix_unicode(text):
    if not isinstance(text, str):
        return text
    return unicodedata.normalize("NFKC", text)

df["speech"] = df["speech"].apply(fix_unicode)

def clean_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()

df["speech"] = df["speech"].apply(clean_whitespace)

df.to_parquet(OUTPUT_PATH)
print("Saved:", df.shape)
print("To:", OUTPUT_PATH)
