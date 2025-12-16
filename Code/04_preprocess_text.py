"""
04_preprocess_text.py

Preprocess congressional speech data by:
1. Restricting to 1980s speeches (97th–100th Congress)
2. Applying paragraph-level filtering (>=2 sentences, >=200 characters)
3. Cleaning text via Unicode and whitespace normalization

Output:
    data/processed/speeches_clean_1980s_paragraph.parquet
"""

import pandas as pd
import re
import unicodedata
from pathlib import Path
import argparse


def fix_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert file_id to int and keep 1980s (97–100th Congress)
    df["file_id"] = df["file_id"].astype(int)
    df = df[df["file_id"].isin([97, 98, 99, 100])]

    # Keep valid party labels
    df = df[df["party"].isin(["D", "R"])]

    # Drop missing or empty text
    df = df.dropna(subset=["speech"])
    df = df[df["speech"].str.strip().str.len() > 0]

    # Paragraph-level filtering
    df["num_sent"] = df["speech"].str.count(r"[\.!?]")
    df = df[(df["num_sent"] >= 2) & (df["speech"].str.len() >= 200)]

    # Text cleaning
    df["speech"] = df["speech"].apply(fix_unicode)
    df["speech"] = df["speech"].apply(clean_whitespace)

    return df


def main(args):
    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    print("Loaded:", df.shape)

    df_clean = preprocess(df)

    out_path = out_dir / "speeches_clean_1980s_paragraph.parquet"
    df_clean.to_parquet(out_path)

    print("Saved cleaned dataset:", df_clean.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/speeches_with_party.parquet",
        help="Path to speeches with party labels",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    args = parser.parse_args()
    main(args)
