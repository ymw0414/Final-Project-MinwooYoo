"""
06_tokenize_and_concat_1980plus_unfiltered.py

This script tokenizes the unfiltered 1981-plus dataset (file_id >= 97) and
constructs a HuggingFace-ready dataset.

Process:
    1. Stream the speeches_clean_1981plus_unfiltered.parquet file in chunks
    2. Tokenize each chunk with the RoBERTa tokenizer (max_length=256)
    3. Save each tokenized chunk as chunk_i/
    4. Concatenate all chunks into a single merged HF dataset

Notes:
    - No paragraph-level filtering (all speeches included)
    - Labels: D -> 0, R -> 1

Output:
    data/processed/tokenized_1981plus_unfiltered_streaming/chunk_*
    data/processed/tokenized_1981plus_unfiltered_full/
"""

import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import RobertaTokenizerFast
import pandas as pd
import shutil

# ---------------------------------------------------------
# Paths (LOCAL WINDOWS VERSION)
# ---------------------------------------------------------
DATA_PATH = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\speeches_clean_1981plus_unfiltered.parquet"
)

CHUNK_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\tokenized_1981plus_unfiltered_streaming"
)
FINAL_DIR = Path(
    r"C:\Users\ymw04\Dropbox\shifting_slant\data\processed\tokenized_1981plus_unfiltered_full"
)

CHUNK_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Load tokenizer
# ---------------------------------------------------------
print("Loading tokenizer...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")


def tokenize_batch(batch):
    """Tokenize a batch of texts with fixed max_length."""
    return tokenizer(
        batch["speech"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )


# ---------------------------------------------------------
# Step 1: Stream → tokenize → save chunks
# ---------------------------------------------------------
print("Opening parquet for streaming...")
parquet = pq.ParquetFile(DATA_PATH)

CHUNK_SIZE = 50_000
chunk_id = 0
chunk_paths = []

for batch in parquet.iter_batches(batch_size=CHUNK_SIZE):
    print(f"Processing raw chunk {chunk_id}...")

    df = batch.to_pandas()

    # Keep only sessions >= 97
    df["file_id"] = df["file_id"].astype(int)
    df = df[df["file_id"] >= 97]

    if df.empty:
        print("  No valid rows in this batch. Skipping.")
        chunk_id += 1
        continue

    # Map labels
    df["labels"] = df["party"].map({"D": 0, "R": 1})

    # Convert to HF dataset
    ds = Dataset.from_pandas(df[["speech", "labels"]])
    ds = ds.map(tokenize_batch, batched=True)
    ds = ds.remove_columns(["speech"])

    out_path = CHUNK_DIR / f"chunk_{chunk_id}"
    ds.save_to_disk(str(out_path))

    print(f"Saved chunk {chunk_id} → {out_path}")
    chunk_paths.append(out_path)

    del df, ds
    chunk_id += 1


print(f"Finished tokenizing into {len(chunk_paths)} chunks.")


# ---------------------------------------------------------
# Step 2: Load chunks → concatenate → save final dataset
# ---------------------------------------------------------
print("Loading and concatenating chunks...")

datasets = [load_from_disk(str(p)) for p in chunk_paths]
merged = concatenate_datasets(datasets)

print("Saving final merged tokenized dataset...")
merged.save_to_disk(str(FINAL_DIR))

print("All steps complete.")
print(f"Final dataset saved to: {FINAL_DIR}")
