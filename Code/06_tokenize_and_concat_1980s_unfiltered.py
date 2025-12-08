"""
06_tokenize_and_concat_1980s_unfiltered.py

This script performs chunked tokenization of ALL 1980s speeches WITHOUT
paragraph filtering. It processes the file in memory-efficient batches,
tokenizes each chunk, saves them separately, and finally concatenates
all tokenized chunks into one HuggingFace dataset.

Dataset:
    speeches_clean_1980s_unfiltered.parquet

Pipeline:
    1. Stream or load rows
    2. Tokenize in chunks
    3. Save each chunk as chunk_i/
    4. Concatenate into a final dataset

Outputs:
    data/processed/tokenized_1980s_unfiltered_streaming/chunk_*
    data/processed/tokenized_1980s_unfiltered_full/
"""

import pyarrow.parquet as pq
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import RobertaTokenizerFast
import pandas as pd
import shutil


# ---------------------------------------------------------
# Paths (local)
# ---------------------------------------------------------
DATA_PATH = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/speeches_clean_1980s_unfiltered.parquet")

CHUNK_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/tokenized_1980s_unfiltered_streaming")
FINAL_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/tokenized_1980s_unfiltered_full")

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
# Step 1: Stream -> tokenize -> save chunks
# ---------------------------------------------------------
print("Opening parquet...")
parquet = pq.ParquetFile(DATA_PATH)

CHUNK_SIZE = 40_000   # good for your 3080 Ti
chunk_id = 0
chunk_paths = []

for batch in parquet.iter_batches(batch_size=CHUNK_SIZE):
    print(f"Processing raw chunk {chunk_id}...")

    df = batch.to_pandas()

    # Labels
    df["labels"] = df["party"].map({"D": 0, "R": 1})

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
# Step 2: Load tokenized chunks -> merge -> save final dataset
# ---------------------------------------------------------
print("Loading chunks and concatenating...")

datasets = [load_from_disk(str(p)) for p in chunk_paths]
merged = concatenate_datasets(datasets)

print("Saving final merged tokenized dataset...")
merged.save_to_disk(str(FINAL_DIR))

print("All steps complete.")
print(f"Final dataset saved to: {FINAL_DIR}")
