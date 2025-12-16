"""
06_tokenize_and_concat.py

Tokenize the paragraph-filtered 1980s dataset using RoBERTa
and save a HuggingFace dataset for downstream fine-tuning.

Outputs:
    data/processed/tokenized_1980s_paragraph_streaming/chunk_*
    data/processed/tokenized_1980s_paragraph_full/
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset, load_from_disk, concatenate_datasets
from transformers import RobertaTokenizerFast
import pandas as pd


def tokenize_batch(batch, tokenizer, max_length):
    return tokenizer(
        batch["speech"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def stream_and_tokenize(
    data_path: Path,
    chunk_dir: Path,
    tokenizer,
    chunk_size: int,
    max_length: int,
):
    parquet = pq.ParquetFile(data_path)
    chunk_paths = []
    chunk_id = 0

    for batch in parquet.iter_batches(batch_size=chunk_size):
        print(f"Processing chunk {chunk_id}...")

        df = batch.to_pandas()
        df["labels"] = df["party"].map({"D": 0, "R": 1})

        ds = Dataset.from_pandas(df[["speech", "labels"]])
        ds = ds.map(
            lambda b: tokenize_batch(b, tokenizer, max_length),
            batched=True,
        )
        ds = ds.remove_columns(["speech"])

        out_path = chunk_dir / f"chunk_{chunk_id}"
        ds.save_to_disk(str(out_path))
        chunk_paths.append(out_path)

        del df, ds
        chunk_id += 1

    return chunk_paths


def main(args):
    data_path = Path(args.data_path)
    chunk_dir = Path(args.chunk_dir)
    final_dir = Path(args.final_dir)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    print("Tokenizing dataset in streaming mode...")
    chunk_paths = stream_and_tokenize(
        data_path=data_path,
        chunk_dir=chunk_dir,
        tokenizer=tokenizer,
        chunk_size=args.chunk_size,
        max_length=args.max_length,
    )

    print("Concatenating chunks...")
    datasets = [load_from_disk(str(p)) for p in chunk_paths]
    merged = concatenate_datasets(datasets)

    print("Saving final tokenized dataset...")
    merged.save_to_disk(str(final_dir))

    print("Done.")
    print(f"Final dataset saved to: {final_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/speeches_clean_1980s_paragraph.parquet",
        help="Input cleaned speech dataset",
    )
    parser.add_argument(
        "--chunk_dir",
        type=str,
        default="data/processed/tokenized_1980s_paragraph_streaming",
        help="Directory to store tokenized chunks",
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="data/processed/tokenized_1980s_paragraph_full",
        help="Directory to store merged tokenized dataset",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=20000,
        help="Number of rows per streaming chunk",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization",
    )
    args = parser.parse_args()
    main(args)
