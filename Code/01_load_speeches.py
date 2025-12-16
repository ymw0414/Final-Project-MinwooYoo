"""
01_load_speeches.py

Load Congressional speech text files, parse them safely, and
save the merged dataset as a parquet file.

Input:
    Raw speech text files (speeches_043.txt ~ speeches_111.txt)

Output:
    data/processed/speeches_merged.parquet
"""

import pandas as pd
from pathlib import Path
import argparse


def load_speeches(raw_dir: Path) -> pd.DataFrame:
    all_rows = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"speeches_{suffix}.txt"

        if not file.exists():
            print("skip:", file)
            continue

        with open(file, "r", encoding="cp1252") as f:
            next(f)  # skip header

            for line in f:
                parts = line.rstrip("\n").split("|", 1)
                if len(parts) != 2:
                    continue

                speech_id, speech = parts
                all_rows.append((speech_id.strip(), speech.strip(), suffix))

    return pd.DataFrame(all_rows, columns=["speech_id", "speech", "file_id"])


def main(args):
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_speeches(raw_dir)

    out_path = out_dir / "speeches_merged.parquet"
    df.to_parquet(out_path)

    print("Done.", df.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw Congressional speech text files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    main(args)
