"""
02_merge_speaker_map.py

Load SpeakerMap files, merge speaker metadata, and
save the combined table as a parquet file.

Input:
    SpeakerMap_043.txt ~ SpeakerMap_111.txt

Output:
    data/processed/speaker_map.parquet
"""

import pandas as pd
from pathlib import Path
import argparse


def load_speaker_map(raw_dir: Path) -> pd.DataFrame:
    all_rows = []

    for i in range(43, 112):
        suffix = f"{i:03d}"
        file = raw_dir / f"SpeakerMap_{suffix}.txt"

        if not file.exists():
            print("skip:", file)
            continue

        df = pd.read_csv(
            file,
            sep="|",
            header=None,
            names=["speech_id", "speaker", "state", "party"],
            dtype=str,
            encoding="cp1252",
        )
        all_rows.append(df)

    return pd.concat(all_rows, ignore_index=True)


def main(args):
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speaker_map = load_speaker_map(raw_dir)

    out_path = out_dir / "speaker_map.parquet"
    speaker_map.to_parquet(out_path)

    print("Done.", speaker_map.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_dir",
        type=str,
        required=True,
        help="Path to raw SpeakerMap files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data",
    )
    args = parser.parse_args()
    main(args)
