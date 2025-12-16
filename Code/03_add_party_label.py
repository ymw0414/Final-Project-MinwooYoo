"""
03_add_party_label.py

Merge speech text with SpeakerMap metadata to attach
party labels (D/R) to each speech.

Input:
    data/processed/speeches_merged.parquet
    data/processed/speaker_map.parquet

Output:
    data/processed/speeches_with_party.parquet
"""

import pandas as pd
from pathlib import Path
import argparse


def add_party_label(
    speeches_path: Path, speaker_map_path: Path
) -> pd.DataFrame:
    speeches = pd.read_parquet(speeches_path)
    speaker_map = pd.read_parquet(speaker_map_path)

    merged = speeches.merge(
        speaker_map, on="speech_id", how="left"
    )
    return merged


def main(args):
    speeches_path = Path(args.speeches_path)
    speaker_map_path = Path(args.speaker_map_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = add_party_label(speeches_path, speaker_map_path)

    out_path = out_dir / "speeches_with_party.parquet"
    merged.to_parquet(out_path)

    print("Done.", merged.shape)
    print("Saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speeches_path",
        type=str,
        default="data/processed/speeches_merged.parquet",
        help="Path to merged speech text parquet file",
    )
    parser.add_argument(
        "--speaker_map_path",
        type=str,
        default="data/processed/speaker_map.parquet",
        help="Path to speaker metadata parquet file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Output directory",
    )
    args = parser.parse_args()
    main(args)
