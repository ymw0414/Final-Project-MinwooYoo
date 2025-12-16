Code Directory

This directory contains the core implementation of the NLP pipeline for the final project.

Each script represents a single stage of the pipeline and is designed to be run
independently, with inputs produced by earlier stages.

Execution Order

Scripts are numbered to reflect the recommended execution sequence:

01_load_speeches.py
02_merge_speaker_map.py
03_add_party_label.py
04_preprocess_text.py
05_train_baseline.py
06_tokenize_and_concat.py
07_train_roberta.py
08_streamlit_app.py

Data Paths

All scripts avoid hardcoded absolute paths.

Raw data locations and output directories are specified via command-line arguments.
A standard directory layout is assumed:

data/
├── raw/
│   └── congressional_speeches/
│       ├── speeches_043.txt
│       ├── speeches_044.txt
│       └── ...
└── processed/


Example usage:

python Code/01_load_speeches.py \
  --raw_dir data/raw/congressional_speeches


Outputs from each stage are written to data/processed/ by default.

Notes

Scripts are intentionally modular and script-based rather than class-based.

Transformer fine-tuning assumes GPU availability.

Experimental variants were removed to keep the codebase focused and reproducible.