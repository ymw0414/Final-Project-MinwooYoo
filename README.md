Final-Project-Minwoo Yoo

Final project repository for DATS 6312: Natural Language Processing.

This project builds an end-to-end NLP pipeline that classifies the partisan leaning
(Democrat vs Republican) of political text, using U.S. Congressional speeches as training data.

The pipeline covers data construction, preprocessing, baseline modeling, transformer fine-tuning,
and an interactive Streamlit demo.

Project Overview

Task: Binary political text classification (D vs R)

Data: U.S. Congressional speeches

Models:

TF-IDF + Logistic Regression (baseline)

RoBERTa-base (fine-tuned)

Output:

Trained classifiers

Streamlit app for real-time inference

Repository Structure

All core code is located in the Code/ directory.
Scripts are numbered to reflect execution order.

Code/
├── 01_load_speeches.py
├── 02_merge_speaker_map.py
├── 03_add_party_label.py
├── 04_preprocess_text.py
├── 05_train_baseline.py
├── 06_tokenize_and_concat.py
├── 07_train_roberta.py
└── 08_streamlit_app.py

Pipeline Description
1. Data Construction

01_load_speeches.py
Load raw Congressional speech text and construct the base dataset.

02_merge_speaker_map.py
Merge speaker metadata to associate speeches with individual legislators.

03_add_party_label.py
Attach binary party labels (Democrat vs Republican) to each speech.

2. Text Preprocessing

04_preprocess_text.py
Clean raw text and apply quality filters to produce the final modeling dataset.

3. Baseline Model

05_train_baseline.py
Train a TF-IDF + Logistic Regression classifier as a transparent baseline.

4. Transformer Tokenization

06_tokenize_and_concat.py
Tokenize text using the RoBERTa tokenizer and create train/validation/test splits.

5. RoBERTa Fine-Tuning

07_train_roberta.py
Fine-tune roberta-base for partisan text classification.

6. Interactive Demo

08_streamlit_app.py
Streamlit application that takes arbitrary text input and outputs predicted political leaning.

Requirements

Python 3.9+ is recommended.

Key libraries:

pandas

numpy

scikit-learn

torch

transformers

datasets

streamlit

tqdm

Dependencies can be installed via:

pip install -r requirements.txt

Data

Due to size constraints, raw data is not included in this repository.

The project assumes access to publicly available U.S. Congressional speech data
and speaker metadata (e.g., Congressional Record–style text and SpeakerMap-style files).

Users must place raw data files in the expected data directory before running the pipeline.

Reproducibility and Usage

Scripts are intended to be run sequentially:

01_load_speeches.py
02_merge_speaker_map.py
03_add_party_label.py
04_preprocess_text.py
05_train_baseline.py
06_tokenize_and_concat.py
07_train_roberta.py


To launch the demo:

streamlit run Code/08_streamlit_app.py


Transformer fine-tuning assumes GPU availability.

Notes

This repository contains the final cleaned pipeline used for the course project.

Alternative preprocessing choices and experimental variants were explored during development
but were removed for clarity and reproducibility.