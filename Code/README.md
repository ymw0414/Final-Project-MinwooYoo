Final-Project-MinwooYoo

Final project repository for DATS 6312 (Natural Language Processing).

This project builds an end-to-end NLP pipeline that classifies the partisan leaning (Democratic vs Republican) of political text, using U.S. Congressional speeches as training data.

Project Overview

Task: Binary classification of political text (D vs R)

Data: U.S. Congressional speeches

Methods:

TF-IDF + Logistic Regression (baseline)

RoBERTa-base fine-tuning

Output:

Trained classification models

Streamlit app for interactive inference

Pipeline Structure

Scripts are numbered to reflect execution order.

1. Data Construction

01_load_speeches.py
Load raw Congressional speech text and construct the base dataset.

02_merge_speaker_map.py
Merge speaker metadata to associate speeches with individual legislators.

03_add_party_label.py
Attach binary party labels (Democrat vs Republican) to each speech.

2. Text Preprocessing

04_preprocess_text.py
Clean raw text and apply quality filters to construct the final modeling dataset.

3. Baseline Model

05_train_baseline.py
Train a TF-IDF + Logistic Regression classifier as a baseline model.

4. Transformer Tokenization

06_tokenize_and_concat.py
Tokenize text using the RoBERTa tokenizer and create train/validation/test splits.

5. RoBERTa Fine-Tuning

07_train_roberta.py
Fine-tune roberta-base for partisan text classification.

6. Interactive Application

08_streamlit_app.py
Streamlit application that takes text input and outputs predicted political leaning.

Usage

Scripts are intended to be run sequentially from 01 to 08.
Transformer training assumes GPU availability.

Notes

This repository contains the final cleaned pipeline used for the course project.
Intermediate experiments and alternative variants were removed for clarity.