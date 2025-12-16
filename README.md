# Political Text Classification (DATS 6312: Natural Language Processing)

🔗 Live Demo  
https://final-project-minwooyoo-bcbbuifbzcrqmmalx8z3gn.streamlit.app/

---

Overview

End-to-end NLP pipeline for partisan classification of political text
(Democrat vs Republican) using U.S. Congressional speeches.

The project covers data construction, text preprocessing, baseline modeling,
transformer fine-tuning, and an interactive Streamlit demo for real-time inference.

---

Project Overview

Task  
Binary political text classification (D vs R)

Data  
U.S. Congressional speeches

Models  
- TF-IDF + Logistic Regression (baseline)  
- RoBERTa-base (fine-tuned)

Outputs  
- Trained classifiers  
- Streamlit app for real-time inference

---

Repository Structure

All core code is located in the Code directory.
Scripts are numbered to reflect execution order.

Code/
- 01_load_speeches.py
- 02_merge_speaker_map.py
- 03_add_party_label.py
- 04_preprocess_text.py
- 05_train_baseline.py
- 06_tokenize_and_concat.py
- 07_train_roberta.py
- 08_streamlit_app.py

---

Pipeline Description

1. Data Construction

01_load_speeches.py  
Load raw Congressional speech text and construct the base dataset.

02_merge_speaker_map.py  
Merge speaker metadata to associate speeches with legislators.

03_add_party_label.py  
Attach binary party labels (Democrat vs Republican).

2. Text Preprocessing

04_preprocess_text.py  
Clean raw text and apply quality filters to produce the modeling dataset.

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

---

Requirements

Python 3.9+ is recommended.

Key libraries  
- pandas  
- numpy  
- scikit-learn  
- torch  
- transformers  
- datasets  
- streamlit  
- tqdm  
- matplotlib  

Install dependencies with:
pip install -r requirements.txt

---

Data

Due to size constraints, raw data is not included.

The pipeline assumes access to publicly available U.S. Congressional speech data
and speaker metadata (e.g., Congressional Record–style text and SpeakerMap-style files).

Raw data files should be placed under:
data/raw/

---

Reproducibility and Usage

Scripts should be run sequentially:

python Code/01_load_speeches.py  
python Code/02_merge_speaker_map.py  
python Code/03_add_party_label.py  
python Code/04_preprocess_text.py  
python Code/05_train_baseline.py  
python Code/06_tokenize_and_concat.py  
python Code/07_train_roberta.py  

To launch the demo:

streamlit run Code/08_streamlit_app.py

Transformer fine-tuning assumes GPU availability.

---

Notes

This repository contains the final cleaned pipeline used for the project.

Alternative preprocessing choices and experimental variants were explored
during development but were removed for clarity and reproducibility.

This project was developed as part of DATS 6312: Natural Language Processing.
