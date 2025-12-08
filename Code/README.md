## Naming Conventions

Before listing the scripts, here is what each dataset label means:

- **1980plus**  
  Includes all speeches from 1981 to the 2010s (wide timespan).  
  Used for a broader TF-IDF baseline and broad RoBERTa comparison.

- **1980s**  
  Restricts the dataset to speeches from **1981–1989 only**.  
  This is the target period for your economics application.

- **paragraph_filtered**  
  Applies strict quality filters:  
    - ≥ 2 sentences  
    - ≥ 200 characters  
  Produces a high-quality subset (≈0.39M speeches) that yields the best performance.

- **epoch1 / epoch2 / epoch3**  
  RoBERTa fine-tuning runs for 1, 2, or 3 epochs.  
  Same dataset; only training duration differs.

---

## Scripts Overview

### 1. Load & Merge Raw Data
- **01_load_speeches.py**  
  Load raw Congressional speeches and combine into a unified dataframe.
- **02_merge_speaker_map.py**  
  Load & merge SpeakerMap metadata (speaker → attributes).
- **03_add_party_label.py**  
  Attach party labels (D/R) to each speech using SpeakerMap.

---

### 2. Text Preprocessing
- **04_preprocess_text_1980plus_unfiltered.py**  
  Clean text and apply minimum filter (≥30 chars) for the broad 1980+ dataset.
- **04_preprocess_text_1980s_unfiltered.py**  
  Clean and extract the 1981–1989 subset using the minimum filter.
- **04_preprocess_text_1980s_paragraph_filtered.py**  
  Apply strict filtering (≥2 sentences & ≥200 chars) to produce the high-quality 1980s dataset.

---

### 3. Baseline Training (TF-IDF)
- **05_train_baseline_1980plus_unfiltered.py**  
  Train TF-IDF + Logistic Regression on the broad 1980+ dataset.
- **05_train_baseline_1980s_unfiltered.py**  
  Train TF-IDF baseline on the 1980s subset.
- **05_train_baseline_1980s_paragraph_filtered.py**  
  Train TF-IDF baseline on the high-quality paragraph-filtered 1980s dataset.

---

### 4. Tokenization for RoBERTa
- **06_tokenize_and_concat_1980plus_unfiltered.py**  
- **06_tokenize_and_concat_1980s_unfiltered.py**  
- **06_tokenize_and_concat_1980s_paragraph_filtered.py**  
Tokenize text using RoBERTa tokenizer and build train/val/test splits.

---

### 5. RoBERTa Fine-Tuning
- **07_train_roberta_1980plus_unfiltered.py**  
- **07_train_roberta_1980s_unfiltered.py**  
- **07_train_roberta_1980s_paragraph_filtered_epoch1.py**  
- **07_train_roberta_1980s_paragraph_filtered_epoch2.py**  
- **07_train_roberta_1980s_paragraph_filtered_epoch3.py**  
Fine-tune `roberta-base` on each dataset variant, with 1–3 epoch runs.
