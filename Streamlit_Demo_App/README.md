# Newspaper Slant Classifier Demo

This repository contains a Streamlit web application that classifies political slant (Democrat vs. Republican) from text using a HuggingFace model. The app is minimal, fast, and easy to deploy on Streamlit Community Cloud.

---

## How It Works

- The model is loaded directly from HuggingFace Hub.
- The user enters text into the Streamlit interface.
- The app returns:
  - Predicted class (Democrat or Republican)
  - Raw probability scores

---

## File Structure

app.py            # Main Streamlit app (UI)
inference.py      # Model loading and inference logic
requirements.txt  # Dependencies for deployment

---

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

---

## Deploy on Streamlit Cloud

1. Upload this project to GitHub.
2. Go to: https://share.streamlit.io
3. Click "New app".
4. Select this GitHub repository.
5. Set the main file to:

app.py

6. Click Deploy.

A public URL will be generated automatically.

---

## Model Used

HuggingFace model:

ymw04/streamlit-demo

This model is automatically downloaded by inference.py during prediction.

---

## Requirements

Listed in requirements.txt:

streamlit  
transformers  
torch  
huggingface_hub

---

## Contact

Feel free to reach out for improvements or troubleshooting.
