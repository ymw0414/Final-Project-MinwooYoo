import streamlit as st
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Political Speech Classifier",
    page_icon="🏛️",
    layout="centered"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
# IMPORTANT: use HuggingFace Hub path, NOT a local folder
MODEL_PATH = "minwooyoo/roberta_1980s_paragraph_filtered_epoch3"

@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("🏛️ Political Speech Classifier Demo")
st.info("Model was trained exclusively on speeches from **1981–1989 (97th–100th Congress)**.")

speech = st.text_area(
    "Enter speech text:",
    placeholder="Type a Congressional-style political speech excerpt...",
    height=200
)

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("Run Classification"):
    if len(speech.strip()) == 0:
        st.warning("Please enter text before running classification.")
    else:
        inputs = tokenizer(
            speech,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]
            pred = torch.argmax(probs).item()

        party = "Democrat (D)" if pred == 0 else "Republican (R)"
        score = float(probs[pred])

        color = "#1d67ff" if pred == 0 else "#cc0000"

        # -----------------------------
        # CLEAN RESULT CARD
        # -----------------------------
        st.markdown(
            f"""
            <div style="
                background-color:#f8f9fa;
                padding:20px;
                border-radius:10px;
                border-left:8px solid {color};
                margin-top:20px;">
                <h3 style="margin-bottom:10px;">
                    Prediction: <span style="color:{color};">{party}</span>
                </h3>
                <p style="font-size:16px; margin-bottom:6px;">
                    Confidence Score: <b>{score:.4f}</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(score)

        st.caption(
            f"The model predicts that this speech resembles typical **{party}** rhetorical patterns."
        )
