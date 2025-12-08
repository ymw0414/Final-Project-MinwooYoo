from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load your uploaded model from HuggingFace
model_id = "ymw04/streamlit-demo"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Prediction function
def predict(text):
    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=-1)

    # Return probability list
    return probs[0].tolist()
