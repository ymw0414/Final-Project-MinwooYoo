from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Correct HuggingFace model ID
model_id = "ymw0414/roberta_1980s_paragraph_filtered_epoch3"

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

    # Probability list
    return probs[0].tolist()
