from pathlib import Path
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

# =========================================================
# Paths (LOCAL)
# =========================================================
CHUNK_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/data/processed/tokenized_1980s_streaming")
MODEL_SAVE_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/models/roberta_1980s")
EVAL_DIR = Path("C:/Users/ymw04/Dropbox/shifting_slant/evaluation")

MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Load datasets
# =========================================================
print("Collecting chunk directories...")
chunk_dirs = sorted(
    [p for p in CHUNK_DIR.iterdir() if p.is_dir() and p.name.startswith("chunk_")],
    key=lambda x: int(x.name.split("_")[1])
)

if not chunk_dirs:
    raise RuntimeError(f"No chunk_* directories found in {CHUNK_DIR}")

print(f"Found {len(chunk_dirs)} chunks.")
print("Loading chunks and concatenating...")

datasets = [load_from_disk(str(p)) for p in chunk_dirs]
dataset = concatenate_datasets(datasets)

print("Dataset loaded:")
print(dataset)

# =========================================================
# Train/Valid/Test Split
# =========================================================
dataset = dataset.shuffle(seed=42)

train_test = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

train_ds = train_test["train"]
valid_ds = test_valid["train"]
test_ds = test_valid["test"]

cols = ["input_ids", "attention_mask", "labels"]

train_ds.set_format(type="torch", columns=cols)
valid_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

print("Train size:", len(train_ds))
print("Valid size:", len(valid_ds))
print("Test size:", len(test_ds))

# =========================================================
# DataLoaders (optimized for RTX 3080 Ti)
# =========================================================
BATCH_SIZE = 32  # safe for 3080 Ti (12GB VRAM)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =========================================================
# Model / optimizer / scheduler
# =========================================================
print("Loading tokenizer and model...")
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

NUM_EPOCHS = 2  # 1980s dataset is smaller, so 2 epochs OK
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(0.06 * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

scaler = torch.cuda.amp.GradScaler()  # fp16

# =========================================================
# Evaluation function
# =========================================================
def evaluate(model, dataloader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=-1)

            all_labels.extend(batch["labels"].numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_r = f1_score(all_labels, all_preds, pos_label=1)
    return acc, f1_r

# =========================================================
# Training Loop (with progress + ETA)
# =========================================================
print("Starting training...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    start_time = time.time()

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.cuda.amp.autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % 200 == 0:
            percent = (step + 1) / len(train_loader) * 100
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            eta = (len(train_loader) - (step + 1)) / steps_per_sec

            print(
                f"Epoch {epoch+1} | Step {step+1}/{len(train_loader)} "
                f"({percent:.2f} percent) | Loss {total_loss/(step+1):.4f} "
                f"| ETA {eta/60:.2f} min"
            )

    print(f"Epoch {epoch+1} finished. Avg loss = {total_loss/len(train_loader):.4f}")

    val_acc, val_f1 = evaluate(model, valid_loader)
    print(f"Validation acc: {val_acc:.4f}, F1 (R): {val_f1:.4f}")

# =========================================================
# Final Test
# =========================================================
print("Testing model...")
test_acc, test_f1 = evaluate(model, test_loader)
print(f"Test accuracy: {test_acc:.4f} | Test F1 (R): {test_f1:.4f}")

# =========================================================
# Save Model + Metrics
# =========================================================
model.save_pretrained(str(MODEL_SAVE_DIR))
tokenizer.save_pretrained(str(MODEL_SAVE_DIR))

metrics_path = EVAL_DIR / "roberta_1980s_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Test accuracy: {test_acc:.6f}\n")
    f.write(f"Test F1 (R): {test_f1:.6f}\n")

print("Model saved.")
print("Metrics saved.")
print("Done.")
