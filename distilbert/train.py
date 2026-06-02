from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import load_data, encode_labels, tokenize_dataset
import os
import torch
import sys

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
print("Using model:", MODEL_NAME)

# ✅ Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL_NAME = "distilbert-base-uncased"

# Load dataset
dataset = load_data()
dataset, label2id, id2label = encode_labels(dataset)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenization
dataset = tokenize_dataset(dataset, tokenizer)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=12,
    id2label=id2label,
    label2id=label2id
)

# ✅ Move model to GPU explicitly (important)
model.to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,                 # keep only last 2 checkpoints
    num_train_epochs=5,
    per_device_train_batch_size=8,      # safe for 4GB GPU
    per_device_eval_batch_size=8,
    logging_dir="results/logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),     # ✅ faster on GPU
    report_to="none"                    # avoid extra logging overhead
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# ✅ Resume if checkpoint exists
trainer.train()

# Save final model
model_dir = f"models/{MODEL_NAME.replace('/', '_')}"
os.makedirs(model_dir, exist_ok=True)

trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)