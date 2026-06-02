from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from utils import load_data, encode_labels, tokenize_dataset
from sklearn.metrics import classification_report
import torch
import sys
MODEL_PATH = sys.argv[1]

dataset = load_data()
dataset, label2id, id2label = encode_labels(dataset)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

dataset = tokenize_dataset(dataset, tokenizer)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

trainer = Trainer(model=model)

predictions = trainer.predict(dataset["test"])
preds = torch.argmax(torch.tensor(predictions.predictions), dim=1)

true_labels = dataset["test"]["labels"]

print(classification_report(true_labels, preds, target_names=list(label2id.keys())))