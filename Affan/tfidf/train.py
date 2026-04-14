import os
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ──────────────────────────────────────────────────────────────────
PATH        = "./../Dataset/Dataset_large/"
OUTPUT_DIR  = "./output_idf"
EPOCHS      = 50
BATCH_SIZE  = 8
LR          = 1e-3
MAX_FEAT    = 50000
NGRAM       = (1, 4)

LAYER_CONFIGS = [
    (512,   256),
    (1024,  512),
    (2048,  1024),
    (4096,  2048),
    (8192,  4096),
    (16384, 8192),
    (32768, 16384),
]

# ── Dataset ──────────────────────────────────────────────────────────────────
class SparseDataset(Dataset):
    def __init__(self, X_sparse, y):
        self.X = X_sparse
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx].toarray().squeeze(), dtype=torch.float32)
        return x, self.y[idx]

# ── Model ────────────────────────────────────────────────────────────────────
class TFIDF_NN(nn.Module):
    def __init__(self, input_dim, l1, l2, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, l1),
            nn.ReLU(),
            nn.BatchNorm1d(l1),
            nn.Dropout(0.3),
            nn.Linear(l1, l2),
            nn.ReLU(),
            nn.BatchNorm1d(l2),
            nn.Dropout(0.3),
            nn.Linear(l2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Training helpers ─────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(train):
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{(correct/total):.4f}"
            })

    return total_loss / total, correct / total

def train_config(config, input_dim, num_classes, train_loader, val_loader, device, out_dir):
    l1, l2 = config
    config_name = f"{l1}_{l2}"

    print(f"\n🚀 Starting config: {config_name}")
    ckpt_dir = out_dir / "checkpoints" / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model     = TFIDF_NN(input_dim, l1, l2, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        print(f"\n📚 Epoch {epoch}/{EPOCHS}")

        t_loss, t_acc = run_epoch(
            model, train_loader, criterion, optimizer, device,
            train=True, desc=f"Train {epoch}"
        )

        v_loss, v_acc = run_epoch(
            model, val_loader, criterion, optimizer, device,
            train=False, desc=f"Val   {epoch}"
        )

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(f"✅ Epoch {epoch} done")
        print(f"   Train → loss: {t_loss:.4f}, acc: {t_acc:.4f}")
        print(f"   Val   → loss: {v_loss:.4f}, acc: {v_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, ckpt_dir / f"epoch_{epoch:03d}.pt")

    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history["val_loss"][-1], history["val_acc"][-1], history

# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_test(model, test_loader, label_encoder, device, out_dir):
    print("\n🧪 Running final test evaluation...")
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
            logits = model(X_batch.to(device))
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_targets.extend(y_batch.numpy())

    y_pred = np.array(all_preds)
    y_test = np.array(all_targets)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Final Accuracy: {acc:.4f}")

    return acc

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("🌟 Starting pipeline...")
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Device: {device}")

    # ── Load data ──
    print("\n📂 Loading datasets...")
    train_df = pd.read_parquet(Path(PATH) / "train" / "train.parquet")
    val_df   = pd.read_parquet(Path(PATH) / "val" / "val.parquet")
    test_df  = pd.read_parquet(Path(PATH) / "test" / "test.parquet")

    print(f"Train size: {len(train_df)}")
    print(f"Val size:   {len(val_df)}")
    print(f"Test size:  {len(test_df)}")

    text_col, label_col = "generation", "model"

    # ── Encode labels ──
    print("\n🔠 Encoding labels...")
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val   = le.transform(val_df[label_col])
    y_test  = le.transform(test_df[label_col])

    print(f"Classes: {len(le.classes_)}")

    # ── TF-IDF ──
    print("\n🧠 Building TF-IDF...")
    tfidf = TfidfVectorizer(max_features=MAX_FEAT, ngram_range=NGRAM)

    X_train_sp = tfidf.fit_transform(train_df[text_col])
    X_val_sp   = tfidf.transform(val_df[text_col])
    X_test_sp  = tfidf.transform(test_df[text_col])

    print(f"TF-IDF shape: {X_train_sp.shape}")

    # ── Loaders ──
    print("\n📦 Creating DataLoaders...")
    train_loader = DataLoader(SparseDataset(X_train_sp, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SparseDataset(X_val_sp,   y_val),   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(SparseDataset(X_test_sp,  y_test),  batch_size=BATCH_SIZE)

    input_dim = X_train_sp.shape[1]
    num_classes = len(le.classes_)

    # ── Training loop ──
    print("\n🏁 Starting hyperparameter search...")
    best_loss = float("inf")
    best_model = None

    for config in tqdm(LAYER_CONFIGS, desc="Configs"):
        start = time.time()

        model, val_loss, val_acc, _ = train_config(
            config, input_dim, num_classes,
            train_loader, val_loader, device, out_dir
        )

        print(f"⏱️ Time: {time.time() - start:.1f}s")

        if val_loss < best_loss:
            print("🔥 New best model found!")
            best_loss = val_loss
            best_model = model

    print("\n🏆 Training complete!")

    # ── Test ──
    evaluate_test(best_model, test_loader, le, device, out_dir)

if __name__ == "__main__":
    main()