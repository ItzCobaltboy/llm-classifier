"""
utils.py — Shared utilities for LLM-authorship classification.

Contains:
  - Reproducibility setup
  - Stylometric feature extraction
  - PyTorch Dataset + DataLoader factory
  - Model architecture (LLMClassifier)
  - Training / evaluation loop
  - Prediction + SHAP explainability
  - Checkpoint save/load helpers
"""

import os
import re
import pickle
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import textstat
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# ── Constants ──────────────────────────────────────────────────────────────────
SEED = 42
TEXT_COL  = "text"
LABEL_COL = "generated_by"
STYLO_DIM = 15   # must match feature count in stylometric_features()

STOPWORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "up", "about", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "so", "yet", "not", "no",
    "it", "its", "this", "that",
])

HEDGE_WORDS = frozenset([
    "perhaps", "maybe", "might", "could", "possibly",
    "likely", "somewhat", "generally", "often",
])


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Multi-GPU helper ───────────────────────────────────────────────────────────
def wrap_model_multi_gpu(model: nn.Module) -> nn.Module:
    """Wrap with DataParallel when multiple GPUs are available."""
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"  ✦ Using {n_gpus} GPUs via DataParallel")
        model = nn.DataParallel(model)
    return model


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the base model, stripping DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


# ── Stylometric features ───────────────────────────────────────────────────────
def stylometric_features(text: str) -> np.ndarray:
    """
    Extract 15 hand-crafted stylometric features from a text string.
    Returns a float32 numpy array of shape (STYLO_DIM,).
    """
    words     = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    n_words = len(words) + 1e-8
    n_sents = len(sentences) + 1e-8

    # --- Lexical ---
    avg_word_len    = np.mean([len(w) for w in words]) if words else 0.0
    type_token_ratio = len(set(words)) / n_words
    stopword_ratio  = sum(1 for w in words if w in STOPWORDS) / n_words
    hedge_ratio     = sum(1 for w in words if w in HEDGE_WORDS) / n_words
    upper_ratio     = sum(1 for w in re.findall(r'\b[A-Z]{2,}\b', text)) / n_words

    # --- Syntactic / sentence-level ---
    sent_lens    = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lens) if sent_lens else 0.0
    sent_len_std = np.std(sent_lens)  if sent_lens else 0.0

    # --- Punctuation & formatting ---
    punct_rate   = sum(1 for c in text if c in '.,!?;:') / (len(text) + 1e-8)
    bullet_rate  = len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE)) / n_sents
    question_rate = text.count('?') / n_sents
    paren_rate   = text.count('(') / n_words

    # --- Readability ---
    flesch          = textstat.flesch_reading_ease(text)
    syllable_score  = textstat.syllable_count(text) / n_words
    fog_index       = textstat.gunning_fog(text)

    # --- Diversity ---
    bigrams          = list(zip(words, words[1:]))
    bigram_diversity = len(set(bigrams)) / (len(bigrams) + 1e-8)

    return np.array([
        avg_word_len, type_token_ratio, avg_sent_len, punct_rate,
        flesch, stopword_ratio, sent_len_std, bullet_rate,
        question_rate, paren_rate, upper_ratio, bigram_diversity,
        syllable_score, fog_index, hedge_ratio,
    ], dtype=np.float32)


def compute_stylo_matrix(texts: list[str]) -> np.ndarray:
    """Compute stylometric features for a list of texts."""
    return np.stack([
        stylometric_features(t)
        for t in tqdm(texts, desc="  Stylometric features")
    ])


# ── PyTorch Dataset ────────────────────────────────────────────────────────────
class LLMDataset(Dataset):
    def __init__(self, texts, labels, stylo_feats, tokenizer, max_len: int):
        self.texts     = texts
        self.labels    = labels
        self.stylo     = stylo_feats
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length     = self.max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "stylo":          torch.tensor(self.stylo[idx]),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def make_loader(
    df_split,
    stylo_arr: np.ndarray,
    tokenizer,
    max_len: int,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    ds = LLMDataset(
        texts       = df_split[TEXT_COL].tolist(),
        labels      = df_split["label"].tolist(),
        stylo_feats = stylo_arr,
        tokenizer   = tokenizer,
        max_len     = max_len,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


# ── Model ──────────────────────────────────────────────────────────────────────
class LLMClassifier(nn.Module):
    """
    Transformer encoder (CLS token) concatenated with stylometric features,
    fed into a two-layer MLP classifier.
    """
    def __init__(self, model_name: str, stylo_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden_size   = self.encoder.config.hidden_size  # e.g. 768
        combined_size = hidden_size + stylo_dim

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, stylo):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = enc_out.last_hidden_state[:, 0, :]       # (batch, hidden)
        x       = torch.cat([cls_vec, stylo], dim=1)       # (batch, hidden + stylo_dim)
        return self.classifier(x)                          # (batch, num_classes)


# ── Weighted loss ──────────────────────────────────────────────────────────────
def build_criterion(train_labels: np.ndarray, num_classes: int, device: torch.device) -> nn.Module:
    class_weights = compute_class_weight(
        class_weight = "balanced",
        classes      = np.arange(num_classes),
        y            = train_labels,
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    return nn.CrossEntropyLoss(weight=weights_tensor)


# ── Optimizer ─────────────────────────────────────────────────────────────────
def build_optimizer(
    model: nn.Module,
    lr: float,
    use_differential_lr: bool,
    weight_decay: float = 0.01,
) -> AdamW:
    """
    use_differential_lr=False → single uniform LR for all parameters.
    use_differential_lr=True  → lower LR (1e-5) for encoder, higher (3e-4) for head.
    """
    base = unwrap_model(model)
    if use_differential_lr:
        return AdamW([
            {"params": base.encoder.parameters(),    "lr": 1e-5},
            {"params": base.classifier.parameters(), "lr": 3e-4},
        ], weight_decay=weight_decay)
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


# ── Training / evaluation ──────────────────────────────────────────────────────
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    training: bool = True,
) -> tuple[float, float]:
    model.train() if training else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc="  Train" if training else "  Val", leave=False):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            stylo     = batch["stylo"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids, attn_mask, stylo)
            loss   = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    epochs: int, checkpoint_path: str, device: torch.device,
) -> dict:
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler, device, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, scheduler, device, training=False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        saved = ""
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(unwrap_model(model).state_dict(), checkpoint_path)
            saved = "  ← saved"

        print(
            f"  Epoch {epoch:>2}/{epochs}  "
            f"train loss: {tr_loss:.4f}  acc: {tr_acc:.3f}  |  "
            f"val loss: {vl_loss:.4f}  acc: {vl_acc:.3f}{saved}"
        )

    return history


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Testing"):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["stylo"].to(device),
            )
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].numpy())

    return np.array(all_labels), np.array(all_preds)


def print_eval_report(all_labels, all_preds, class_names):
    print(f"  Test Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"  Macro F1      : {f1_score(all_labels, all_preds, average='macro'):.4f}")
    print()
    print(classification_report(all_labels, all_preds, target_names=class_names))


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_training_curves(history: dict, run_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs_range, history["train_loss"], label="Train", marker="o")
    axes[0].plot(epochs_range, history["val_loss"],   label="Val",   marker="o")
    axes[0].set_title(f"Loss — {run_name}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs_range, history["train_acc"], label="Train", marker="o")
    axes[1].plot(epochs_range, history["val_acc"],   label="Val",   marker="o")
    axes[1].set_title(f"Accuracy — {run_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, class_names: list, run_name: str) -> None:
    cm = confusion_matrix(all_labels, all_preds)
    n  = len(class_names)
    plt.figure(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix — {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_text(text: str, model, tokenizer, scaler, label_encoder, max_len: int, device) -> dict:
    model.eval()
    enc = tokenizer(text, max_length=max_len, padding="max_length",
                    truncation=True, return_tensors="pt")
    stylo_feat = torch.tensor(
        scaler.transform([stylometric_features(text)]), dtype=torch.float
    ).to(device)

    with torch.no_grad():
        logits = unwrap_model(model)(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
            stylo_feat,
        )
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    classes    = label_encoder.classes_
    pred_class = classes[np.argmax(probs)]
    return {
        "prediction": pred_class,
        "confidence": f"{max(probs) * 100:.1f}%",
        "all_probs":  {c: f"{p * 100:.1f}%" for c, p in zip(classes, probs)},
    }


# ── SHAP explainability ────────────────────────────────────────────────────────
def run_shap_explanation(
    model, tokenizer, scaler, train_df, test_df,
    max_len: int, batch_size: int, device,
) -> None:
    base_model = unwrap_model(model)

    def _get_combined_features(texts, stylo_arr):
        """Extract [CLS] + stylometric feature vectors for SHAP."""
        all_feats = []
        ds = LLMDataset(texts, [0] * len(texts), stylo_arr, tokenizer, max_len)
        dl = DataLoader(ds, batch_size=batch_size)
        base_model.eval()
        with torch.no_grad():
            for batch in dl:
                cls_vec = base_model.encoder(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                ).last_hidden_state[:, 0, :]
                feats = torch.cat([cls_vec, batch["stylo"].to(device)], dim=1)
                all_feats.append(feats.cpu().numpy())
        return np.vstack(all_feats)

    # Background sample for SHAP kernel
    bg_texts = train_df[TEXT_COL].sample(50, random_state=SEED).tolist()
    bg_stylo = scaler.transform(compute_stylo_matrix(bg_texts))
    bg_feats = _get_combined_features(bg_texts, bg_stylo)

    # Explanation sample from test set
    ex_texts = test_df[TEXT_COL].sample(20, random_state=SEED).tolist()
    ex_stylo = scaler.transform(compute_stylo_matrix(ex_texts))
    ex_feats = _get_combined_features(ex_texts, ex_stylo)

    def _clf_head(x):
        t = torch.tensor(x, dtype=torch.float).to(device)
        with torch.no_grad():
            return torch.softmax(base_model.classifier(t), dim=1).cpu().numpy()

    explainer   = shap.KernelExplainer(_clf_head, shap.sample(bg_feats, 30))
    shap_values = explainer.shap_values(ex_feats[:5], nsamples=100)

    hidden_size   = base_model.encoder.config.hidden_size
    feature_names = (
        [f"emb_{i}" for i in range(hidden_size)]
        + ["avg_word_len", "type_token_ratio", "avg_sent_len", "punct_rate",
           "flesch", "stopword_ratio", "sent_len_std", "bullet_rate",
           "question_rate", "paren_rate", "upper_ratio", "bigram_diversity",
           "syllable_score", "fog_index", "hedge_ratio"]
    )
    shap.summary_plot(
        shap_values[:, :, 0], ex_feats[:5],
        feature_names=feature_names, max_display=15, show=True,
    )


# ── Persistence ────────────────────────────────────────────────────────────────
def save_artifacts(label_encoder, scaler, run_name: str) -> None:
    with open(f"label_encoder_{run_name}.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(f"scaler_{run_name}.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved label_encoder_{run_name}.pkl and scaler_{run_name}.pkl")
