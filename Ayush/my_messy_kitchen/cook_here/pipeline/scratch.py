from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.preprocessing import LabelEncoder
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import BpeTrainer
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency while importing scratch pipeline requirements. "
        "Install: pandas pyarrow numpy matplotlib torch scikit-learn tokenizers\n"
        f"Original error: {exc}"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train" / "train.parquet"
VAL_PATH = BASE_DIR / "val" / "val.parquet"
TEST_PATH = BASE_DIR / "test" / "test.parquet"
OUTPUT_DIR = BASE_DIR / "scratch_outputs"

TEXT_COL = "text"
LABEL_COL = "generated_by"
ENCODED_LABEL_COL = "label"

SEED = 42
VOCAB_SIZE = 16_000
MAX_LENGTH = 256
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_DIM_1 = 256
HIDDEN_DIM_2 = 128
DROPOUT = 0.3
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 12
PATIENCE = 3

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


class TextDataset(Dataset):
    def __init__(self, token_ids: np.ndarray, attention_masks: np.ndarray, labels: np.ndarray) -> None:
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.attention_masks = torch.as_tensor(attention_masks, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.token_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class DoubleHiddenLayerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_classes: int,
        pad_id: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        masked_embeddings = embedded * mask
        pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_inputs_exist() -> None:
    missing = [str(path) for path in (TRAIN_PATH, VAL_PATH, TEST_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing parquet files: {missing}")


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required_cols = {TEXT_COL, LABEL_COL}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise KeyError(f"{path.name} is missing required columns: {sorted(missing_cols)}")

    clean_df = df[[TEXT_COL, LABEL_COL]].copy()
    clean_df[TEXT_COL] = (
        clean_df[TEXT_COL]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    clean_df[LABEL_COL] = clean_df[LABEL_COL].fillna("unknown").astype(str).str.strip()
    clean_df = clean_df[clean_df[TEXT_COL] != ""].reset_index(drop=True)
    return clean_df


def add_encoded_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder]:
    label_encoder = LabelEncoder()
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    label_encoder.fit(train_df[LABEL_COL])
    known_labels = set(label_encoder.classes_)

    for name, split_df in {"val": val_df, "test": test_df}.items():
        unknown = sorted(set(split_df[LABEL_COL]) - known_labels)
        if unknown:
            raise ValueError(f"{name} split contains labels unseen in train split: {unknown}")

    for split_df in (train_df, val_df, test_df):
        split_df[ENCODED_LABEL_COL] = label_encoder.transform(split_df[LABEL_COL])
        split_df.drop(columns=[LABEL_COL], inplace=True)

    return train_df, val_df, test_df, label_encoder


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[ENCODED_LABEL_COL]).copy()
    y = df[ENCODED_LABEL_COL].to_numpy(dtype=np.int64)
    return X, y


def train_tokenizer(texts: Iterable[str]) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: Iterable[str], max_length: int) -> tuple[np.ndarray, np.ndarray]:
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing one or more special token ids.")

    token_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []

    for text in texts:
        ids = tokenizer.encode(text).ids[: max_length - 2]
        ids = [bos_id, *ids, eos_id]
        attention = [1] * len(ids)

        pad_count = max_length - len(ids)
        if pad_count > 0:
            ids.extend([pad_id] * pad_count)
            attention.extend([0] * pad_count)

        token_rows.append(ids)
        mask_rows.append(attention)

    return np.asarray(token_rows, dtype=np.int64), np.asarray(mask_rows, dtype=np.float32)


def build_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision,
        recall_macro=recall,
        f1_macro=f1,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, Metrics, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)

            predictions = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(y_true, y_pred)
    return avg_loss, metrics, y_true, y_pred


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, list[float], list[float], list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    patience_counter = 0

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_f1_scores: list[float] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss, epoch_val_metrics, _, _ = evaluate(model, val_loader, device, criterion)

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_f1_scores.append(epoch_val_metrics.f1_macro)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={epoch_train_loss:.4f} | "
            f"val_loss={epoch_val_loss:.4f} | "
            f"val_f1_macro={epoch_val_metrics.f1_macro:.4f}"
        )

        if epoch_val_metrics.f1_macro > best_val_f1:
            best_val_f1 = epoch_val_metrics.f1_macro
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, val_f1_scores


def save_metrics_plot(train_losses: list[float], val_losses: list[float], val_f1_scores: list[float], output_dir: Path) -> None:
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_losses, marker="o", label="Train loss")
    axes[0].plot(epochs, val_losses, marker="s", label="Val loss")
    axes[0].set_title("Scratch model loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_f1_scores, marker="o", color="tab:green", label="Val macro F1")
    axes[1].set_title("Scratch validation F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro F1")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_metric_bars(val_metrics: Metrics, test_metrics: Metrics, output_path: Path) -> None:
    metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    val_values = [getattr(val_metrics, name) for name in metric_names]
    test_values = [getattr(test_metrics, name) for name in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, val_values, width=width, label="Validation", color="tab:blue")
    ax.bar(x + width / 2, test_values, width=width, label="Test", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Scratch pipeline metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_predictions(texts: list[str], y_true: np.ndarray, y_pred: np.ndarray, label_encoder: LabelEncoder, output_path: Path) -> None:
    prediction_df = pd.DataFrame(
        {
            TEXT_COL: texts,
            "true_label_id": y_true,
            "pred_label_id": y_pred,
            "true_label": label_encoder.inverse_transform(y_true),
            "pred_label": label_encoder.inverse_transform(y_pred),
        }
    )
    prediction_df.to_csv(output_path, index=False)


def save_run_summary(
    tokenizer: Tokenizer,
    label_encoder: LabelEncoder,
    val_metrics: Metrics,
    test_metrics: Metrics,
    output_path: Path,
) -> None:
    summary = {
        "vocab_size": tokenizer.get_vocab_size(),
        "classes": label_encoder.classes_.tolist(),
        "validation": val_metrics.__dict__,
        "test": test_metrics.__dict__,
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    set_seed(SEED)
    ensure_inputs_exist()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    train_df, val_df, test_df, label_encoder = add_encoded_labels(train_df, val_df, test_df)

    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)
    X_test, y_test = split_xy(test_df)

    tokenizer = train_tokenizer(X_train[TEXT_COL].tolist())
    tokenizer.save(str(OUTPUT_DIR / "subword_tokenizer.json"))

    train_token_ids, train_masks = encode_texts(tokenizer, X_train[TEXT_COL].tolist(), MAX_LENGTH)
    val_token_ids, val_masks = encode_texts(tokenizer, X_val[TEXT_COL].tolist(), MAX_LENGTH)
    test_token_ids, test_masks = encode_texts(tokenizer, X_test[TEXT_COL].tolist(), MAX_LENGTH)

    train_dataset = TextDataset(train_token_ids, train_masks, y_train)
    val_dataset = TextDataset(val_token_ids, val_masks, y_val)
    test_dataset = TextDataset(test_token_ids, test_masks, y_test)

    train_loader = build_loader(train_dataset, BATCH_SIZE, shuffle=True)
    val_loader = build_loader(val_dataset, BATCH_SIZE, shuffle=False)
    test_loader = build_loader(test_dataset, BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    if pad_id is None:
        raise ValueError("Could not resolve PAD token id.")

    model = DoubleHiddenLayerClassifier(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=EMBED_DIM,
        hidden_dim_1=HIDDEN_DIM_1,
        hidden_dim_2=HIDDEN_DIM_2,
        num_classes=len(label_encoder.classes_),
        pad_id=pad_id,
        dropout=DROPOUT,
    ).to(device)

    model, train_losses, val_losses, val_f1_scores = train_model(model, train_loader, val_loader, device)
    torch.save(model.state_dict(), OUTPUT_DIR / "scratch_model.pt")

    criterion = nn.CrossEntropyLoss()
    val_loss, val_metrics, val_true, val_pred = evaluate(model, val_loader, device, criterion)
    test_loss, test_metrics, test_true, test_pred = evaluate(model, test_loader, device, criterion)

    print("\nValidation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("\nTest classification report:\n")
    print(classification_report(test_true, test_pred, target_names=label_encoder.classes_, zero_division=0))

    save_metrics_plot(train_losses, val_losses, val_f1_scores, OUTPUT_DIR)
    save_metric_bars(val_metrics, test_metrics, OUTPUT_DIR / "metrics.png")
    save_confusion_matrix(
        val_true,
        val_pred,
        label_encoder.classes_,
        OUTPUT_DIR / "val_confusion_matrix.png",
        "Scratch validation confusion matrix",
    )
    save_confusion_matrix(
        test_true,
        test_pred,
        label_encoder.classes_,
        OUTPUT_DIR / "test_confusion_matrix.png",
        "Scratch test confusion matrix",
    )
    save_predictions(X_val[TEXT_COL].tolist(), val_true, val_pred, label_encoder, OUTPUT_DIR / "val_predictions.csv")
    save_predictions(X_test[TEXT_COL].tolist(), test_true, test_pred, label_encoder, OUTPUT_DIR / "test_predictions.csv")
    save_run_summary(tokenizer, label_encoder, val_metrics, test_metrics, OUTPUT_DIR / "run_summary.json")


if __name__ == "__main__":
    main()
