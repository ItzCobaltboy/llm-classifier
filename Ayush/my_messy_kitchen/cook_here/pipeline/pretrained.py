from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import tiktoken
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    from sklearn.preprocessing import LabelEncoder
except ImportError as exc:
    raise SystemExit(
        "Missing dependency while importing pretrained pipeline requirements. "
        "Install: pandas pyarrow numpy matplotlib scikit-learn sentence-transformers tiktoken\n"
        f"Original error: {exc}"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train" / "train.parquet"
VAL_PATH = BASE_DIR / "val" / "val.parquet"
TEST_PATH = BASE_DIR / "test" / "test.parquet"
OUTPUT_DIR = BASE_DIR / "pretrained_outputs"

TEXT_COL = "text"
LABEL_COL = "generated_by"
ENCODED_LABEL_COL = "label"

SEED = 42
MAX_TIKTOKEN_LENGTH = 256
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class Metrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


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


def truncate_texts_with_tiktoken(texts: list[str], max_tokens: int) -> tuple[list[str], list[int]]:
    encoding = tiktoken.get_encoding("cl100k_base")
    truncated_texts: list[str] = []
    token_lengths: list[int] = []

    for text in texts:
        tokens = encoding.encode(text, disallowed_special=())
        token_lengths.append(min(len(tokens), max_tokens))
        truncated_tokens = tokens[:max_tokens]
        truncated_texts.append(encoding.decode(truncated_tokens))

    return truncated_texts, token_lengths


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


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


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray, output_path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Greens")
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


def save_metric_bars(val_metrics: Metrics, test_metrics: Metrics, output_path: Path) -> None:
    names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    val_values = [getattr(val_metrics, name) for name in names]
    test_values = [getattr(test_metrics, name) for name in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, val_values, width=width, label="Validation", color="tab:blue")
    ax.bar(x + width / 2, test_values, width=width, label="Test", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Pretrained pipeline metrics")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_token_length_plot(train_lengths: list[int], val_lengths: list[int], test_lengths: list[int], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(train_lengths, bins=30, alpha=0.5, label="Train")
    ax.hist(val_lengths, bins=30, alpha=0.5, label="Val")
    ax.hist(test_lengths, bins=30, alpha=0.5, label="Test")
    ax.set_title("Token lengths after tiktoken truncation")
    ax.set_xlabel("Token count")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    label_encoder: LabelEncoder,
    val_metrics: Metrics,
    test_metrics: Metrics,
    output_path: Path,
) -> None:
    summary = {
        "embedding_model": EMBEDDING_MODEL_NAME,
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

    train_texts, train_lengths = truncate_texts_with_tiktoken(X_train[TEXT_COL].tolist(), MAX_TIKTOKEN_LENGTH)
    val_texts, val_lengths = truncate_texts_with_tiktoken(X_val[TEXT_COL].tolist(), MAX_TIKTOKEN_LENGTH)
    test_texts, test_lengths = truncate_texts_with_tiktoken(X_test[TEXT_COL].tolist(), MAX_TIKTOKEN_LENGTH)

    save_token_length_plot(train_lengths, val_lengths, test_lengths, OUTPUT_DIR / "token_lengths.png")

    encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    X_train_embed = embed_texts(encoder, train_texts)
    X_val_embed = embed_texts(encoder, val_texts)
    X_test_embed = embed_texts(encoder, test_texts)

    np.save(OUTPUT_DIR / "train_embeddings.npy", X_train_embed)
    np.save(OUTPUT_DIR / "val_embeddings.npy", X_val_embed)
    np.save(OUTPUT_DIR / "test_embeddings.npy", X_test_embed)

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    classifier.fit(X_train_embed, y_train)
    with (OUTPUT_DIR / "classifier.pkl").open("wb") as f:
        pickle.dump(classifier, f)

    val_pred = classifier.predict(X_val_embed)
    test_pred = classifier.predict(X_test_embed)

    val_metrics = compute_metrics(y_val, val_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)
    print("\nTest classification report:\n")
    print(classification_report(y_test, test_pred, target_names=label_encoder.classes_, zero_division=0))

    save_metric_bars(val_metrics, test_metrics, OUTPUT_DIR / "metrics.png")
    save_confusion_matrix(
        y_val,
        val_pred,
        label_encoder.classes_,
        OUTPUT_DIR / "val_confusion_matrix.png",
        "Pretrained validation confusion matrix",
    )
    save_confusion_matrix(
        y_test,
        test_pred,
        label_encoder.classes_,
        OUTPUT_DIR / "test_confusion_matrix.png",
        "Pretrained test confusion matrix",
    )
    save_predictions(val_texts, y_val, val_pred, label_encoder, OUTPUT_DIR / "val_predictions.csv")
    save_predictions(test_texts, y_test, test_pred, label_encoder, OUTPUT_DIR / "test_predictions.csv")
    save_run_summary(label_encoder, val_metrics, test_metrics, OUTPUT_DIR / "run_summary.json")


if __name__ == "__main__":
    main()
