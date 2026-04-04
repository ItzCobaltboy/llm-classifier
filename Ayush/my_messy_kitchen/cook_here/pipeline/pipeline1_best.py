# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline 1 — Best tokenizer + best embedding + best classifier
#  Sentence-BERT (all-mpnet-base-v2) + LinearSVC
#  Tokenization, pooling handled internally by sentence-transformers
# ══════════════════════════════════════════════════════════════════════════════
#  pip install sentence-transformers scikit-learn pandas matplotlib seaborn

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_parquet("train.parquet")
val_df   = pd.read_parquet("val.parquet")
test_df  = pd.read_parquet("test.parquet")

print(f"  Train : {len(train_df):,} rows")
print(f"  Val   : {len(val_df):,} rows")
print(f"  Test  : {len(test_df):,} rows")
print(f"  Labels: {sorted(train_df['generated_by'].unique())}")


# ── 2. Basic cleaning ──────────────────────────────────────────────────────────
def clean(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # remove non-ASCII
    return text

for df in [train_df, val_df, test_df]:
    df["text"] = df["text"].apply(clean)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.dropna(subset=["text", "generated_by"], inplace=True)


# ── 3. Encode labels ───────────────────────────────────────────────────────────
le = LabelEncoder()
y_train = le.fit_transform(train_df["generated_by"])
y_val   = le.transform(val_df["generated_by"])
y_test  = le.transform(test_df["generated_by"])


# ── 4. Embed ───────────────────────────────────────────────────────────────────
# all-mpnet-base-v2 is the strongest general-purpose sentence embedding model.
# It handles tokenization + transformer encoding + mean pooling internally.
# Output: (num_samples, 768) — one 1D vector per sentence, ready for classifiers.

print("\nLoading sentence-transformer model...")
embedder = SentenceTransformer("all-mpnet-base-v2")

print("Generating embeddings (this may take a few minutes)...")
X_train = embedder.encode(
    train_df["text"].tolist(), batch_size=64,
    show_progress_bar=True, normalize_embeddings=True,
)
X_val = embedder.encode(
    val_df["text"].tolist(), batch_size=64,
    show_progress_bar=True, normalize_embeddings=True,
)
X_test = embedder.encode(
    test_df["text"].tolist(), batch_size=64,
    show_progress_bar=True, normalize_embeddings=True,
)
print(f"  Embedding shape: {X_train.shape}")


# ── 5. Train classifier ────────────────────────────────────────────────────────
# LinearSVC is the fastest and most accurate classical classifier for
# high-dimensional dense embeddings from transformers.
print("\nTraining LinearSVC...")
clf = LinearSVC(C=1.0, max_iter=5000)
clf.fit(X_train, y_train)


# ── 6. Evaluate ────────────────────────────────────────────────────────────────
def evaluate(clf, X, y_true, split_name):
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'─' * 55}")
    print(f"  {split_name}")
    print(f"{'─' * 55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=le.classes_)}")
    return y_pred

y_val_pred  = evaluate(clf, X_val,  y_val,  "Validation Set")
y_test_pred = evaluate(clf, X_test, y_test, "Test Set")


# ── 7. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, labels, title, cmap="Blues"):
    cm  = confusion_matrix(y_true, y_pred)
    n   = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    fname = title.replace(" ", "_").replace("—", "").lower() + ".png"
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"  Saved: {fname}")

plot_confusion_matrix(
    y_test, y_test_pred, le.classes_,
    "Pipeline 1 — Test Confusion Matrix",
)
