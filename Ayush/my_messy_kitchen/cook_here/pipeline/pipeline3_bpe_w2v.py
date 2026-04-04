# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline 3 — BPE / WordPiece  ×  Word2Vec / FastText  ×  LinearSVC
#  Trains all 4 combinations, evaluates each, reports the best.
# ══════════════════════════════════════════════════════════════════════════════
#  pip install tokenizers gensim scikit-learn pandas matplotlib seaborn

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from gensim.models import Word2Vec, FastText
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
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

for df in [train_df, val_df, test_df]:
    df["text"] = df["text"].apply(clean)
    df.drop_duplicates(subset=["text"], inplace=True)
    df.dropna(subset=["text", "generated_by"], inplace=True)

train_texts = train_df["text"].tolist()
val_texts   = val_df["text"].tolist()
test_texts  = test_df["text"].tolist()


# ── 3. Encode labels ───────────────────────────────────────────────────────────
le = LabelEncoder()
y_train = le.fit_transform(train_df["generated_by"])
y_val   = le.transform(val_df["generated_by"])
y_test  = le.transform(test_df["generated_by"])


# ── 4. Tokenizer training ──────────────────────────────────────────────────────
# Both tokenizers are trained on the TRAIN set only.
# val/test texts are tokenized using the already-fitted vocab.

VOCAB_SIZE = 16_000

def build_tokenizer(kind: str, texts: list[str]) -> Tokenizer:
    """Train a BPE or WordPiece tokenizer on the given corpus."""
    if kind == "bpe":
        tok     = Tokenizer(BPE(unk_token="[UNK]"))
        trainer = BpeTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]"],
        )
    else:  # wordpiece
        tok     = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=["[UNK]", "[PAD]"],
        )
    tok.pre_tokenizer = Whitespace()
    tok.train_from_iterator(texts, trainer)
    return tok

def tokenize(tokenizer: Tokenizer, texts: list[str]) -> list[list[str]]:
    """Returns a list of token-string lists (not IDs)."""
    return [tokenizer.encode(t).tokens for t in texts]

print("\nTraining tokenizers (on train set only)...")
tokenizers_map = {}
for name in ["bpe", "wordpiece"]:
    print(f"  {name.upper()} ...")
    tokenizers_map[name] = build_tokenizer(name, train_texts)


# ── 5. Embedding helpers ───────────────────────────────────────────────────────
EMBED_DIM = 100
WINDOW    = 5
MIN_COUNT = 1
WORKERS   = 4
EPOCHS    = 10

def train_embedding(kind: str, tokenized_corpus: list[list[str]]):
    """Train Word2Vec or FastText on a tokenized corpus."""
    kwargs = dict(
        sentences=tokenized_corpus,
        vector_size=EMBED_DIM,
        window=WINDOW,
        min_count=MIN_COUNT,
        workers=WORKERS,
        epochs=EPOCHS,
    )
    if kind == "word2vec":
        return Word2Vec(**kwargs)
    else:  # fasttext
        return FastText(**kwargs)

def mean_pool(token_strings: list[str], emb_model) -> np.ndarray:
    """Average all token vectors in a sentence into one 1D vector."""
    vecs = [emb_model.wv[t] for t in token_strings if t in emb_model.wv]
    if not vecs:
        return np.zeros(emb_model.vector_size)
    return np.mean(vecs, axis=0)

def embed_corpus(tokenized: list[list[str]], emb_model) -> np.ndarray:
    """Produce (num_samples, embed_dim) matrix from tokenized texts."""
    return np.array([mean_pool(tokens, emb_model) for tokens in tokenized])


# ── 6. Evaluation helper ───────────────────────────────────────────────────────
def evaluate(clf, X, y_true, split_name):
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"    {split_name} — Acc: {acc:.4f}  Macro F1: {f1:.4f}")
    return y_pred, acc, f1


# ── 7. Run all 4 combinations ─────────────────────────────────────────────────
# Combinations: BPE×W2V, BPE×FT, WP×W2V, WP×FT
embedding_kinds = {"Word2Vec": "word2vec", "FastText": "fasttext"}
results = []

for tok_label, tok_kind in [("BPE", "bpe"), ("WordPiece", "wordpiece")]:
    print(f"\n{'═'*55}")
    print(f"  Tokenizer: {tok_label}")
    print(f"{'═'*55}")

    tokenizer = tokenizers_map[tok_kind]
    train_tok = tokenize(tokenizer, train_texts)
    val_tok   = tokenize(tokenizer, val_texts)
    test_tok  = tokenize(tokenizer, test_texts)

    for emb_label, emb_kind in embedding_kinds.items():
        combo = f"{tok_label} + {emb_label}"
        print(f"\n  Embedding: {emb_label} ...")

        emb_model = train_embedding(emb_kind, train_tok)

        X_train = embed_corpus(train_tok, emb_model)
        X_val   = embed_corpus(val_tok,   emb_model)
        X_test  = embed_corpus(test_tok,  emb_model)

        print(f"    Embedding shape: {X_train.shape}")

        clf = LinearSVC(C=1.0, max_iter=5000)
        clf.fit(X_train, y_train)

        val_pred,  val_acc,  val_f1  = evaluate(clf, X_val,  y_val,  "Val ")
        test_pred, test_acc, test_f1 = evaluate(clf, X_test, y_test, "Test")

        results.append({
            "Combination":  combo,
            "Val Acc":      round(val_acc,  4),
            "Val F1":       round(val_f1,   4),
            "Test Acc":     round(test_acc, 4),
            "Test F1":      round(test_f1,  4),
            "_test_pred":   test_pred,
        })


# ── 7. Summary table ───────────────────────────────────────────────────────────
display_cols = ["Combination", "Val Acc", "Val F1", "Test Acc", "Test F1"]
results_df   = pd.DataFrame(results)[display_cols]

print("\n" + "═" * 65)
print("  COMBINATION COMPARISON")
print("═" * 65)
print(results_df.sort_values("Test F1", ascending=False).to_string(index=False))


# ── 8. Grouped bar chart ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
sub = results_df.sort_values("Test F1", ascending=True)
colors = ["#7F77DD", "#5DCAA5", "#D85A30", "#378ADD"]
bars = ax.barh(sub["Combination"], sub["Test F1"], color=colors[:len(sub)], height=0.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Test Macro F1")
ax.set_title("Pipeline 3 — Tokenizer × Embedding Combination Comparison")
for i, v in enumerate(sub["Test F1"]):
    ax.text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("pipeline3_comparison.png", dpi=150)
plt.show()
print("  Saved: pipeline3_comparison.png")


# ── 9. Best combination — full report + confusion matrix ──────────────────────
best   = max(results, key=lambda r: r["Test F1"])
y_pred = best["_test_pred"]

print(f"\nBest combination : {best['Combination']}")
print(f"Test Accuracy    : {best['Test Acc']:.4f}")
print(f"Test Macro F1    : {best['Test F1']:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

n   = len(le.classes_)
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples",
    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
)
ax.set_title(f"Pipeline 3 Best ({best['Combination']}) — Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()
plt.savefig("pipeline3_best_cm.png", dpi=150)
plt.show()
print("  Saved: pipeline3_best_cm.png")
