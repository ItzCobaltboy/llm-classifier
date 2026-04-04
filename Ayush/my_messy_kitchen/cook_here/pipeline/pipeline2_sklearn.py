# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline 2 — All best sklearn classifiers trained and compared
#  Features: TF-IDF (sklearn-native, no external embedding needed)
#  Models: LogisticRegression, LinearSVC, ComplementNB, KNN,
#          RandomForest, GradientBoosting
# ══════════════════════════════════════════════════════════════════════════════
#  pip install scikit-learn pandas matplotlib seaborn

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


# ── 3. TF-IDF features ────────────────────────────────────────────────────────
# sublinear_tf dampens the effect of very frequent terms.
# ngram_range (1,2) captures both unigrams and bigrams.
print("\nBuilding TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)
X_train_sp = tfidf.fit_transform(train_df["text"])   # sparse matrix
X_val_sp   = tfidf.transform(val_df["text"])
X_test_sp  = tfidf.transform(test_df["text"])

# Dense arrays needed for KNN, RF, GBM
X_train_d  = X_train_sp.toarray()
X_val_d    = X_val_sp.toarray()
X_test_d   = X_test_sp.toarray()

print(f"  Feature matrix : {X_train_sp.shape}")


# ── 4. Encode labels ───────────────────────────────────────────────────────────
le = LabelEncoder()
y_train = le.fit_transform(train_df["generated_by"])
y_val   = le.transform(val_df["generated_by"])
y_test  = le.transform(test_df["generated_by"])


# ── 5. Evaluation helper ───────────────────────────────────────────────────────
def evaluate(clf, X, y_true, split_name, dense=False):
    y_pred = clf.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'─' * 55}")
    print(f"  {split_name}")
    print(f"{'─' * 55}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=le.classes_)}")
    return y_pred, acc, f1


# ── 6. Define classifiers ──────────────────────────────────────────────────────
# Each entry: (name, classifier, use_dense_input)
classifiers = [
    ("Logistic Regression",    LogisticRegression(C=1.0, max_iter=1000, n_jobs=-1),          False),
    ("LinearSVC",              LinearSVC(C=1.0, max_iter=5000),                               False),
    ("Complement Naive Bayes", ComplementNB(),                                                 False),
    ("KNN (k=5)",              KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1), True),
    ("Random Forest",          RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42), True),
    ("Gradient Boosting",      GradientBoostingClassifier(n_estimators=100, random_state=42), True),
]


# ── 6. Train and evaluate all ──────────────────────────────────────────────────
results = []

for name, clf, use_dense in classifiers:
    print(f"\nTraining: {name} ...")
    Xtr = X_train_d if use_dense else X_train_sp
    Xva = X_val_d   if use_dense else X_val_sp
    Xte = X_test_d  if use_dense else X_test_sp

    clf.fit(Xtr, y_train)

    val_pred,  val_acc,  val_f1  = evaluate(clf, Xva, y_val,  f"{name} — Validation")
    test_pred, test_acc, test_f1 = evaluate(clf, Xte, y_test, f"{name} — Test")

    results.append({
        "Model":     name,
        "Val Acc":   round(val_acc,  4),
        "Val F1":    round(val_f1,   4),
        "Test Acc":  round(test_acc, 4),
        "Test F1":   round(test_f1,  4),
        "_clf":      clf,
        "_dense":    use_dense,
        "_test_pred": test_pred,
    })


# ── 7. Summary table ───────────────────────────────────────────────────────────
display_cols = ["Model", "Val Acc", "Val F1", "Test Acc", "Test F1"]
results_df   = pd.DataFrame(results)[display_cols]

print("\n" + "═" * 65)
print("  MODEL COMPARISON")
print("═" * 65)
print(results_df.sort_values("Test F1", ascending=False).to_string(index=False))


# ── 8. Bar chart — all models ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, metric, title in zip(
    axes,
    ["Val F1", "Test F1"],
    ["Validation Macro F1", "Test Macro F1"],
):
    sub = results_df.sort_values(metric, ascending=True)
    ax.barh(sub["Model"], sub[metric], color="#378ADD", height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1 Score")
    ax.set_title(title)
    for i, v in enumerate(sub[metric]):
        ax.text(v + 0.008, i, f"{v:.4f}", va="center", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Pipeline 2 — sklearn Model Comparison", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("pipeline2_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved: pipeline2_comparison.png")


# ── 9. Full report + confusion matrix for best model ──────────────────────────
best   = max(results, key=lambda r: r["Test F1"])
y_pred = best["_test_pred"]

print(f"\nBest model: {best['Model']}  (Test F1 = {best['Test F1']})")
print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")

n   = len(le.classes_)
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax,
)
ax.set_title(f"Pipeline 2 Best ({best['Model']}) — Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.tight_layout()
plt.savefig("pipeline2_best_cm.png", dpi=150)
plt.show()
print("  Saved: pipeline2_best_cm.png")
