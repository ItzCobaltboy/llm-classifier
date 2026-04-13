"""
EDA -- Large Dataset Analysis
==============================
Produces graphs + stats for the large cleaned dataset (~991k rows, 12 classes).
Uses pyarrow for I/O (no pandas), scipy for KDE, matplotlib for all plots.
CPU only -- no GPU required.

Output folder: ./output/eda/
  class_distribution.png
  text_length_words.png
  text_length_chars.png
  words_per_sentence.png
  length_boxplot.png
  mean_word_count.png
  tfidf_top_terms.png
  vocab_size.png
  punctuation_density.png
  summary.json
  top_tfidf_terms.json

Usage:
  python eda_large.py
  python eda_large.py --data_root /content/drive/MyDrive/Dataset_cleaned   # Colab
"""

import argparse, json, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")           # headless -- no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.stats import gaussian_kde
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", default="../Dataset_cleaned")
parser.add_argument("--out_dir",   default="./output/eda")
parser.add_argument("--sample",    type=int, default=5000,
                    help="Samples per class for heavy analyses (TF-IDF etc.)")
args = parser.parse_args()

DATA_ROOT = Path(args.data_root)
OUT_DIR   = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# tab20 palette -- 20 distinct colours, use first 12
TAB20 = plt.get_cmap("tab20").colors  # type: ignore
PALETTE = [TAB20[i] for i in range(12)]

print(f"Output -> {OUT_DIR.resolve()}")


# ── 1. Load data via pyarrow ──────────────────────────────────────────────────
# Large dataset columns are 'model' (label) and 'generation' (text)

print("\nLoading large dataset via pyarrow...")
t0 = time.time()

all_texts, all_labels = [], []
split_info = {}

for split in ["train", "val", "test"]:
    path  = DATA_ROOT / "Dataset_large" / split / f"{split}.parquet"
    table = pq.read_table(path, columns=["model", "generation"])
    texts  = table.column("generation").to_pylist()
    labels = table.column("model").to_pylist()
    all_texts  += texts
    all_labels += labels
    split_info[split] = {"n": len(texts)}
    print(f"  {split}: {len(texts):,} rows")

print(f"  Total: {len(all_texts):,} rows  ({time.time()-t0:.1f}s)")

classes     = sorted(set(all_labels))
NUM_CLASSES = len(classes)
label2idx   = {c: i for i, c in enumerate(classes)}

# Group indices by class for per-class analyses
class_indices = {c: [i for i, l in enumerate(all_labels) if l == c] for c in classes}
print(f"  Classes ({NUM_CLASSES}): {classes}")


# ── 2. Text length features ───────────────────────────────────────────────────

print("\nComputing text lengths...")

word_counts = np.array([len(t.split()) for t in all_texts])
char_counts = np.array([len(t)         for t in all_texts])

def avg_words_per_sentence(text):
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([len(s.split()) for s in sentences]))

sent_lens = np.array([avg_words_per_sentence(t) for t in all_texts])

# Per-class arrays
idx_arr    = np.array([label2idx[l] for l in all_labels])
class_wc   = {c: word_counts[idx_arr == label2idx[c]] for c in classes}
class_cc   = {c: char_counts[idx_arr == label2idx[c]] for c in classes}
class_sl   = {c: sent_lens  [idx_arr == label2idx[c]] for c in classes}


# ── 3. Summary stats (JSON) ───────────────────────────────────────────────────

summary = {
    "dataset":    "large",
    "splits":     split_info,
    "total":      len(all_texts),
    "classes":    NUM_CLASSES,
    "class_list": classes,
    "global": {
        "word_count": {
            "mean":   round(float(word_counts.mean()), 1),
            "median": round(float(np.median(word_counts)), 1),
            "std":    round(float(word_counts.std()), 1),
            "min":    int(word_counts.min()),
            "max":    int(word_counts.max()),
        },
        "char_count": {
            "mean":   round(float(char_counts.mean()), 1),
            "median": round(float(np.median(char_counts)), 1),
            "std":    round(float(char_counts.std()), 1),
        },
        "avg_words_per_sentence": {
            "mean": round(float(sent_lens.mean()), 2),
            "std":  round(float(sent_lens.std()), 2),
        },
    },
    "per_class": {}
}

for c in classes:
    wc = class_wc[c]
    summary["per_class"][c] = {
        "n":             len(wc),
        "word_mean":     round(float(wc.mean()), 1),
        "word_median":   round(float(np.median(wc)), 1),
        "word_std":      round(float(wc.std()), 1),
        "char_mean":     round(float(class_cc[c].mean()), 1),
        "sent_len_mean": round(float(class_sl[c].mean()), 2),
    }

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Saved summary.json")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def savefig(name):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {name}")


def kde_plot(ax, data, color, label, clip_pct=99):
    """Draw a smooth KDE line using scipy's gaussian_kde."""
    data = data[np.isfinite(data) & (data > 0)]
    if len(data) < 2:
        return
    sample = np.random.choice(data, size=min(3000, len(data)), replace=False)
    kde    = gaussian_kde(sample, bw_method=0.3)
    lo, hi = sample.min(), np.percentile(sample, clip_pct)
    xs     = np.linspace(lo, hi, 300)
    ax.plot(xs, kde(xs), color=color, linewidth=1.5, label=label)


# ── 4. Class distribution ─────────────────────────────────────────────────────

print("\nPlotting class distribution...")
counts = [len(class_indices[c]) for c in classes]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(classes, counts, color=PALETTE, edgecolor="white", linewidth=0.5)
ax.set_title("Class Distribution -- Large Dataset (all splits)", fontsize=14, pad=12)
ax.set_xlabel("Class")
ax.set_ylabel("Number of samples")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{cnt:,}", ha="center", va="bottom", fontsize=8)
plt.xticks(rotation=30, ha="right")
savefig("class_distribution.png")


# ── 5. Word count distribution per class (KDE) ────────────────────────────────

print("Plotting word count distributions...")
fig, ax = plt.subplots(figsize=(13, 5))
for i, c in enumerate(classes):
    kde_plot(ax, class_wc[c].astype(float), PALETTE[i], c)
ax.set_title("Word Count Distribution per Class", fontsize=14, pad=12)
ax.set_xlabel("Word count per text")
ax.set_ylabel("Density")
ax.set_xlim(0, np.percentile(word_counts, 99))
ax.legend(fontsize=8, ncol=2)
savefig("text_length_words.png")


# ── 6. Char count distribution per class (KDE) ────────────────────────────────

print("Plotting char count distributions...")
fig, ax = plt.subplots(figsize=(13, 5))
for i, c in enumerate(classes):
    kde_plot(ax, class_cc[c].astype(float), PALETTE[i], c)
ax.set_title("Character Count Distribution per Class", fontsize=14, pad=12)
ax.set_xlabel("Character count per text")
ax.set_ylabel("Density")
ax.set_xlim(0, np.percentile(char_counts, 99))
ax.legend(fontsize=8, ncol=2)
savefig("text_length_chars.png")


# ── 7. Words per sentence distribution ────────────────────────────────────────

print("Plotting avg words-per-sentence...")
fig, ax = plt.subplots(figsize=(13, 5))
for i, c in enumerate(classes):
    kde_plot(ax, class_sl[c], PALETTE[i], c)
ax.set_title("Avg Words per Sentence -- per Class", fontsize=14, pad=12)
ax.set_xlabel("Avg words per sentence")
ax.set_ylabel("Density")
ax.set_xlim(0, np.percentile(sent_lens[sent_lens > 0], 99))
ax.legend(fontsize=8, ncol=2)
savefig("words_per_sentence.png")


# ── 8. Boxplot: word count per class ─────────────────────────────────────────

print("Plotting word count boxplot...")
fig, ax = plt.subplots(figsize=(13, 5))
data_for_box = [class_wc[c] for c in classes]
bp = ax.boxplot(data_for_box, patch_artist=True, notch=False,
                flierprops=dict(marker=".", markersize=1, alpha=0.2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(1, NUM_CLASSES + 1))
ax.set_xticklabels(classes, rotation=30, ha="right")
ax.set_title("Word Count per Class -- Boxplot", fontsize=14, pad=12)
ax.set_ylabel("Word count")
ax.set_ylim(0, np.percentile(word_counts, 99.5))
savefig("length_boxplot.png")


# ── 9. Mean word count per class ─────────────────────────────────────────────

print("Plotting mean word count per class...")
means = [float(class_wc[c].mean()) for c in classes]
sems  = [float(class_wc[c].std() / np.sqrt(len(class_wc[c]))) for c in classes]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(classes, means, yerr=sems, color=PALETTE,
       capsize=4, edgecolor="white", linewidth=0.5)
ax.set_title("Mean Word Count per Class (+/- SEM)", fontsize=14, pad=12)
ax.set_ylabel("Mean word count")
plt.xticks(rotation=30, ha="right")
savefig("mean_word_count.png")


# ── 10. TF-IDF top terms per class ────────────────────────────────────────────

print(f"\nComputing TF-IDF top terms (sample={args.sample} per class)...")
t_tfidf = time.time()

sampled_texts, sampled_labels = [], []
for c in classes:
    idx    = class_indices[c]
    chosen = np.random.choice(idx, size=min(args.sample, len(idx)), replace=False)
    sampled_texts  += [all_texts[i]  for i in chosen]
    sampled_labels += [all_labels[i] for i in chosen]

tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=5,
    stop_words="english",
    sublinear_tf=True,
)
X = tfidf.fit_transform(sampled_texts)
feature_names = np.array(tfidf.get_feature_names_out())
le_tfidf = LabelEncoder()
y = le_tfidf.fit_transform(sampled_labels)

print(f"  TF-IDF shape: {X.shape}  ({time.time()-t_tfidf:.1f}s)")

N_TERMS   = 15
top_terms = {}
for i, c in enumerate(classes):
    mask       = y == i
    mean_tfidf = np.asarray(X[mask].mean(axis=0)).flatten()
    top_idx    = mean_tfidf.argsort()[::-1][:N_TERMS]
    top_terms[c] = [(feature_names[j], round(float(mean_tfidf[j]), 4))
                    for j in top_idx]

with open(OUT_DIR / "top_tfidf_terms.json", "w") as f:
    json.dump(top_terms, f, indent=2)
print("  Saved top_tfidf_terms.json")

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
axes = axes.flatten()
for i, (c, terms) in enumerate(top_terms.items()):
    ax     = axes[i]
    words  = [t[0] for t in terms][::-1]
    scores = [t[1] for t in terms][::-1]
    ax.barh(words, scores, color=PALETTE[i], edgecolor="white", linewidth=0.4)
    ax.set_title(c, fontsize=11, fontweight="bold")
    ax.set_xlabel("Mean TF-IDF", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.tick_params(axis="x", labelsize=7)
fig.suptitle(
    f"Top {N_TERMS} TF-IDF Terms per Class  (sample={args.sample}/class, bigrams included)",
    fontsize=14, y=1.01,
)
savefig("tfidf_top_terms.png")


# ── 11. Unique vocabulary size per class ─────────────────────────────────────

print("Computing vocabulary sizes...")

sampled_by_class: dict = {c: [] for c in classes}
for text, label in zip(sampled_texts, sampled_labels):
    sampled_by_class[label].append(text)

def vocab_size(texts):
    words: set = set()
    for t in texts:
        words.update(re.findall(r'\b[a-zA-Z]+\b', t.lower()))
    return len(words)

vocab_sizes = {c: vocab_size(sampled_by_class[c]) for c in classes}

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(list(vocab_sizes.keys()), list(vocab_sizes.values()),
       color=PALETTE, edgecolor="white")
ax.set_title(f"Unique Vocabulary Size per Class  (sample={args.sample}/class)",
             fontsize=14, pad=12)
ax.set_ylabel("Unique word types")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.xticks(rotation=30, ha="right")
savefig("vocab_size.png")


# ── 12. Punctuation density per class ────────────────────────────────────────

print("Computing punctuation density...")

PUNCT = set('.,!?;:"\'-\u2013\u2014()[]{}')

def punct_density(text):
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in PUNCT) / len(text)

fig, ax = plt.subplots(figsize=(13, 5))
for i, c in enumerate(classes):
    densities = np.array([punct_density(all_texts[j])
                          for j in class_indices[c][:5000]], dtype=float)
    kde_plot(ax, densities, PALETTE[i], c, clip_pct=99)
ax.set_title("Punctuation Density per Class  (punct chars / total chars)",
             fontsize=14, pad=12)
ax.set_xlabel("Punctuation density")
ax.set_ylabel("Density")
ax.legend(fontsize=8, ncol=2)
savefig("punctuation_density.png")


# ── 13. Console summary ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  DATASET SUMMARY -- LARGE")
print("=" * 60)
print(f"  Total samples : {len(all_texts):,}")
print(f"  Classes       : {NUM_CLASSES}  (perfectly balanced)")
for split, info in split_info.items():
    print(f"  {split:6s}        : {info['n']:,} ({info['n']//NUM_CLASSES:,} per class)")
print()
print(f"  {'Class':<16} {'N':>8}  {'WrdMean':>8}  {'WrdMed':>7}  {'CharMean':>9}  {'SentLen':>8}  {'Vocab':>7}")
print("  " + "-" * 68)
for c in classes:
    p  = summary["per_class"][c]
    vs = vocab_sizes.get(c, 0)
    print(f"  {c:<16} {p['n']:>8,}  {p['word_mean']:>8.1f}  {p['word_median']:>7.1f}"
          f"  {p['char_mean']:>9.1f}  {p['sent_len_mean']:>8.2f}  {vs:>7,}")

print()
print(f"  Global word count  -- mean: {summary['global']['word_count']['mean']}  "
      f"median: {summary['global']['word_count']['median']}  "
      f"std: {summary['global']['word_count']['std']}")
print(f"  Global char count  -- mean: {summary['global']['char_count']['mean']}")
print(f"  Words per sentence -- mean: {summary['global']['avg_words_per_sentence']['mean']}")
print()
print(f"  All plots saved to: {OUT_DIR.resolve()}/")
print("=" * 60)
