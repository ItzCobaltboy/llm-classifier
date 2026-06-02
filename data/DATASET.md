# Dataset

The dataset contains short texts labelled by which LLM (or human) generated them. It is not included in this repository. This document describes its structure, schema, and statistics in full so it can be reconstructed or substituted.

## Source

The dataset originates from a public text-attribution benchmark. Texts cover a variety of topics and prompt types; each sample is a standalone generation (not a conversation).

## Classes

12 classes, one per source:

| Label | Type |
|-------|------|
| `chatgpt` | OpenAI ChatGPT |
| `cohere` | Cohere (base) |
| `cohere-chat` | Cohere (chat) |
| `gpt2` | OpenAI GPT-2 |
| `gpt3` | OpenAI GPT-3 |
| `gpt4` | OpenAI GPT-4 |
| `human` | Human-written text |
| `llama-chat` | Meta LLaMA (chat) |
| `mistral` | Mistral (base) |
| `mistral-chat` | Mistral (chat) |
| `mpt` | MosaicML MPT (base) |
| `mpt-chat` | MosaicML MPT (chat) |

## Schema

Column names differ by dataset size. This is a known inconsistency from the original data source.

**Small and Medium datasets**

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | The generated or human-written text |
| `generated_by` | string | Class label (one of the 12 above) |

**Large dataset**

| Column | Type | Description |
|--------|------|-------------|
| `generation` | string | The generated or human-written text |
| `model` | string | Class label (one of the 12 above) |

`data/change_columns.py` converts the large-dataset schema to the small/medium schema.

## Sizes and splits

All three sizes use an approximate 80 / 10 / 10 train / val / test split. The raw (unbalanced) counts are:

| Dataset | Split | Total rows | Rows per class (approx.) |
|---------|-------|-----------|--------------------------|
| small | train | ~9,840 | ~820 |
| small | val | ~984 | ~82 |
| small | test | 984 | 82 (exact, from test reports) |
| medium | train | ~98,160 | ~8,180 |
| medium | val | ~9,816 | ~818 |
| medium | test | 9,816 | 818 (exact, from test reports) |
| large | train | ~793,000 | varies |
| large | val | ~99,000 | varies |
| large | test | ~99,000 | varies |

Total large dataset: approximately 991,000 rows across all splits.

The raw data is **not** class-balanced. Human texts tend to be significantly longer than LLM-generated texts.

## Text length characteristics

Based on EDA of the large dataset:

- Human-written texts are the longest on average, often exceeding 500 words before truncation.
- LLM-generated texts cluster around 100-300 words depending on the model.
- Chat-variant models (`cohere-chat`, `llama-chat`, `mistral-chat`, `mpt-chat`) tend to produce shorter responses than their base counterparts.
- Sentence length (avg words per sentence) varies noticeably across models and is a discriminative feature.

## Cleaning applied

`data/clean.py` applies two transformations to produce `Dataset_cleaned/`:

1. **Truncation** -- each text is truncated to 300 words (whitespace-tokenized). This normalises length and removes the human-text length advantage.
2. **Downsampling** -- each class is downsampled to the size of the smallest class, making the dataset perfectly balanced.

After cleaning, each class has equal representation in every split.

## File format

Files are stored as Apache Parquet. Each split is a single file:

```
Dataset/
  Dataset_small/
    train/train.parquet
    val/val.parquet
    test/test.parquet
  Dataset_medium/  (same structure)
  Dataset_large/   (same structure)
```

Read with:

```python
import pandas as pd
df = pd.read_parquet("Dataset/Dataset_small/train/train.parquet")
```

No index column is stored. All string values are UTF-8.
