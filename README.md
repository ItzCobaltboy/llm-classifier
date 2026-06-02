# LLM Classifier

Multi-class text classifier that identifies which LLM generated a given text. 12-way classification over: `chatgpt`, `cohere`, `cohere-chat`, `gpt2`, `gpt3`, `gpt4`, `human`, `llama-chat`, `mistral`, `mistral-chat`, `mpt`, `mpt-chat`.

## Repository layout

```
llm-classifier/
├── data/              Cleaning and EDA scripts
├── bert-dnn/          Frozen sentence embeddings + DNN head
├── bert-finetune/     BERT with last 2 layers unfrozen + DNN head
├── bert-rnn/          Frozen BERT + bidirectional GRU head
├── distilbert/        HuggingFace Trainer end-to-end fine-tuning
├── tfidf-nn/          TF-IDF + dense network
├── Dataset/           Raw parquet data (small / medium / large)
└── Dataset_cleaned/   Cleaned and class-balanced parquet data
```

## Dataset

The dataset is not included in this repository -- the files are too large to host on GitHub. See [data/DATASET.md](data/DATASET.md) for full schema and statistics. To reproduce results, obtain the source data and place it under `Dataset/` following the structure below.

| Folder           | Approx. rows | Approx. size |
|------------------|-------------|-------------|
| `Dataset_small`  | ~12 k       | ~50 MB      |
| `Dataset_medium` | ~120 k      | ~500 MB     |
| `Dataset_large`  | ~991 k      | ~4 GB       |

Each folder contains `train/`, `val/`, `test/` with a single `.parquet` file each.

Column names differ by dataset size:
- Small / Medium: `text`, `generated_by`
- Large: `generation`, `model`

Run `data/clean.py` to produce `Dataset_cleaned/` (truncates to 300 words, downsamples to balance classes). Run `data/change_columns.py` to convert the large-dataset column names to the small/medium format.

## Approaches

### bert-dnn

Generates pooled sentence embeddings with four frozen models (Qwen3-0.6B, E5-base, BERT mean-pool, MiniLM-L6), then trains a 3-layer DNN on top. Embeddings are cached as `.pt` tensors. See [bert-dnn/README.md](bert-dnn/README.md).

### bert-finetune

BERT with the last 2 encoder layers unfrozen, mixed-precision training, dual learning rates, and a 5-layer DNN head. See [bert-finetune/README.md](bert-finetune/README.md).

### bert-rnn

Fully frozen BERT encoder feeding into a bidirectional GRU (2 layers, 256 hidden). See [bert-rnn/README.md](bert-rnn/README.md).

### distilbert

Any HuggingFace sequence classification model fine-tuned end-to-end via the `Trainer` API. Defaults to `distilbert-base-uncased`. See [distilbert/README.md](distilbert/README.md).

### tfidf-nn

TF-IDF (1-4 gram, 50k features) with a 2-layer dense network. Includes grid search over 7 layer-size configurations and data-cleaning utilities. See [tfidf-nn/README.md](tfidf-nn/README.md).

## Results

| Model | Dataset | Test accuracy |
|-------|---------|--------------|
| BERT-DNN (frozen) | small | 46.2% |
| E5-DNN | small | 33.2% |
| Qwen3-DNN | small | 28.0% |
| MiniLM-DNN | small | 27.4% |
| BERT-DNN (frozen) | medium | 59.4% |
| BERT fine-tune (last 2 layers) | medium | see `bert-finetune/output/medium/stats.jsonl` |

## Requirements

```
transformers
sentence-transformers
torch
pandas
scikit-learn
pyarrow
tqdm
```

Python 3.10+. GPU required for BERT fine-tuning; other pipelines run on CPU.
