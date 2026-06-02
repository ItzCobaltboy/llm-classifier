# bert-dnn

Trains a DNN classifier on top of frozen sentence embeddings. Four embedding models are supported; embeddings are pre-generated and cached as `.pt` tensors so training runs fast on CPU.

## Files

| File | Description |
|------|-------------|
| `generate.py` | Encode all dataset splits with all four models, save `.pt` tensors |
| `train.py` | Load cached embeddings and train DNN |
| `classifier_models.py` | DNN and RNN model definitions + Classifier training wrapper |
| `embedding_models.py` | Reference snippets for loading each embedding model |
| `output/` | Training results per run (`stats.json`) |

## Embedding models

| Model | Dim |
|-------|-----|
| `Qwen/Qwen3-Embedding-0.6B` | 1536 |
| `intfloat/e5-base` | 768 |
| `google-bert/bert-base-uncased` (mean-pool) | 768 |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 |

## DNN architecture

Input -> 512 -> 256 -> 128 -> 12 (ReLU + Dropout 0.3 between layers)

## Usage

**Step 1: generate embeddings**

```bash
python generate.py
```

Reads from `../Dataset_cleaned/`, writes tensors to `embeddings/<size>/<split>/{qwen3,e5,bert,minilm}.pt`.

**Step 2: train**

```bash
python train.py
```

Edit `SIZES` and `EMBEDDINGS` at the top of `train.py` to select which combinations to run. Default: `EPOCHS = 100`.

Outputs per run go to `output/<size>_<embedding>_dnn/`: `best_model.pt`, `latest_model.pt`, `stats.json`.

## Results

| Embedding | Dataset | Test accuracy |
|-----------|---------|--------------|
| BERT (mean-pool) | small | 46.2% |
| E5-base | small | 33.2% |
| Qwen3-0.6B | small | 28.0% |
| MiniLM-L6 | small | 27.4% |
| BERT (mean-pool) | medium | 59.4% |
