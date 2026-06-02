# bert-rnn

Frozen BERT encoder feeding token-level hidden states into a bidirectional GRU classifier. Only the GRU and FC head are trained; BERT weights are never updated.

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script |

## Model

- Base: `google-bert/bert-base-uncased` (fully frozen)
- GRU: 2 layers, 256 hidden, bidirectional -> 512-dim output
- Head: Dropout -> 512 -> 256 -> 12 (ReLU)

## Training config

| Parameter | Value |
|-----------|-------|
| Dataset | small |
| Epochs | 50 |
| Batch size | 16 |
| LR | 1e-3 |
| Max sequence length | 256 |
| Checkpoint every | 10 epochs |

## Usage

```bash
python train.py
```

Reads from `../Dataset_cleaned/Dataset_small/`. Saves `best_model.pt`, `latest_model.pt`, and `stats.json` to `output/small_bert_rnn/`.
