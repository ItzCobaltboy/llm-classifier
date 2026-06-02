# bert-finetune

Fine-tunes BERT with the last 2 encoder layers unfrozen and a 5-layer DNN classification head. Uses mixed precision (fp16) and gradient accumulation for memory efficiency.

## Files

| File | Description |
|------|-------------|
| `train.py` | Training script |
| `output/medium/stats.jsonl` | Per-epoch stats for medium dataset run |
| `output/large/stats.jsonl` | Per-epoch stats for large dataset run |

## Model

- Base: `google-bert/bert-base-uncased`
- All layers frozen except the last 2 encoder layers
- Head: 768 -> 1024 -> 512 -> 256 -> 128 -> 12 (ReLU + Dropout 0.3)

## Training config

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch size | 8 (effective 32 via gradient accumulation) |
| LR (BERT layers) | 2e-5 |
| LR (head) | 1e-3 |
| LR schedule | CosineAnnealingLR |
| Mixed precision | fp16 (GPU only) |
| Max sequence length | 256 |

## Usage

Edit `SIZES` at the top of `train.py`, then:

```bash
python train.py
```

Saves `best_model.pt`, `latest_model.pt`, and appends per-epoch stats to `output/<run>/stats.jsonl`. Checkpoint is saved every epoch (overwrites).

## Notes

- Reads from `../Dataset_cleaned/` (columns: `generation`, `model`)
- `NUM_WORKERS = 0` is required on Windows with Python 3.14 due to multiprocessing limitations
