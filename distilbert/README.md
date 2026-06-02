# distilbert

End-to-end fine-tuning of any HuggingFace sequence classification model using the `Trainer` API.

## Files

| File | Description |
|------|-------------|
| `train.py` | Fine-tune and save a model |
| `evaluate.py` | Run test-set evaluation and print classification report |
| `predict.py` | Interactive single-text prediction |
| `utils.py` | `load_data()`, `encode_labels()`, `tokenize_dataset()` helpers |
| `experiments.ipynb` | Exploratory training notebook |

## Usage

**Train**

```bash
python train.py [model_name]
# default: distilbert-base-uncased
```

Saves model to `models/<model_name>/`.

**Evaluate**

```bash
python evaluate.py models/<model_name>
```

**Predict**

```bash
python predict.py
# prompts: Enter text:
```

Loads from `models/bert`.

## Training config

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 8 |
| Max sequence length | 256 |
| fp16 | yes (if GPU available) |
| Checkpoints kept | 2 |

## Data

Reads `../Dataset/Dataset_small/` (columns: `text`, `generated_by`). Paths are relative to the repo root.

## Dependencies

```bash
pip install transformers datasets torch scikit-learn
```
