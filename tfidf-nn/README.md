# tfidf-nn

TF-IDF feature extraction with a 2-layer dense neural network. Runs a grid search over 7 hidden-layer size configurations.

## Files

| File | Description |
|------|-------------|
| `train.py` | TF-IDF vectorization + DNN grid search |
| `notebooks/` | Jupyter notebooks: TF-IDF + NN and BERT variation experiments |
| `data_cleaning/` | Cleaning and resampling utilities |
| `output/` | Saved label encoder |

## Model

- TF-IDF: 1-4 gram, 50k features
- Network: input -> L1 -> ReLU -> BN -> Dropout -> L2 -> ReLU -> BN -> Dropout -> 12
- Layer configs searched: (512,256), (1024,512), (2048,1024), (4096,2048), (8192,4096), (16384,8192), (32768,16384)

## Usage

```bash
python train.py
```

Note: dataset paths in `train.py` are hardcoded to `D:\VSCode\llm-classifier\Dataset\Dataset_large\` -- update before running.

Saves per-epoch checkpoints under `output/checkpoints/<config>/` and picks the best config by validation loss.

## Data cleaning (`data_cleaning/`)

Place `utils.py` and a transformation script adjacent to your `Dataset/` folder and run the script. Each script applies one text transformation and one balancing strategy:

| Script | Transformation | Balancing |
|--------|---------------|-----------|
| `truncate_downsample.py` | Truncate to 500 words | Downsample |
| `truncate_upsample.py` | Truncate to 500 words | Upsample |
| `pad_n_truncate_downsample.py` | Truncate or pad to 500 words | Downsample |
| `pad_n_truncate_upsample.py` | Truncate or pad to 500 words | Upsample |
| `remove_outliers_downsample.py` | Remove IQR outliers | Downsample |
| `remove_outliers_upsample.py` | Remove IQR outliers | Upsample |
| `split_downsample.py` | Split into 500-word chunks | Downsample |
| `split_upsample.py` | Split into 500-word chunks | Upsample |

`MAX_WORDS` defaults to 500. Noise removal, lowercasing, and punctuation stripping are available but commented out in `utils.py`.
