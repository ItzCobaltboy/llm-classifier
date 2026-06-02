# data

Scripts for cleaning, balancing, and exploring the dataset.

## Files

| File | Description |
|------|-------------|
| `clean.py` | Truncate to 300 words, downsample to balance classes, write to `../Dataset_cleaned/` |
| `change_columns.py` | Rename `generation`/`model` columns to `text`/`generated_by` (converts large-dataset format to small/medium format) |
| `eda_large.py` | EDA plots and stats for the large dataset (~991k rows) |
| `eda_medium.py` | Class distribution and word-count stats for small/medium datasets |

## Usage

**Clean and balance**

```bash
python clean.py
```

Edit `SIZES` at the top to select which dataset sizes to process. Outputs go to `../Dataset_cleaned/`.

**Convert column names**

Edit the `path` variable at the bottom of `change_columns.py`, then:

```bash
python change_columns.py
```

**EDA (large dataset)**

```bash
python eda_large.py
# optional: --data_root /path/to/Dataset_cleaned --out_dir ./output/eda --sample 5000
```

Outputs 11 plots and two JSON summary files to `./output/eda/`.
