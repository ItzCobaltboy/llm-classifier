"""
Step 1: Clean and balance the dataset.
- Truncate texts to MAX_WORDS to normalise length (human texts are very long)
- Downsample each class to the smallest class count for balance
- Saves cleaned parquets to Dataset_cleaned/
"""
import os
import pandas as pd

MAX_WORDS = 300
RAW_DIR   = "./../Dataset"
CLEAN_DIR = "./../Dataset_cleaned"
SIZES     = ["small", "medium"]
SPLITS    = ["train", "val", "test"]


def truncate(text: str, n: int = MAX_WORDS) -> str:
    return " ".join(text.split()[:n])


def balance(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample every class to the size of the smallest class."""
    n = df["generated_by"].value_counts().min()
    return (
        pd.concat([grp.sample(n, random_state=42) for _, grp in df.groupby("generated_by")])
          .reset_index(drop=True)
    )


for size in SIZES:
    for split in SPLITS:
        src = f"{RAW_DIR}/Dataset_{size}/{split}/{split}.parquet"
        df  = pd.read_parquet(src)
        print(df.head())

        df["text"] = df["text"].apply(truncate)
        df = balance(df)

        out_dir = f"{CLEAN_DIR}/Dataset_{size}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        df.to_parquet(f"{out_dir}/{split}.parquet", index=False)

        n_per_class = df["generated_by"].value_counts().iloc[0]
        print(f"[{size:6s}/{split:5s}] shape={str(df.shape):15s} samples/class={n_per_class}")
