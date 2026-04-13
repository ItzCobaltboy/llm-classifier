import pandas as pd

SPLITS = ["train", "val", "test"]
SIZES  = ["small", "medium"]  # skipping large for now

for size in SIZES:
    print(f"\n{'='*50}")
    print(f"Dataset: {size.upper()}")
    print(f"{'='*50}")
    for split in SPLITS:
        path = f"./../Dataset/Dataset_{size}/{split}/{split}.parquet"
        df = pd.read_parquet(path)

        df["word_count"] = df["text"].str.split().str.len()

        print(f"\n--- {split} | shape: {df.shape} ---")
        print("Class distribution + word count stats:")
        stats = (
            df.groupby("generated_by")["word_count"]
            .agg(["count", "mean", "min", "max"])
            .rename(columns={"count": "n", "mean": "avg_words"})
            .astype({"n": int, "avg_words": int, "min": int, "max": int})
        )
        print(stats.to_string())
