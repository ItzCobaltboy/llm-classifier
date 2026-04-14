"""
run_cleaning.py — Parameterized dataset cleaning and balancing.

Replaces the 8 individual *_downsample.py / *_upsample.py scripts.

Usage:
    python run_cleaning.py --transform truncate --sampling downsample
    python run_cleaning.py --transform split --sampling upsample
    python run_cleaning.py --transform pad_n_truncate --sampling downsample
    python run_cleaning.py --transform remove_outliers --sampling upsample

Transforms : truncate | pad_n_truncate | split | remove_outliers
Sampling   : downsample | upsample
"""

import argparse
import gc

import pandas as pd

from utils import (
    downsample, upsample, remove_duplicates,
    truncate_only, pad_n_truncate, split_texts, remove_outliers,
    save_data,
)

TRANSFORMS = {
    "truncate": truncate_only,
    "pad_n_truncate": pad_n_truncate,
    "split": split_texts,
    "remove_outliers": remove_outliers,
}

SAMPLERS = {
    "downsample": downsample,
    "upsample": upsample,
}


def main():
    parser = argparse.ArgumentParser(description="Clean and balance the dataset.")
    parser.add_argument(
        "--transform",
        choices=TRANSFORMS.keys(),
        required=True,
        help="Text length strategy to apply before deduplication.",
    )
    parser.add_argument(
        "--sampling",
        choices=SAMPLERS.keys(),
        required=True,
        help="Class-balance strategy applied after deduplication.",
    )
    args = parser.parse_args()

    transform_fn = TRANSFORMS[args.transform]
    sample_fn = SAMPLERS[args.sampling]
    name = f"{args.transform}_{args.sampling}"

    for size in ["small", "medium", "original"]:
        for split in ["train", "val", "test"]:
            print(f"Processing {name}: {size} | {split}")
            df = pd.read_parquet(f"./Dataset/Dataset_{size}/{split}/{split}.parquet")
            df_final = sample_fn(remove_duplicates(transform_fn(df)))
            out_dir = f"./Dataset_cleaned/{name}/Dataset_{size}/{split}"
            save_data(df_final, out_dir, split)
            del df, df_final
            gc.collect()


if __name__ == "__main__":
    main()
