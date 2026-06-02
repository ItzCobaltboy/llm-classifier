import pandas as pd
import gc
gc.collect()
from utils import *

for size in ["small", "medium", "original"]:
    for split in ["train", "val", "test"]:
        print(f"Processing remove_outliers_downsample: {size} | {split}")
        df = pd.read_parquet(f'./Dataset/Dataset_{size}/{split}/{split}.parquet')

        #df = clean_noise(df)            # Remove URLs, HTML tags and extra whitespaces
        #df = lowercase_text(df)         # Convert to lowercase
        #df = remove_punctuation(df)     # Remove Punctuation marks
        
        df_final = downsample(remove_duplicates(remove_outliers(df)))
        
        out_dir = f"./Dataset_cleaned/remove_outliers_downsample/Dataset_{size}/{split}"
        save_data(df_final, out_dir, split)
        del df, df_final
        gc.collect()
