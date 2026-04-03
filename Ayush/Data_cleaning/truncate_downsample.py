import pandas as pd
import gc
from utils import *

for size in ["small", "medium", "original"]:
    for split in ["train", "val", "test"]:
        print(f"Processing truncate_downsample: {size} | {split}")
        df = pd.read_parquet(f'./Dataset/Dataset_{size}/{split}/{split}.parquet')
        
        df_final = downsample(remove_duplicates(truncate_only(df)))
        
        out_dir = f"./Dataset_cleaned/truncate_downsample/Dataset_{size}/{split}"
        save_data(df_final, out_dir, split)
        del df, df_final
        gc.collect()
