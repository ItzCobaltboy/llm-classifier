import pandas as pd
import gc
from utils import *

for size in ["small", "medium", "original"]:
    for split in ["train", "val", "test"]:
        print(f"Processing pad_n_truncate_upsample: {size} | {split}")
        df = pd.read_parquet(f'./Dataset/Dataset_{size}/{split}/{split}.parquet')
        
        df_final = upsample(remove_duplicates(pad_n_truncate(df)))
        
        out_dir = f"./Dataset_cleaned/pad_n_truncate_upsample/Dataset_{size}/{split}"
        save_data(df_final, out_dir, split)
        del df, df_final
        gc.collect()
