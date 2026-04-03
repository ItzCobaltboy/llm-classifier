import pandas as pd
import os
import gc

RAW_DIR   = "./Dataset"
CLEAN_DIR = "./Dataset_cleaned"

SIZES     = ["small", "medium", "original"]
SPLITS    = ["train", "val", "test"]

#######_______Remove Outliers_______#######
def remove_outliers(df):
    word_counts = df['text'].str.split().str.len()
    Q1 = word_counts.quantile(0.25)
    Q3 = word_counts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return df[(word_counts >= lower_bound) & (word_counts <= upper_bound)].copy()

#######_______max word count per data = 500_______#######
def truncate(df, max_words=500):
    df_truncated = df.copy()
    df_truncated['text'] = df_truncated['text'].str.split().str[:max_words].str.join(" ")
    return df_truncated

def split_texts(df, max_length=500):
    df_split = df.copy()
    def chunk_words(text):
        words = text.split()
        return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    
    df_split['text'] = df_split['text'].apply(chunk_words)
    return df_split.explode('text', ignore_index=True)

#######_______Balancing the data (downsampling)_______#######
def balance(df):
    min_class_size = df["generated_by"].value_counts().min()
    return pd.concat([
        grp.sample(min_class_size, random_state=42) 
        for _, grp in df.groupby("generated_by")
    ]).reset_index(drop=True)


#######_______Making the transformations_______#######

for size in SIZES:
    for split in SPLITS:
        print(f"\n--- Processing [{size.upper()} | {split.upper()}] ---")
        
        # Load raw data
        df = pd.read_parquet(f'{RAW_DIR}/Dataset_{size}/{split}/{split}.parquet')
        
        # 1. OUTLIERS PIPELINE
        out_dir = f"{CLEAN_DIR}/remove_outliers/Dataset_{size}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        
        df_outliers = balance(remove_outliers(df))
        df_outliers.to_parquet(f"{out_dir}/{split}.parquet", index=False)
        print(f"Outliers Removed -> shape={df_outliers.shape}, samples/class={df_outliers['generated_by'].value_counts().iloc[0]}")
        del df_outliers
        
        # 2. TRUNCATE PIPELINE
        out_dir = f"{CLEAN_DIR}/truncate/Dataset_{size}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        
        df_trunc = balance(truncate(df))
        df_trunc.to_parquet(f"{out_dir}/{split}.parquet", index=False)
        print(f"Truncated        -> shape={df_trunc.shape}, samples/class={df_trunc['generated_by'].value_counts().iloc[0]}")
        del df_trunc
        
        # 3. SPLIT PIPELINE
        out_dir = f"{CLEAN_DIR}/split/Dataset_{size}/{split}"
        os.makedirs(out_dir, exist_ok=True)
        
        df_exp = balance(split_texts(df))
        df_exp.to_parquet(f"{out_dir}/{split}.parquet", index=False)
        print(f"Split/Exploded   -> shape={df_exp.shape}, samples/class={df_exp['generated_by'].value_counts().iloc[0]}")
        del df_exp
        
        del df 
        gc.collect()
