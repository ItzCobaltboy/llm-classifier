import pandas as pd
import os

MAX_WORDS = 500

#######_______Cleaning & Formatting_______#######
def remove_duplicates(df):
    return df.drop_duplicates(subset=['text'], keep='first').copy()

def clean_noise(df):
    df_cleaned = df.copy()
    df_cleaned['text'] = df_cleaned['text'].str.replace(r'<[^<>]*>', '', regex=True)
    df_cleaned['text'] = df_cleaned['text'].str.replace(r'http\S+|www\.\S+', '', regex=True)
    df_cleaned['text'] = df_cleaned['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df_cleaned

def lowercase_text(df):
    df_lower = df.copy()
    df_lower['text'] = df_lower['text'].str.lower()
    return df_lower

def remove_punctuation(df):
    df_nopunct = df.copy()
    df_nopunct['text'] = df_nopunct['text'].str.replace(r'[^\w\s]', '', regex=True)
    return df_nopunct

#######_______Sampling_______#######
def downsample(df):
    min_size = df["generated_by"].value_counts().min()
    return pd.concat([
        grp.sample(min_size, random_state=42) 
        for _, grp in df.groupby("generated_by")
    ]).reset_index(drop=True)

def upsample(df):
    max_size = df["generated_by"].value_counts().max()
    return pd.concat([
        grp.sample(max_size, replace=True, random_state=42) 
        for _, grp in df.groupby("generated_by")
    ]).reset_index(drop=True)

#######_______Core Transformations_______#######
def remove_outliers(df):
    word_counts = df['text'].str.split().str.len()
    Q1 = word_counts.quantile(0.25)
    Q3 = word_counts.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(word_counts >= lower_bound) & (word_counts <= upper_bound)].copy()

def truncate_only(df, max_words=MAX_WORDS):
    df_trunc = df.copy()
    df_trunc['text'] = df_trunc['text'].apply(lambda x: " ".join(str(x).split()[:max_words]))
    return df_trunc

def pad_n_truncate(df, max_words=MAX_WORDS, pad_token="<PAD>"):
    def adjust_text(text):
        words = str(text).split()
        if len(words) >= max_words:
            return " ".join(words[:max_words])
        else:
            padding_needed = max_words - len(words)
            return " ".join(words + [pad_token] * padding_needed)
    
    df_truncated = df.copy()
    df_truncated['text'] = df_truncated['text'].apply(adjust_text)
    return df_truncated

def split_texts(df, max_length=MAX_WORDS):
    def chunk_words(text):
        words = str(text).split()
        return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    
    df_split = df.copy()
    df_split['text'] = df_split['text'].apply(chunk_words)
    return df_split.explode('text', ignore_index=True)

#######_______Saving Helper_______#######
def save_data(df, path, split):
    os.makedirs(path, exist_ok=True)
    df.to_parquet(f"{path}/{split}.parquet", index=False)
